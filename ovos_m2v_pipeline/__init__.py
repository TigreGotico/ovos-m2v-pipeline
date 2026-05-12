import time
import numpy as np
from typing import List, Optional, Union, Dict, Iterable, Tuple

from model2vec.inference import StaticModelPipeline
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline
from ovos_utils.bracket_expansion import expand_template
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

# Labels that bypass the registered-intent check and are always matched
_SPECIAL_LABELS = {"ocp:play", "common_query:common_query", "stop:stop"}

# Map each special label to a substring that must appear in `session.pipeline`
# entries for the label to be considered. Without the matching downstream
# pipeline in the session there is no service to route the intent to.
_SPECIAL_LABEL_PIPELINES = {
    "ocp:play": "ovos-ocp-pipeline-plugin",
    "common_query:common_query": "ovos-common-query-pipeline-plugin",
    "stop:stop": "ovos-stop-pipeline-plugin",
}


class PrototypeIntentStore:
    """Mutable store of L2-normalised prototype embeddings per intent label.

    Inference is cosine nearest-neighbour: for each label the highest cosine
    similarity across its prototypes is used as the match score.

    The store starts empty and is populated incrementally at runtime as skills
    register their Padatious intents (which carry example utterances).

    Typical usage
    -------------
    Build from lists (offline / testing)::

        store = PrototypeIntentStore()
        store.add(model, "skill:intent", ["example 1", "example 2"], k=5)

    Persist and reload (optional)::

        store.save("prototypes.npz")
        store = PrototypeIntentStore.load("prototypes.npz")
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if embeddings is not None and len(embeddings):
            embeddings = np.atleast_2d(embeddings)
            labels_arr = np.asarray(labels, dtype=object)
            if len(embeddings) != len(labels_arr):
                raise ValueError(
                    f"embeddings and labels must have the same length, "
                    f"got {len(embeddings)} vs {len(labels_arr)}"
                )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._embeddings: np.ndarray = np.where(
                norms > 0, embeddings / norms, embeddings
            ).astype(np.float32)
            self._labels: np.ndarray = labels_arr
        else:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self._labels = np.array([], dtype=object)

    # ------------------------------------------------------------------
    # Public read-only views
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def unique_labels(self) -> np.ndarray:
        return np.unique(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        model,
        label: str,
        sentences: List[str],
        k: int = 5,
        random_state: int = 42,
    ) -> int:
        """Embed up to *k* sentences and add/replace prototypes for *label*.

        Returns the number of prototypes actually added.
        """
        if not sentences:
            return 0
        self.remove(label)
        rng = np.random.default_rng(random_state)
        chosen = (
            rng.choice(sentences, k, replace=False).tolist()
            if len(sentences) > k
            else sentences
        )
        embs = np.atleast_2d(model.encode(chosen)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / norms, embs)

        if not len(self):
            self._embeddings = embs
            self._labels = np.array([label] * len(chosen), dtype=object)
        else:
            self._embeddings = np.vstack([self._embeddings, embs])
            self._labels = np.concatenate(
                [self._labels, np.array([label] * len(chosen), dtype=object)]
            )
        return len(chosen)

    def remove(self, label: str) -> None:
        """Remove all prototypes for *label*."""
        if not len(self):
            return
        mask = self._labels != label
        self._embeddings = self._embeddings[mask]
        self._labels = self._labels[mask]

    def remove_skill(self, skill_id: str) -> None:
        """Remove all prototypes whose label starts with ``<skill_id>:``."""
        if not len(self):
            return
        prefix = skill_id + ":"
        mask = np.array([not str(lbl).startswith(prefix) for lbl in self._labels])
        self._embeddings = self._embeddings[mask]
        self._labels = self._labels[mask]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Return the max cosine similarity per label.

        Parameters
        ----------
        query_embedding:
            Raw (unnormalised) embedding vector of shape ``(dim,)``.
        """
        if not len(self):
            return {}
        q = query_embedding.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = self._embeddings @ q  # (n_prototypes,)
        label_scores: Dict[str, float] = {}
        for lbl, sim in zip(self._labels, sims):
            lbl = str(lbl)
            if lbl not in label_scores or sim > label_scores[lbl]:
                label_scores[lbl] = float(sim)
        return label_scores

    # ------------------------------------------------------------------
    # Bulk construction helpers (offline / testing)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model,
        sentences: List[str],
        labels: List[str],
        k: int = 5,
        random_state: int = 42,
    ) -> "PrototypeIntentStore":
        """Build a store from parallel *sentences* / *labels* lists."""
        store = cls()
        label_to_sentences: Dict[str, List[str]] = {}
        for sent, lbl in zip(sentences, labels):
            label_to_sentences.setdefault(lbl, []).append(sent)
        for lbl, sents in label_to_sentences.items():
            store.add(model, lbl, sents, k=k, random_state=random_state)
        return store

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to a NumPy ``.npz`` archive."""
        np.savez(path, embeddings=self._embeddings, labels=self._labels.astype(str))
        LOG.info(
            f"Saved {len(self)} prototypes "
            f"for {len(self.unique_labels)} labels -> {path}"
        )

    @classmethod
    def load(cls, path: str) -> "PrototypeIntentStore":
        """Load from a NumPy ``.npz`` archive."""
        data = np.load(path, allow_pickle=False)
        return cls(data["embeddings"].astype(np.float32), data["labels"])


def _parse_intent_file(path: str) -> List[str]:
    """Return expanded, non-empty, non-comment lines from a Padatious ``.intent`` file.

    Template syntax (``(a|b)``, ``[optional]``) is expanded via
    ``ovos_utils.bracket_expansion.expand_template`` so that every concrete
    utterance variant is represented as a separate prototype.
    """
    try:
        sentences: List[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.lstrip().startswith("#"):
                    sentences.extend(expand_template(line))
        return sentences
    except OSError as exc:
        LOG.warning(f"Could not read intent file '{path}': {exc}")
        return []


class Model2VecIntentPipeline(ConfidenceMatcherPipeline):
    """A pipeline that integrates Model2Vec with OVOS for intent matching.

    Two operating modes are supported via ``config["mode"]``:

    ``"classifier"`` (default)
        Loads a ``StaticModelPipeline`` (embedding model + trained linear
        classifier head).  Scores are softmax probabilities.

    ``"prototype"``
        Loads a bare ``StaticModel`` (embeddings only, **no training step**).
        At runtime, builds a ``PrototypeIntentStore`` from the example
        utterances supplied by Padatious intent registrations.  Scores are
        cosine similarities.

        Padatious ``.intent`` files are read directly when
        ``padatious:register_intent`` fires; the file path is taken from
        ``message.data["file_name"]``.  Adapt intents are tracked by label
        name only and are not matched in prototype mode.

    Configuration keys (prototype mode)
    ------------------------------------
    ``prototype_k`` : int, default 5
        Maximum number of prototype embeddings kept per label.
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        config = config or Configuration().get('intents', {}).get("ovos_m2v_pipeline") or dict()
        super().__init__(bus, config)
        model_path = self.config.get("model", "Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2")
        if not model_path:
            raise FileNotFoundError("'model' not set in configuration for ovos_m2v_pipeline")

        self.intents: set = set()
        self.ignore_labels: List[str] = self.config.get("ignore_intents") or []
        self._syncing = False

        mode = self.config.get("mode", "classifier")

        if mode == "prototype":
            from model2vec import StaticModel

            self.model = StaticModel.from_pretrained(model_path)
            self.prototype_store: Optional[PrototypeIntentStore] = PrototypeIntentStore()
            self._prototype_k: int = self.config.get("prototype_k", 5)

            self.bus.on("mycroft.ready", self._handle_ready_prototype)
            self.bus.on("padatious:register_intent", self._handle_register_padatious)
            self.bus.on("register_intent", self._handle_register_adapt)
            self.bus.on("detach_intent", self._handle_detach_intent)
            self.bus.on("detach_skill", self._handle_detach_skill)

            LOG.info(
                f"Loaded Model2VecIntents pipeline (prototype mode) "
                f"with model: '{model_path}'"
            )
        else:
            # Load the model
            self.model = StaticModelPipeline.from_pretrained(model_path)
            self.prototype_store = None

            # Register event handlers for intent synchronization
            self.bus.on("mycroft.ready", self.handle_sync_intents)
            self.bus.on("padatious:register_intent", self.handle_sync_intents)
            self.bus.on("register_intent", self.handle_sync_intents)
            self.bus.on("detach_intent", self.handle_sync_intents)
            self.bus.on("detach_skill", self.handle_sync_intents)

            LOG.info(f"Loaded Model2VecIntents pipeline with model: '{model_path}'")

            # Seed the intent allowlist from skills loaded before this pipeline.
            # Bus events keep it in sync for skills that load/unload later.
            self._initial_intent_sync()

    def _initial_intent_sync(self) -> None:
        """Query adapt + padatious manifests once at startup.

        Skills loaded *before* this pipeline never emit `register_intent`, so
        without this pull they would stay invisible until the next
        `mycroft.ready` / register / detach event.
        """
        timeout = self.config.get("timeout", 1)
        adapt: List[str] = []
        padatious: List[str] = []
        try:
            adapt = self._get_adapt_intents(timeout)
        except RuntimeError:
            LOG.debug("Model2Vec: adapt manifest not available at startup")
        try:
            padatious = self._get_padatious_intents(timeout)
        except RuntimeError:
            LOG.debug("Model2Vec: padatious manifest not available at startup")
        if adapt or padatious:
            self.intents = list(set(adapt + padatious))
            LOG.debug(f"Model2Vec seeded {len(self.intents)} intents on startup")

    # ------------------------------------------------------------------
    # Classifier-mode intent synchronisation
    # ------------------------------------------------------------------

    def _get_adapt_intents(self, timeout: int = 1) -> List[str]:
        """
        Retrieves Adapt intent names from the message bus, excluding ignored labels.

        Args:
            timeout: Maximum time in seconds to wait for the response.

        Returns:
            A list of Adapt intent names excluding ignored labels.

        Raises:
            RuntimeError: If no response is received from the bus.
        """
        msg = Message("intent.service.adapt.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.adapt.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i["name"] for i in res.data["intents"] if i["name"] not in self.ignore_labels]

    def _get_padatious_intents(self, timeout: int = 1) -> List[str]:
        """
        Retrieves Padatious intent names from the message bus, excluding ignored labels.

        Args:
            timeout: Maximum time in seconds to wait for the response.

        Returns:
            A list of Padatious intent names not present in the ignore list.

        Raises:
            RuntimeError: If no response is received from the bus.
        """
        msg = Message("intent.service.padatious.manifest.get")
        res = self.bus.wait_for_response(msg, "intent.service.padatious.manifest", timeout=timeout)
        if not res:
            raise RuntimeError("Failed to retrieve intent names")
        return [i for i in res.data["intents"] if i not in self.ignore_labels]

    def handle_sync_intents(self, message: Message) -> None:
        """
        Synchronizes registered intents when new skills are loaded or existing ones are detached.

        Args:
            message: The message that triggered intent synchronization.
        """
        # Sync newly (de)registered intents with debounce
        if self._syncing:
            return
        self._syncing = True
        try:
            time.sleep(3)
            timeout = self.config.get("timeout", 1)
            self.intents = set(
                self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout)
            )
            LOG.debug(f"Model2Vec registered intents: {len(self.intents)}")
        except RuntimeError:
            pass
        finally:
            self._syncing = False

    def _allowed_special_labels(self, message: Optional[Message]) -> set:
        """Return the special labels enabled by the caller's session pipeline.

        `ocp:play`, `common_query:common_query`, and `stop:stop` are only
        meaningful when their respective downstream pipelines are present in
        ``session.pipeline``.  If we cannot determine the session (no message
        or no session in context) we fall back to all special labels so the
        plugin keeps working in headless / test contexts.
        """
        if message is None:
            return set(_SPECIAL_LABELS)
        try:
            sess = SessionManager.get(message)
            sess_pipeline = list(sess.pipeline or [])
        except Exception:
            return set(_SPECIAL_LABELS)
        if not sess_pipeline:
            return set(_SPECIAL_LABELS)
        allowed = set()
        for label, needle in _SPECIAL_LABEL_PIPELINES.items():
            if any(needle in p for p in sess_pipeline):
                allowed.add(label)
        return allowed

    # ------------------------------------------------------------------
    # Prototype-mode intent registration
    # ------------------------------------------------------------------

    def _handle_ready_prototype(self, message: Message) -> None:
        LOG.info(
            f"Model2Vec prototype store ready: {len(self.prototype_store)} prototypes "
            f"for {len(self.prototype_store.unique_labels)} labels"
        )

    def _handle_register_padatious(self, message: Message) -> None:
        """Build prototypes from a Padatious ``.intent`` file or inline samples.

        ``message.data["samples"]`` (pre-expanded list) is preferred when
        present; otherwise the file at ``message.data["file_name"]`` is read
        and its template syntax is expanded via ``expand_template``.
        """
        name: str = message.data.get("name", "")
        if not name or name in self.ignore_labels:
            return

        inline = message.data.get("samples") or []
        if inline:
            sentences: List[str] = []
            for s in inline:
                sentences.extend(expand_template(s))
        else:
            file_name: str = message.data.get("file_name", "")
            sentences = _parse_intent_file(file_name) if file_name else []

        if not sentences:
            LOG.warning(f"No examples found for Padatious intent '{name}' - skipping prototypes")
            return
        n = self.prototype_store.add(
            self.model, name, sentences, k=self._prototype_k
        )
        self.intents.add(name)
        LOG.debug(f"Prototype store: added {n} prototype(s) for '{name}'")

    def _handle_register_adapt(self, message: Message) -> None:
        """Track Adapt intent labels (no example sentences -> no prototypes)."""
        name: str = message.data.get("name", "")
        if name and name not in self.ignore_labels:
            self.intents.add(name)

    def _handle_detach_intent(self, message: Message) -> None:
        name: str = message.data.get("intent_name", "")
        if name:
            self.prototype_store.remove(name)
            self.intents.discard(name)
            LOG.debug(f"Prototype store: removed prototypes for '{name}'")

    def _handle_detach_skill(self, message: Message) -> None:
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        if skill_id:
            self.prototype_store.remove_skill(skill_id)
            self.intents = {i for i in self.intents if not i.startswith(skill_id + ":")}
            LOG.debug(f"Prototype store: removed prototypes for skill '{skill_id}'")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_special_label_map(label: str) -> Tuple[str, str]:
        """Return ``(skill_id, canonical_label)`` after OCP / stop / query remaps."""
        if label == "ocp:play":
            return "ovos.common_play", "ovos.common_play.play_search"
        if label == "common_query:common_query":
            return "common_query.openvoiceos", "common_query.question"
        if label == "stop:stop":
            return "stop.openvoiceos", "mycroft.stop"
        return label.split(":")[0], label

    def _match(self, utterance: str,
               message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Yield ``(skill_id, label, score)`` tuples sorted by score descending.

        Args:
            utterance: The utterance to match.
            message: The incoming bus message (used to read session.pipeline for
                     special-label gating in both classifier and prototype modes).
        """
        if self.prototype_store is not None:
            yield from self._match_prototype(utterance, message)
        else:
            yield from self._match_classifier(utterance, message)

    def _match_classifier(self, utterance: str,
                           message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Classifier-mode matching using softmax probabilities."""
        inputs = [utterance]
        probs_ = self.model.predict_proba(inputs)
        # Include special-case labels gated by the session's active pipelines.
        special = self._allowed_special_labels(message)
        mask = np.isin(self.model.classes_, list(self.intents) + list(special))
        if not mask.any():
            LOG.warning("No model classes match registered intents")
            return
        classes = self.model.classes_[mask]
        probs = probs_[:, mask]
        # Renormalize probs over the surviving subset
        if self.config.get("renormalize"):
            row_sum = probs.sum(axis=1, keepdims=True)
            probs = np.where(row_sum > 0, probs / row_sum, probs)

        # Associate predictions with labels
        for input_text, prob_row in zip(inputs, probs):
            # Zip together class labels with their probabilities
            class_probs = list(zip(classes, prob_row))
            # Sort by probability descending
            class_probs.sort(key=lambda x: x[1], reverse=True)
            for label, prob in class_probs:
                LOG.debug(f"Match candidate: {label} - prob: {prob}")

                # HACK: special case for OCP, it isnt a regular intent
                skill_id, label = self._apply_special_label_map(label)

                yield skill_id, label, float(prob)

    def _match_prototype(self, utterance: str,
                         message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Cosine nearest-neighbour against the prototype store.

        Special labels (ocp:play, common_query:common_query, stop:stop) are
        only forwarded when the matching downstream pipeline is present in the
        caller's session, consistent with classifier mode.
        """
        emb = self.model.encode([utterance])[0]
        label_scores = self.prototype_store.scores(emb)
        special = self._allowed_special_labels(message)
        for label, score in sorted(label_scores.items(), key=lambda x: x[1], reverse=True):
            LOG.debug(f"Match candidate: {label} - cosine: {score:.4f}")
            if label in self.ignore_labels:
                continue
            # Gate special labels the same way as classifier mode
            if label in _SPECIAL_LABELS and label not in special:
                LOG.debug(f"discarding special label: {label} - not in session pipeline")
                continue
            skill_id, label = self._apply_special_label_map(label)
            yield skill_id, label, score

    # ------------------------------------------------------------------
    # Confidence-tier API
    # ------------------------------------------------------------------

    def match_high(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a high-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_high", 0.7)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match candidate: {match}")
            return match
        return None

    def match_medium(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a medium-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_medium", 0.5)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match: {match}")
            return match
        return None

    def match_low(self, utterances: List[str], lang: str, message: Message) -> Optional[IntentHandlerMatch]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a low-confidence match is found, None otherwise.
        """
        if not utterances:
            return None
        min_conf = self.config.get("conf_low", 0.15)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0], message):
            if prob < min_conf:
                LOG.debug(f"discarding match: {label} - confidence < {min_conf}")
                return None
            match = IntentHandlerMatch(
                match_type=label,
                match_data={"utterance": utterances[0], "confidence": prob},
                skill_id=skill_id or "ovos-m2v-pipeline",
                utterance=utterances[0]
            )
            LOG.debug(f"Match candidate: {match}")
            return match
        return None


class Model2VecPrototypePipeline(Model2VecIntentPipeline):
    """Prototype-mode Model2Vec pipeline exposed as a standalone OVOS plugin.

    Identical to ``Model2VecIntentPipeline`` with ``mode`` forced to
    ``"prototype"``.  Configuration is read from
    ``intents.ovos_m2v_prototype_pipeline`` so it can coexist with the
    classifier plugin in the same OVOS instance.

    Example ``mycroft.conf``::

        "intents": {
            "ovos-m2v-pipeline": {
                "model": "Jarbas/ovos-model2vec-intents-LaBSE"
            },
            "ovos-m2v-prototype-pipeline": {
                "model": "minishlab/M2V_multilingual_output",
                "prototype_k": 5
            }
        }
    """

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if config is None:
            config = (
                Configuration().get("intents", {}).get("ovos_m2v_prototype_pipeline") or {}
            )
        config["mode"] = "prototype"
        super().__init__(bus, config)
