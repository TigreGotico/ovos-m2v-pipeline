import itertools
import re
import time
import numpy as np
from typing import List, Optional, Union, Dict, Iterable, Tuple

from model2vec.inference import StaticModelPipeline
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline
from ovos_spec_tools import SpecMessage
from ovos_spec_tools.context import gate_satisfied
from ovos_spec_tools.expansion import expand as expand_template
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG

from ovos_m2v_pipeline.strategies import (
    PrototypeStrategy,
    select_anchors,
    score_labels,
)

#: Regex matching ``{slot}`` placeholders in OVOS-INTENT-1 template samples.
_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

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
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
    ) -> None:
        self.strategy: PrototypeStrategy = PrototypeStrategy(strategy)
        self.top_k: int = top_k
        self.tau: float = tau
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
        k: Optional[int] = None,
        random_state: int = 42,
    ) -> int:
        """Embed *sentences* and add/replace prototypes for *label*.

        ``k`` caps how many anchors the subsampling / clustering strategies
        keep; ``None`` (the default) keeps every sample so exact training
        samples always score a perfect match.

        Returns the number of prototypes actually added.
        """
        if not sentences:
            return 0
        self.remove(label)
        # Embed every sample first; the strategy decides which / how many
        # to keep as anchors at storage time.
        embs = np.atleast_2d(model.encode(list(sentences))).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / norms, embs)
        anchors = select_anchors(
            embs, self.strategy, k=k, random_state=random_state,
        ).astype(np.float32)
        n_added = len(anchors)
        if n_added == 0:
            return 0

        if not len(self):
            self._embeddings = anchors
            self._labels = np.array([label] * n_added, dtype=object)
        else:
            self._embeddings = np.vstack([self._embeddings, anchors])
            self._labels = np.concatenate(
                [self._labels, np.array([label] * n_added, dtype=object)]
            )
        return n_added

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
        return score_labels(
            q, self._embeddings, self._labels,
            self.strategy, top_k=self.top_k, tau=self.tau,
        )

    # ------------------------------------------------------------------
    # Bulk construction helpers (offline / testing)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model,
        sentences: List[str],
        labels: List[str],
        k: Optional[int] = None,
        random_state: int = 42,
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
    ) -> "PrototypeIntentStore":
        """Build a store from parallel *sentences* / *labels* lists."""
        store = cls(strategy=strategy, top_k=top_k, tau=tau)
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
    def load(
        cls,
        path: str,
        *,
        strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        top_k: int = 3,
        tau: float = 0.1,
    ) -> "PrototypeIntentStore":
        """Load from a NumPy ``.npz`` archive."""
        data = np.load(path, allow_pickle=False)
        return cls(
            data["embeddings"].astype(np.float32), data["labels"],
            strategy=strategy, top_k=top_k, tau=tau,
        )


def _parse_intent_file(path: str, ctx: str = "") -> List[str]:
    """Return expanded, non-empty, non-comment lines from a Padatious ``.intent`` file.

    Template syntax (``(a|b)``, ``[optional]``) is expanded via
    ``ovos_spec_tools.expansion.expand`` so that every concrete
    utterance variant is represented as a separate prototype.

    A line that is not parsable as OVOS-INTENT-1 grammar is logged and
    skipped so that a single bad template never discards the rest of the
    file (OVOS-INTENT-4 §6.3); *ctx* carries the §5.3 identifier fields
    for those warnings.
    """
    try:
        sentences: List[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.lstrip().startswith("#"):
                    try:
                        sentences.extend(expand_template(line))
                    except Exception as exc:
                        LOG.warning(f"skipping malformed template {line!r}: "
                                    f"{exc} {ctx}")
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
    ``prototype_k`` : int, optional (default: unlimited)
        Maximum number of prototype embeddings kept per label (used by the
        strategies that subsample / cluster — see ``prototype_strategy``).
        Unset (the default) keeps every registered sample as an anchor so
        that an exact training sample always scores a perfect match;
        set an integer to cap memory at the cost of recall.
    ``prototype_strategy`` : str, default ``"max_over_all"``
        One of the ``PrototypeStrategy`` values
        (``mean_centroid`` / ``medoid`` / ``max_over_all`` / ``top_k_mean`` /
        ``farthest_point`` / ``kmeans_centers`` / ``softmax_weighted``).
    ``prototype_top_k`` : int, default 3
        K for ``top_k_mean``.
    ``prototype_tau`` : float, default 0.1
        Softmax temperature for ``softmax_weighted``.
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
        #: Registered entity value-sets (OVOS-INTENT-4 §7), keyed by entity
        #: name (lowercase). Used to fill ``{slot}`` placeholders in template
        #: samples before embedding. Disabled (left empty) in classifier mode.
        self.entities: Dict[str, List[str]] = {}
        #: OVOS-CONTEXT-1 §6/§6.1 gating declarations per registered label,
        #: keyed by label -> (requires_context, excludes_context). Each list
        #: holds bare-string keys or ``{"key", "scope"}`` mappings and is
        #: evaluated at match time via ``gate_satisfied``.
        self._context_gates: Dict[str, Tuple[list, list]] = {}
        #: Per-label suppression phrases (OVOS-INTENT-4 §6.1 ``blacklist``),
        #: keyed by intent label. A candidate is dropped at match time when the
        #: utterance contains one of its label's blacklisted phrases. Named and
        #: matched consistently with the padacioso engine's ``excluded_keywords``.
        self.excluded_keywords: Dict[str, List[str]] = {}

        mode = self.config.get("mode", "classifier")

        if mode == "prototype":
            from model2vec import StaticModel

            self.model = StaticModel.from_pretrained(model_path)
            self._prototype_k: Optional[int] = self.config.get("prototype_k")
            self._prototype_strategy: PrototypeStrategy = PrototypeStrategy(
                self.config.get("prototype_strategy",
                                PrototypeStrategy.MAX_OVER_ALL.value)
            )
            self._prototype_top_k: int = self.config.get("prototype_top_k", 3)
            self._prototype_tau: float = self.config.get("prototype_tau", 0.1)
            self.prototype_store: Optional[PrototypeIntentStore] = PrototypeIntentStore(
                strategy=self._prototype_strategy,
                top_k=self._prototype_top_k,
                tau=self._prototype_tau,
            )

            self.bus.on("mycroft.ready", self._handle_ready_prototype)
            # Legacy registration topics (kept alongside OVOS-INTENT-4).
            self.bus.on("padatious:register_intent", self._handle_register_padatious)
            self.bus.on("register_intent", self._handle_register_adapt)
            self.bus.on("detach_intent", self._handle_detach_intent)
            self.bus.on("detach_skill", self._handle_detach_skill)
            # OVOS-INTENT-4 registration topics. m2v matches on example
            # utterances, so it is a TEMPLATE-style engine: it consumes
            # `ovos.intent.register.template` (§6) and NOT
            # `ovos.intent.register.keyword` (§11).
            self._wire_intent4_handlers()

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
            # OVOS-INTENT-4 registration topics. In classifier mode the model
            # is frozen, so registrations only (de)gate which trained classes
            # are eligible — exactly like the legacy manifest sync above.
            self._wire_intent4_handlers()

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

        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        ctx = (f"[skill_id={skill_id!r} name={name!r} "
               f"lang={message.data.get('lang')!r} topic={message.msg_type}]")

        inline = message.data.get("samples") or []
        if inline:
            sentences: List[str] = []
            for s in inline:
                try:
                    sentences.extend(expand_template(s))
                except Exception as exc:
                    LOG.warning(f"skipping malformed template {s!r}: {exc} {ctx}")
        else:
            file_name: str = message.data.get("file_name", "")
            sentences = _parse_intent_file(file_name, ctx) if file_name else []

        if not sentences:
            # zero valid templates -> the whole registration is malformed
            LOG.warning(f"rejecting registration: no valid template remains {ctx}")
            return
        try:
            n = self.prototype_store.add(
                self.model, name, sentences, k=self._prototype_k
            )
        except Exception as exc:
            LOG.error(f"Failed to add prototypes for Padatious intent "
                      f"'{name}': {exc}")
            return
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
    # OVOS-INTENT-4 registration (template method) — consumed in addition
    # to the legacy topics. m2v is a template-style engine (it matches on
    # example utterances), so it subscribes to `ovos.intent.register.template`
    # (§6) and `ovos.entity.register` (§7), plus the shared deregister /
    # enable / disable topics (§8). It deliberately does NOT consume
    # `ovos.intent.register.keyword` (§11 — keyword engines only).
    # ------------------------------------------------------------------

    def _wire_intent4_handlers(self) -> None:
        """Subscribe to the OVOS-INTENT-4 registration topics (§4)."""
        self.bus.on(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                    self._handle_intent4_register_template)
        self.bus.on(SpecMessage.ENTITY_REGISTER.value,
                    self._handle_intent4_register_entity)
        self.bus.on(SpecMessage.INTENT_DEREGISTER.value,
                    self._handle_intent4_deregister_intent)
        self.bus.on(SpecMessage.ENTITY_DEREGISTER.value,
                    self._handle_intent4_deregister_entity)
        self.bus.on(SpecMessage.SKILL_DEREGISTER.value,
                    self._handle_intent4_deregister_skill)
        self.bus.on(SpecMessage.INTENT_ENABLE.value,
                    self._handle_intent4_enable)
        self.bus.on(SpecMessage.INTENT_DISABLE.value,
                    self._handle_intent4_disable)

    @staticmethod
    def _intent4_label(message: Message) -> str:
        """Build the internal ``<skill_id>:<intent_name>`` label from §3.2 fields."""
        skill_id = message.data.get("skill_id") or message.context.get("skill_id", "")
        intent_name = message.data.get("intent_name", "")
        if not skill_id or not intent_name:
            return ""
        return f"{skill_id}:{intent_name}"

    def _intent4_warn(self, topic: str, message: Message, reason: str) -> None:
        """Log a malformed-registration rejection at WARN (§5.3 / §6.3 / §7.2)."""
        LOG.warning(
            f"rejecting {topic} registration: {reason} "
            f"[skill_id={message.data.get('skill_id')!r} "
            f"name={message.data.get('intent_name') or message.data.get('entity_name')!r} "
            f"lang={message.data.get('lang')!r}]"
        )

    def _expand_entities(self, samples: List[str]) -> List[str]:
        """Fill ``{slot}`` placeholders in template *samples* with registered
        entity values (OVOS-INTENT-4 §7). Samples without placeholders, or whose
        entity is unregistered, are passed through with the placeholder left
        literal (entities are an optional hint, §7).
        """
        if not self.entities:
            return list(samples)
        out: List[str] = []
        for tmpl in samples:
            slots = _SLOT_RE.findall(tmpl)
            if not slots:
                out.append(tmpl)
                continue
            slot_values: List[List[str]] = []
            for slot in slots:
                vals = self.entities.get(slot.lower())
                slot_values.append(vals if vals else ["{" + slot + "}"])
            for combo in itertools.product(*slot_values):
                filled = tmpl
                for slot, val in zip(slots, combo):
                    filled = filled.replace("{" + slot + "}", val, 1)
                out.append(filled.strip())
        return out

    def _handle_intent4_register_template(self, message: Message) -> None:
        """Register a template intent (§6): bracket-expand + entity-fill the
        ``samples`` and embed them as prototypes (prototype mode) or track the
        label name (classifier mode)."""
        topic = SpecMessage.INTENT_REGISTER_TEMPLATE.value
        label = self._intent4_label(message)
        if not label:
            self._intent4_warn(topic, message, "missing skill_id or intent_name")
            return
        if label in self.ignore_labels:
            return
        samples = message.data.get("samples")
        if not samples:  # missing or empty -> malformed (§6.3)
            self._intent4_warn(topic, message, "samples missing or empty")
            return

        blacklist = message.data.get("blacklist")
        if blacklist:  # §6.1 suppression phrases: drop matches containing these
            self.excluded_keywords[label] = list(blacklist)

        # Classifier mode is frozen: only gate the (already trained) label.
        if self.prototype_store is None:
            self.intents.add(label)
            self._store_context_gate(label, message)
            LOG.debug(f"Model2Vec: tracking INTENT-4 template label '{label}'")
            return

        expanded: List[str] = []
        for s in self._expand_entities(list(samples)):
            try:
                expanded.extend(expand_template(s))
            except Exception as exc:
                # skip the unparsable template, keep the valid ones (§6.3)
                LOG.warning(
                    f"skipping malformed template {s!r}: {exc} "
                    f"[skill_id={message.data.get('skill_id')!r} "
                    f"name={message.data.get('intent_name')!r} "
                    f"lang={message.data.get('lang')!r} topic={topic}]")
        expanded = [s for s in (e.strip() for e in expanded) if s]
        if not expanded:  # zero non-empty expansions -> malformed (§6.3)
            self._intent4_warn(topic, message, "samples expand to zero non-empty templates")
            return
        n = self.prototype_store.add(self.model, label, expanded, k=self._prototype_k)
        self.intents.add(label)
        self._store_context_gate(label, message)
        LOG.debug(f"Prototype store: added {n} prototype(s) for INTENT-4 template '{label}'")

    def _store_context_gate(self, label: str, message: Message) -> None:
        """Record OVOS-CONTEXT-1 §6/§6.1 ``requires_context`` /
        ``excludes_context`` declarations for ``label`` (if any) so they can be
        enforced at match time. Absent declarations clear any stale entry."""
        requires = message.data.get("requires_context")
        excludes = message.data.get("excludes_context")
        if requires or excludes:
            self._context_gates[label] = (requires or [], excludes or [])
        else:
            self._context_gates.pop(label, None)

    def _handle_intent4_register_entity(self, message: Message) -> None:
        """Register an entity value-set hint (§7). No-op in classifier mode."""
        topic = SpecMessage.ENTITY_REGISTER.value
        name: str = message.data.get("entity_name", "")
        samples = message.data.get("samples")
        if not name:
            self._intent4_warn(topic, message, "missing entity_name")
            return
        if not samples:  # missing or empty -> malformed (§7.2)
            self._intent4_warn(topic, message, "samples missing or empty")
            return
        if self.prototype_store is None:
            return  # classifier model is frozen; no slot-fill to perform
        values: set = set()
        for s in samples:
            try:
                for v in expand_template(s):
                    v = v.strip()
                    if v:
                        values.add(v)
            except Exception as exc:
                # skip the unparsable entry, keep the valid ones (§7.2)
                LOG.warning(
                    f"skipping malformed entity sample {s!r}: {exc} "
                    f"[skill_id={message.data.get('skill_id')!r} "
                    f"name={name!r} "
                    f"lang={message.data.get('lang')!r} topic={topic}]")
        if not values:  # zero valid entries -> malformed (§7.2)
            self._intent4_warn(topic, message, "no valid entity sample remains")
            return
        self.entities[name.lower()] = list(values)
        LOG.debug(f"Model2Vec: registered INTENT-4 entity '{name}' ({len(values)} values)")

    def _handle_intent4_deregister_intent(self, message: Message) -> None:
        """Remove one intent (§8.2)."""
        label = self._intent4_label(message)
        if not label:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove(label)
        self.intents.discard(label)
        self._context_gates.pop(label, None)
        self.excluded_keywords.pop(label, None)
        LOG.debug(f"Model2Vec: deregistered INTENT-4 intent '{label}'")

    def _handle_intent4_deregister_entity(self, message: Message) -> None:
        """Remove one entity (§8.3). No-op in classifier mode."""
        name: str = message.data.get("entity_name", "")
        if name:
            self.entities.pop(name.lower(), None)
            LOG.debug(f"Model2Vec: deregistered INTENT-4 entity '{name}'")

    def _handle_intent4_deregister_skill(self, message: Message) -> None:
        """Remove every intent and entity owned by a skill (§8.4)."""
        skill_id: str = message.data.get("skill_id") or message.context.get("skill_id", "")
        if not skill_id:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove_skill(skill_id)
        self.intents = {i for i in self.intents if not i.startswith(skill_id + ":")}
        self._context_gates = {l: g for l, g in self._context_gates.items()
                               if not l.startswith(skill_id + ":")}
        self.excluded_keywords = {i: kw for i, kw in self.excluded_keywords.items()
                                  if not i.startswith(skill_id + ":")}
        LOG.debug(f"Model2Vec: deregistered all INTENT-4 registrations for skill '{skill_id}'")

    def _handle_intent4_disable(self, message: Message) -> None:
        """Suppress an intent without losing its definition (§8.5).

        m2v keeps no separate enabled/disabled flag; suppression is realised
        by dropping the label from the match-eligible set (and its prototypes),
        mirroring deregistration. Re-enabling requires re-registration, which
        is how skills re-arm intents on this engine.
        """
        label = self._intent4_label(message)
        if not label:
            return
        if self.prototype_store is not None:
            self.prototype_store.remove(label)
        self.intents.discard(label)
        self._context_gates.pop(label, None)
        self.excluded_keywords.pop(label, None)
        LOG.debug(f"Model2Vec: disabled INTENT-4 intent '{label}'")

    def _handle_intent4_enable(self, message: Message) -> None:
        """Re-arm a previously disabled intent (§8.5).

        In classifier mode the trained class is re-added to the eligible set.
        In prototype mode the prototypes were dropped on disable and can only
        be restored by re-registration, so this only restores label tracking.
        """
        label = self._intent4_label(message)
        if label and label not in self.ignore_labels:
            self.intents.add(label)
            LOG.debug(f"Model2Vec: enabled INTENT-4 intent '{label}'")

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

    def _excluded_labels(self, utterance: str) -> List[str]:
        """Labels whose §6.1 ``blacklist`` phrases occur in *utterance*.

        Uses the same word-boundary convention as the padacioso engine's
        ``_filter``: single-word phrases must match a whole token, multi-word
        phrases match on a ``\\b``-delimited substring.
        """
        if not self.excluded_keywords:
            return []
        excluded: List[str] = []
        q_lower = utterance.lower()
        query_words = set(q_lower.split())
        for label, phrases in self.excluded_keywords.items():
            def _kw_hit(kw: str, _qw=query_words, _ql=q_lower) -> bool:
                kw = kw.lower()
                if " " not in kw:
                    return kw in _qw
                return bool(re.search(r"\b" + re.escape(kw) + r"\b", _ql))
            if any(_kw_hit(p) for p in phrases):
                excluded.append(label)
        return excluded

    def _session_blacklists(self, message: Optional[Message]) -> Tuple[frozenset, frozenset]:
        """``(blacklisted_intents, blacklisted_skills)`` for the caller's session.

        Mirrors adapt/padatious: candidates whose intent label or skill_id is
        blacklisted in the session are dropped at match time.
        """
        if message is None:
            return frozenset(), frozenset()
        try:
            sess = SessionManager.get(message)
        except Exception:
            return frozenset(), frozenset()
        return (frozenset(sess.blacklisted_intents or []),
                frozenset(sess.blacklisted_skills or []))

    def _match(self, utterance: str,
               message: Optional[Message] = None) -> Iterable[Tuple[str, str, float]]:
        """Yield ``(skill_id, label, score)`` tuples sorted by score descending.

        Candidates suppressed by a §6.1 template ``blacklist`` phrase, or by the
        session's ``blacklisted_intents`` / ``blacklisted_skills``, are dropped.

        Args:
            utterance: The utterance to match.
            message: The incoming bus message (used to read session.pipeline for
                     special-label gating, plus the session blacklists).
        """
        if self.prototype_store is not None:
            candidates = self._match_prototype(utterance, message)
        else:
            candidates = self._match_classifier(utterance, message)
        # OVOS-CONTEXT-1 §6/§6.1 context gate + OVOS-INTENT-4 §6.1 blacklist +
        # session blacklists — applied together so a candidate must survive all
        # three to be yielded.
        intent_context = (SessionManager.get(message).intent_context or {}) if self._context_gates else {}
        excluded = self._excluded_labels(utterance)
        blacklisted_intents, blacklisted_skills = self._session_blacklists(message)
        for skill_id, label, score in candidates:
            gate = self._context_gates.get(label)
            if gate is not None:
                requires, excludes = gate
                if not gate_satisfied(intent_context, requires, excludes, owner_id=skill_id):
                    LOG.debug(f"discarding '{label}': CONTEXT-1 gate not satisfied")
                    continue
            if label in excluded:
                LOG.debug(f"discarding match: {label} - utterance hits §6.1 blacklist")
                continue
            if label in blacklisted_intents or skill_id in blacklisted_skills:
                LOG.debug(f"discarding match: {label} - blacklisted in session")
                continue
            yield skill_id, label, score

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
