import time
from typing import List, Optional, Union, Dict, Iterable, Tuple

from model2vec.inference import StaticModelPipeline
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline
from ovos_utils.fakebus import FakeBus
from ovos_utils.log import LOG


class Model2VecIntentPipeline(ConfidenceMatcherPipeline):
    """
    A pipeline that integrates Model2Vec with OVOS for intent matching.

    This class uses Model2Vec for intent recognition, only considering loaded Adapt and Padatious intents
    retrieved from the message bus.
    """

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        """
        Initializes the Model2VecIntentPipeline with configuration and event handlers.

        Args:
            bus: The message bus client used to interact with the system.
            config: Optional configuration dictionary for setting pipeline options.

        Registers message bus event handlers to synchronize intents on relevant system events.
        """
        config = config or Configuration().get('intents', {}).get("ovos_m2v_pipeline") or dict()
        super().__init__(bus, config)
        model_path = self.config.get("model", "Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2")
        if not model_path:
            raise FileNotFoundError("'model' not set in configuration for ovos_m2v_pipeline")

        # Load the model
        self.model = StaticModelPipeline.from_pretrained(model_path)

        self.intents = []
        self.ignore_labels = self.config.get("ignore_intents") or []

        LOG.info(f"Loaded Model2VecIntents pipeline with model: '{model_path}'")

        # Register event handlers for intent synchronization
        self.bus.on("mycroft.ready", self.handle_sync_intents)
        self.bus.on("padatious:register_intent", self.handle_sync_intents)
        self.bus.on("register_intent", self.handle_sync_intents)
        self.bus.on("detach_intent", self.handle_sync_intents)
        self.bus.on("detach_skill", self.handle_sync_intents)

        self._syncing = False

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
        time.sleep(3)
        timeout = self.config.get("timeout", 1)
        try:
            self.intents = set(self._get_adapt_intents(timeout) + self._get_padatious_intents(timeout))
            LOG.debug(f"Model2Vec registered intents: {len(self.intents)}")
        except RuntimeError:
            pass
        self._syncing = False

    def _match(self, utterance: str) -> Iterable[Tuple[str, str, float]]:
        """
        Matches the most likely intent for a given list of utterances using Model2Vec.

        Args:
            utterances: A list of utterances to match against the model.
            lang: The language of the input utterance.
            message: The incoming message containing additional context.

        Returns:
            An IntentHandlerMatch if a high-confidence match is found, None otherwise.
        """
        inputs = [utterance]
        probs = self.model.predict_proba(inputs)

        # Associate predictions with labels
        for input_text, prob_row in zip(inputs, probs):
            # Zip together class labels with their probabilities
            class_probs = list(zip(self.model.classes_, prob_row))
            # Sort by probability descending
            class_probs.sort(key=lambda x: x[1], reverse=True)
            for label, prob in class_probs:
                LOG.debug(f"Match candidate: {label} - prob: {prob}")

                # HACK: special case for OCP, it isnt a regular intent
                skill_id = label.split(":")[0]
                if label == "ocp:play":
                    skill_id = "ovos.common_play"
                    label = "ovos.common_play.play_search"
                elif label == "common_query:common_query":
                    skill_id = "common_query.openvoiceos"
                    label = "common_query.question"
                elif label == "stop:stop":
                    skill_id = "stop.openvoiceos"
                    label = "mycroft.stop"
                elif label not in self.intents:
                    LOG.debug(f"discarding match: {label} - intent not detected at runtime")
                    continue
                yield skill_id, label, float(prob)

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
        min_conf = self.config.get("conf_high", 0.7)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
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
            An IntentHandlerMatch if a medium -confidence match is found, None otherwise.
        """
        min_conf = self.config.get("conf_medium", 0.5)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
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
        min_conf = self.config.get("conf_low", 0.15)
        LOG.debug(f"Matching intents via Model2Vec (min_conf: {min_conf}) - {utterances[0]}")
        for skill_id, label, prob in self._match(utterances[0]):
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
