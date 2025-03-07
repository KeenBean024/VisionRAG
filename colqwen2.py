from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from typing import Any, List, cast

class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 model implementation that can be used both for retrieval and generation.
    Allows switching between retrieval and generation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_retrieval_enabled = True

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
        or ColQwen2.forward for retrieval based on the current mode.
        """
        if self.is_retrieval_enabled:
            return ColQwen2.forward(self, *args, **kwargs)
        else:
            return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate text using Qwen2VLForConditionalGeneration.generate.
        """
        if not self.is_generation_enabled:
            raise ValueError(
                "Set the model to generation mode by calling `enable_generation()` before calling `generate()`."
            )
        return super().generate(*args, **kwargs)

    @property
    def is_retrieval_enabled(self) -> bool:
        return self._is_retrieval_enabled

    @property
    def is_generation_enabled(self) -> bool:
        return not self.is_retrieval_enabled

    def enable_retrieval(self) -> None:
        """
        Switch to retrieval mode.
        """
        self.enable_adapters()
        self._is_retrieval_enabled = True

    def enable_generation(self) -> None:
        """
        Switch to generation mode.
        """
        self.disable_adapters()
        self._is_retrieval_enabled = False


