"""
ACP Backend Integration for QuantaAlpha LLM Client.

This module provides the integration layer to replace OpenAI API calls
with ACP (Agent Client Protocol) communication to OpenCode.

Usage:
    1. Set environment variable USE_ACP_BACKEND=true
    2. Configure OPENCODE_PATH or ensure opencode is in PATH
    3. Configure external embedding API:
       - EXTERNAL_EMBEDDING_API=https://api.siliconflow.cn/v1/embeddings
       - EXTERNAL_EMBEDDING_API_KEY=your_key
       - EXTERNAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any

# Import settings
from quantaalpha.llm.config import LLM_SETTINGS

# Import the ACP client
try:
    from quantaalpha.llm.acp_client import ACPBackend, ACPClient
except ImportError:
    ACPBackend = None
    ACPClient = None

# Singleton instance
_acp_backend: ACPBackend | None = None


def get_acp_backend() -> ACPBackend:
    """Get the singleton ACP backend instance."""
    global _acp_backend
    if _acp_backend is None and ACPBackend is not None:
        _acp_backend = ACPBackend()
    return _acp_backend


def is_acp_enabled() -> bool:
    """Check if ACP backend is enabled via environment variable."""
    return os.environ.get("USE_ACP_BACKEND", "").lower() in ("true", "1", "yes", "on")


def is_external_embedding_enabled() -> bool:
    """Check if external embedding API is configured."""
    return bool(os.environ.get("EXTERNAL_EMBEDDING_API") and os.environ.get("EXTERNAL_EMBEDDING_API_KEY"))


class ACPChatCompletionMixin:
    """
    Mixin class to add ACP chat completion capability to APIBackend.

    This mixin modifies the chat completion behavior when ACP is enabled.
    """

    def _acp_create_chat_completion_inner_function(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, Any]:
        """
        Create chat completion via ACP backend.

        Returns:
            Tuple of (response_text, finish_reason)
        """
        backend = get_acp_backend()
        if backend is None:
            raise RuntimeError("ACP backend not available")

        if temperature is None:
            temperature = LLM_SETTINGS.chat_temperature
        if max_tokens is None:
            max_tokens = LLM_SETTINGS.chat_max_tokens

        # Get caller info for model selection
        import inspect
        caller_locals = inspect.stack()[2].frame.f_locals
        if "self" in caller_locals:
            tag = caller_locals["self"].__class__.__name__
        else:
            tag = "default"

        # Select model based on reasoning flag and caller
        reasoning_flag = kwargs.get("reasoning_flag", True)
        if reasoning_flag and LLM_SETTINGS.reasoning_model:
            model = LLM_SETTINGS.reasoning_model
        else:
            # Import here to avoid circular dependency
            from quantaalpha.llm.client import APIBackend
            if isinstance(self, APIBackend):
                chat_model_map = json.loads(self.chat_model_map)
                model = chat_model_map.get(tag, self.chat_model)
            else:
                model = LLM_SETTINGS.chat_model

        # Call ACP backend
        response_text = backend.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=kwargs.get("json_mode", False),
        )

        # Handle JSON mode if needed
        if kwargs.get("json_mode"):
            import json
            import re

            # Extract JSON part
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]

            # Try to fix common JSON issues
            try:
                json.loads(response_text)
            except json.JSONDecodeError:
                # Fix LaTeX escapes
                latex_commands = ['text', 'frac', 'left', 'right', 'times', 'cdot', 'sqrt']
                for cmd in latex_commands:
                    response_text = re.sub(r'(?<!\\)\\(' + cmd + r')', r'\\\\\1', response_text)
                response_text = re.sub(r'(?<!\\)\\([_\{\}\[\]])', r'\\\\\1', response_text)

        return response_text, None

    def _acp_create_embedding_inner_function(
        self,
        input_content_list: list[str],
        **kwargs: Any,
    ) -> list[Any]:
        """Create embeddings via ACP backend or external API."""
        backend = get_acp_backend()
        if backend is None:
            raise RuntimeError("ACP backend not available")
        return backend.create_embedding(input_content_list, **kwargs)


def patch_apibackend():
    """
    Patch the APIBackend class to use ACP when enabled.

    This function modifies the APIBackend class at runtime to intercept
    chat completion and embedding calls when USE_ACP_BACKEND is set.
    """
    if not is_acp_enabled():
        return

    # Lazy import to avoid circular dependency
    from quantaalpha.llm.client import APIBackend

    # Store original methods
    original_chat_method = APIBackend._create_chat_completion_inner_function
    original_embedding_method = APIBackend._create_embedding_inner_function

    # Create patched methods
    def patched_chat_method(self, *args, **kwargs):
        if is_acp_enabled():
            try:
                return ACPChatCompletionMixin()._acp_create_chat_completion_inner_function(
                    self, *args, **kwargs
                )
            except Exception as e:
                from quantaalpha.log import logger
                logger.warning(f"ACP chat completion failed, falling back: {e}")
                return original_chat_method(self, *args, **kwargs)
        return original_chat_method(self, *args, **kwargs)

    def patched_embedding_method(self, *args, **kwargs):
        if is_external_embedding_enabled():
            try:
                return ACPChatCompletionMixin()._acp_create_embedding_inner_function(
                    self, *args, **kwargs
                )
            except Exception as e:
                from quantaalpha.log import logger
                logger.warning(f"External embedding failed, falling back: {e}")
                return original_embedding_method(self, *args, **kwargs)
        return original_embedding_method(self, *args, **kwargs)

    # Apply patches
    APIBackend._create_chat_completion_inner_function = patched_chat_method
    APIBackend._create_embedding_inner_function = patched_embedding_method

    # Add cleanup hook
    import atexit

    def cleanup_acp():
        global _acp_backend
        if _acp_backend is not None:
            _acp_backend.shutdown()
            _acp_backend = None

    atexit.register(cleanup_acp)


# Auto-patch on import if enabled
if is_acp_enabled():
    patch_apibackend()
