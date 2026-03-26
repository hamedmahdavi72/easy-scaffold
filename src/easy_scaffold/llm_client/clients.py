from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

from anthropic import APIStatusError as AnthropicAPIStatusError
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from .providers.errors import (
    ProviderAPIError,
    ProviderContentPolicyError,
    ProviderRateLimitError,
)
from .providers.openai_compat import ImageGenerationResult
from .providers.router import route_chat_completion

# HuggingFace tokenizer for chat templates
from transformers import AutoTokenizer

TRANSFORMERS_AVAILABLE = True

from ..common.custom_exceptions import (
    APIServerException,
    ContentBlockedException,
    EmptyResponseException,
    RateLimitException,
    WorkflowException,
)
from ..configs.pydantic_models import LLMConfig, ModelProfile, StageConfig
from .rate_limiter import RateLimiterRegistry
from ..tools.manager import ToolExecutor, get_registry

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """
    LLM client: provider router (Gemini/OpenAI/DeepSeek/Anthropic), OpenAI SDK for vLLM and completion mode.
    """
    
    # Class-level cache for tokenizers to avoid reloading
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(self, llm_config: LLMConfig):
        self._config = llm_config

    def _extract_hf_model_name(self, vllm_model: str) -> str:
        # Remove "vllm/" prefix if present
        if vllm_model.startswith("vllm/"):
            return vllm_model[5:]  # Remove "vllm/" (5 characters)
        return vllm_model

    def _get_tokenizer(self, hf_model_name: str):
        """
        Get or load HuggingFace tokenizer for the given model.
        Uses caching to avoid reloading the same tokenizer.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers library is required for completion_mode. "
                "Install it with: pip install transformers"
            )
        
        if hf_model_name not in self._tokenizer_cache:
            logger.debug(f"Loading tokenizer for HuggingFace model: {hf_model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                
                # Fallback: If tokenizer has no chat_template, assign default ChatML
                # This is common for some distilled models or base models
                if not tokenizer.chat_template:
                    logger.warning(f"Tokenizer for '{hf_model_name}' has no chat_template. Assigning default ChatML template.")
                    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                
                self._tokenizer_cache[hf_model_name] = tokenizer
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer for model '{hf_model_name}': {str(e)}"
                ) from e
        
        return self._tokenizer_cache[hf_model_name]

    def count_tokens(self, text: str, model_profile_name: str) -> int:
        """
        Count the number of tokens in a text string using the model's tokenizer.
        """
        profile = self._config.resolve_profile(model_profile_name)
        
        # Only support HuggingFace tokenizers for now (vLLM/local models)
        if profile.provider == "vllm" or TRANSFORMERS_AVAILABLE:
            try:
                hf_model_name = self._extract_hf_model_name(profile.model)
                tokenizer = self._get_tokenizer(hf_model_name)
                return len(tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Failed to count tokens with tokenizer for {model_profile_name}: {e}")
        
        # Fallback to character-based heuristic for API models or errors
        # Approx 4 characters per token is standard for English, but math is denser (2.5-3)
        # Using 3.0 as a safer heuristic for math content
        return int(len(text) / 3.0)

    def _compute_logit_bias_for_token(
        self,
        tokenizer: Any,
        token_text: str,
        bias_value: float = -100.0,
    ) -> Dict[int, float]:
        """
        Tokenize token_text and return logit_bias dict mapping token IDs to bias values.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            token_text: Text token to suppress (e.g., "`</think>`")
            bias_value: Logit bias value to apply (negative suppresses, default -100.0)
        
        Returns:
            Dictionary mapping token IDs to bias values for OpenAI API logit_bias parameter
        """
        try:
            # Tokenize the token text
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            
            if not token_ids:
                logger.warning(f"Token '{token_text}' did not produce any token IDs")
                return {}
            
            # Create logit_bias dict: {token_id: bias_value}
            logit_bias = {token_id: bias_value for token_id in token_ids}
            
            logger.debug(
                f"Computed logit_bias for token '{token_text}': "
                f"{len(logit_bias)} token IDs with bias {bias_value}"
            )
            
            return logit_bias
        except Exception as e:
            logger.warning(f"Failed to compute logit_bias for token '{token_text}': {e}")
            return {}

    def _extract_thinking_format_from_output(
        self, tokenizer: Any
    ) -> Optional[tuple[str, str]]:
        """
        Extract thinking format by testing what the tokenizer actually produces.
        This is the most reliable method - we see the exact format used.
        """
        # Create a test message with thinking content
        # Use unique markers so we can identify them in output
        test_reasoning = "TEST_REASONING_MARKER_XYZ"
        test_answer = "TEST_ANSWER_MARKER_XYZ"
        
        test_content = f"`<think>`\n\n{test_reasoning}\n\n`</think>`\n\n{test_answer}"
        
        test_messages = [
            {"role": "user", "content": "test question"},
            {"role": "assistant", "content": test_content}
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Find where our test content appears in the formatted output
            # Extract the thinking block format around it
            import re
            
            # Look for the pattern: opening tag, our marker, closing tag
            # This will show us the exact format the tokenizer uses
            pattern = r'(`*<[^>]*(?:think|reasoning)[^>]*>`*).*?TEST_REASONING_MARKER_XYZ.*?(`*</[^>]*(?:think|reasoning)[^>]*>`*)'
            match = re.search(pattern, formatted, re.IGNORECASE | re.DOTALL)
            
            if match:
                open_token = match.group(1)
                close_token = match.group(2)
                return (open_token, close_token)
            
            # Alternative: if test markers aren't found, look for any thinking block pattern
            # Find opening and closing tags
            thinking_open = re.search(
                r'`*<[^>]*(?:think|reasoning)[^>]*>`*', formatted, re.IGNORECASE
            )
            if thinking_open:
                open_token = thinking_open.group(0)
                # Derive close token from open token
                if open_token.startswith('`') and open_token.endswith('`'):
                    # Format: `` `<think>` ``
                    inner = open_token.strip('`')
                    close_token = inner.replace('<', '</')
                    return (open_token, f"`{close_token}`")
                else:
                    # Format: `<think>`
                    close_token = open_token.replace('<', '</')
                    if close_token in formatted:
                        return (open_token, close_token)
                        
        except Exception as e:
            logger.debug(f"Failed to extract thinking format from output: {e}")
        
        return None

    def _extract_thinking_format_from_template(
        self, tokenizer: Any
    ) -> Optional[tuple[str, str]]:
        """
        Extract thinking block format directly from tokenizer's chat template.
        Parses the Jinja template to find thinking-related patterns.
        """
        if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
            return None
        
        template_str = str(tokenizer.chat_template)
        
        # Look for thinking-related blocks in the template
        import re
        
        # Find all XML-like tags containing "think" or "reasoning" (case insensitive)
        # This matches: `<think>`, `` `<think>` ``, `<think>`, etc.
        thinking_tags = re.findall(
            r'`*<[^>]*(?:think|reasoning)[^>]*>`*',
            template_str,
            re.IGNORECASE
        )
        
        if thinking_tags:
            # Get the first thinking tag found (usually the opening one)
            open_token = thinking_tags[0]
            # Derive close token from open token
            # Handle both `<think>` and `` `<think>` `` formats
            if open_token.startswith('`') and open_token.endswith('`'):
                # Format: `` `<think>` ``
                inner = open_token.strip('`')
                close_token = inner.replace('<', '</')
                return (open_token, f"`{close_token}`")
            else:
                # Format: `<think>`
                close_token = open_token.replace('<', '</')
                return (open_token, close_token)
        
        return None

    def _detect_thinking_tokens(self, tokenizer: Any) -> Optional[tuple[str, str]]:
        """
        Detect thinking block format from tokenizer without hardcoding.
        Tries multiple methods to extract the exact format.
        """
        # Method 1: Extract from actual output (most reliable)
        result = self._extract_thinking_format_from_output(tokenizer)
        if result:
            return result
        
        # Method 2: Parse template string
        result = self._extract_thinking_format_from_template(tokenizer)
        if result:
            return result
        
        # Method 3: Check special tokens (if available)
        if hasattr(tokenizer, 'special_tokens_map'):
            special_tokens = tokenizer.special_tokens_map
            for key, value in special_tokens.items():
                if isinstance(value, str) and (
                    'think' in key.lower() or 'reasoning' in key.lower()
                ):
                    # Found a thinking-related special token
                    open_token = value
                    close_token = value.replace('<', '</')
                    return (open_token, close_token)
        
        return None

    def _has_incomplete_thinking(
        self, content: str, open_token: str, close_token: str
    ) -> bool:
        """
        Check if content has incomplete thinking (open tag without matching close).
        
        Args:
            content: The content to check
            open_token: Opening thinking token (e.g., "`<think>`")
            close_token: Closing thinking token (e.g., "`</think>`")
            
        Returns:
            True if thinking block is incomplete (open without close)
        """
        if not content or not open_token:
            return False
        
        has_open = open_token in content
        has_close = close_token in content
        
        # Check if open appears after the last close (incomplete)
        if has_open:
            open_pos = content.rfind(open_token)
            close_pos = content.rfind(close_token) if has_close else -1
            return open_pos > close_pos
        
        return False

    def _format_assistant_message_manually(
        self, assistant_content: str, tokenizer: Any
    ) -> str:
        """
        Format assistant message manually to avoid duplicate thinking blocks.
        Uses the same format the tokenizer would use for assistant role.
        """
        logger.info(
            f"_format_assistant_message_manually: input length={len(assistant_content)}, "
            f"first_200_chars={repr(assistant_content[:200]) if assistant_content else 'EMPTY'}"
        )
        
        # Extract the assistant role format from tokenizer's template
        # Most templates use: <|im_start|>assistant\n{content}
        # But we can test it to be sure
        test_messages = [{"role": "assistant", "content": "test"}]
        try:
            formatted = tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=False
            )
            logger.debug(f"Tokenizer test format: {repr(formatted)}")
            
            # Extract the pattern: everything before "test" is the format
            if "test" in formatted:
                format_prefix = formatted.split("test")[0]
                logger.debug(f"Extracted format_prefix: {repr(format_prefix)}")
                
                # Prevent duplicate thinking blocks if tokenizer adds them and content has them
                # Check for common thinking tokens (including the configured token)
                thinking_tokens = [
                    "<think>", 
                    "`<think>`", 
                    " reasoning\n"
                ]
                for token in thinking_tokens:
                    clean_token = token.strip()
                    if format_prefix.strip().endswith(clean_token) and assistant_content.strip().startswith(clean_token):
                        logger.debug(f"Found duplicate token '{clean_token}', removing from prefix")
                        # Remove the token from the end of the prefix to avoid duplication
                        # Find the last occurrence and slice
                        idx = format_prefix.rfind(clean_token)
                        if idx != -1:
                            format_prefix = format_prefix[:idx].rstrip()
                            if format_prefix.endswith("\n"): # Keep one newline if present
                                format_prefix = format_prefix
                            else:
                                format_prefix += "\n"
                        logger.debug(f"After deduplication, format_prefix: {repr(format_prefix)}")
                        break

                result = format_prefix + assistant_content
                logger.info(
                    f"_format_assistant_message_manually: result length={len(result)}, "
                    f"last_200_chars={repr(result[-200:]) if len(result) > 200 else repr(result)}"
                )
                return result
            else:
                logger.warning(f"'test' not found in tokenizer output: {repr(formatted)}")
        except Exception as e:
            logger.warning(f"Exception in _format_assistant_message_manually: {e}", exc_info=True)
        
        # Fallback to common format
        result = f"<|im_start|>assistant\n{assistant_content}"
        logger.info(
            f"_format_assistant_message_manually: using fallback, result length={len(result)}, "
            f"last_200_chars={repr(result[-200:]) if len(result) > 200 else repr(result)}"
        )
        return result

    def _messages_to_prompt(
        self,
        messages: List[Dict[str, Any]],
        vllm_model: str,
        thinking_block_token: Optional[str] = None,
    ) -> str:
        """
        Convert messages list to a single prompt string using the model's chat template.
        
        For completion mode, the prompt MUST end with an open assistant turn (<|im_start|>assistant)
        with no content after it. This matches the training format where the model generates
        thinking tokens and response naturally.
        
        CRITICAL: Never prefill assistant content unless explicitly continuing a partial generation.
        The model expects to start fresh generation from <|im_start|>assistant.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            vllm_model: vLLM model name (e.g., "vllm/Qwen/Qwen3-4B-Thinking-2507")
            thinking_block_token: Optional opening token for thinking block (unused, kept for compatibility)
            
        Returns:
            Single prompt string formatted according to the model's chat template.
            For completion mode, this will end with <|im_start|>assistant and nothing after it.
        """
        if not messages:
            raise ValueError("Cannot convert empty messages list to prompt")
        
        # Extract HuggingFace model name from vLLM model name
        hf_model_name = self._extract_hf_model_name(vllm_model)
        
        # Get tokenizer for this model
        tokenizer = self._get_tokenizer(hf_model_name)
        
        # CRITICAL FIX: For completion mode, if the last message is assistant with content,
        # we want to continue from that point (append it to the prompt for continuation)
        last_message = messages[-1]
        content = last_message.get("content", "")
        is_partial_assistant = (
            last_message.get("role") == "assistant"
            and content.strip() != ""  # Has content to continue from
        )
        
        if is_partial_assistant:
            # Continuation case: format only messages before assistant, then add open assistant turn
            # This is for resuming generation from a partial response
            try:
                base_prompt = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply chat template for model '{hf_model_name}': {str(e)}"
                ) from e
        
            # Get the assistant content (partial, for continuation)
            assistant_content = messages[-1].get("content", "")
            
            logger.info(
                f"Continuation mode: assistant_content length={len(assistant_content)}, "
                f"first_200_chars={repr(assistant_content[:200]) if assistant_content else 'EMPTY'}, "
                f"last_100_chars={repr(assistant_content[-100:]) if len(assistant_content) > 100 else repr(assistant_content)}"
            )
            
            # Format with open assistant turn + partial content
            # The model will continue from this point
            assistant_format = self._format_assistant_message_manually(
                assistant_content, tokenizer
            )
            
            logger.info(
                f"Formatted assistant message: length={len(assistant_format)}, "
                f"last_200_chars={repr(assistant_format[-200:]) if len(assistant_format) > 200 else repr(assistant_format)}"
            )
            
            prompt = base_prompt + assistant_format
        else:
            # Normal case: User/System was last, or assistant message is empty/incomplete.
            # Remove any assistant messages from the end to ensure clean generation
            clean_messages = messages.copy()
            while clean_messages and clean_messages[-1].get("role") == "assistant":
                # Remove empty or complete assistant messages
                if not clean_messages[-1].get("content", "").strip():
                    clean_messages.pop()
                else:
                    # If there's content but it's not a continuation, remove it
                    # We want fresh generation, not continuation
                    clean_messages.pop()
            
            # Ensure we have at least one message
            if not clean_messages:
                raise ValueError("Cannot create prompt: all messages were removed")
            
            # Generate prompt with open assistant turn
            try:
                prompt = tokenizer.apply_chat_template(
                    clean_messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Add assistant stub (e.g. <|im_start|>assistant\n)
                )
                
                # ENFORCEMENT: If this model is a thinking model (has <think> token config),
                # append <think>\n to force it into thinking mode.
                # Only do this if we are starting fresh (not continuing partial).
                # CRITICAL: Some tokenizers (like Nomos-1/Qwen3) automatically inject <think>
                # when add_generation_prompt=True. We must check for this to avoid double injection.
                if thinking_block_token: # configured in profile
                    # Check if prompt already ends with the thinking token
                    # The tokenizer might add: <think>\n or just <think>
                    token_clean = thinking_block_token.strip()
                    prompt_stripped = prompt.rstrip()
                    
                    # Simple check: Does the prompt (after stripping trailing whitespace) end with the token?
                    # This handles both "<think>" and "<think>\n" cases
                    has_token_at_end = prompt_stripped.endswith(token_clean)
                    
                    if not has_token_at_end:
                        # Tokenizer didn't add it, so we add it
                        if not prompt.endswith("\n"):
                            prompt += "\n"
                        prompt += f"{thinking_block_token}\n"
                    else:
                        # Tokenizer already added it - log for debugging
                        logger.debug(
                            f"Tokenizer already injected thinking token '{thinking_block_token}'. "
                            f"Skipping manual injection. Prompt ends with: {repr(prompt[-50:])}"
                        )
                        
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply chat template for model '{hf_model_name}': {str(e)}"
                ) from e
        
        # CRITICAL VALIDATION: For completion mode, prompt must end with open assistant turn OR thinking token
        # This ensures the model knows it should generate, not that generation is complete
        prompt_rstrip = prompt.rstrip()
        # Acceptable endings: <|im_start|>assistant, <|im_start|>assistant\n, or <think>\n
        valid_endings = [
            "<|im_start|>assistant", 
            "<|im_start|>assistant\n",
        ]
        if thinking_block_token:
             valid_endings.append(f"{thinking_block_token}")
             valid_endings.append(f"{thinking_block_token}\n")
             
        if not any(prompt_rstrip.endswith(end.rstrip()) for end in valid_endings):
            logger.warning(
                f"Completion prompt does not end with open assistant turn. "
                f"This may cause early EOS or short responses. "
                f"Last 100 chars: {repr(prompt[-100:])}"
            )
        
        # Log final prompt ending for debugging
        logger.debug(
            f"_messages_to_prompt final result: "
            f"length={len(prompt)} chars, "
            f"last_100_chars={repr(prompt[-100:])}, "
            f"thinking_token_configured={thinking_block_token is not None}"
        )
        
        return prompt

    def _is_transient_error(self, err: Exception) -> bool:
        """True for 5xx and similar transient provider errors."""
        if isinstance(err, ProviderAPIError):
            code = getattr(err, "status_code", None)
            if code is not None and 500 <= int(code) < 600:
                return True
            error_str = str(err).lower()
            if any(
                x in error_str
                for x in ["500", "502", "503", "504", "internal server error"]
            ):
                return True
        if isinstance(err, OpenAIError):
            status_code = getattr(err, "status_code", None)
            if status_code and 500 <= status_code < 600:
                return True
        if isinstance(err, AnthropicAPIStatusError):
            status_code = getattr(err, "status_code", None)
            if status_code and 500 <= status_code < 600:
                return True
        return False

    def _merge_completion_responses(self, r1: Any, r2: Any) -> Any:
        """
        Merges two completion responses (sequential generation).
        Used for two-step generation (e.g. suppression).
        """
        # Start with r2 as base (captures finish reason of final step)
        merged = r2
        
        # Prepend text from r1
        if hasattr(r1, "choices") and len(r1.choices) > 0:
             r1_text = r1.choices[0].text
             merged.choices[0].text = r1_text + merged.choices[0].text
             
        # Sum usage logic to reflect total cost
        if hasattr(r1, "usage") and hasattr(r2, "usage"):
             merged.usage.prompt_tokens = r1.usage.prompt_tokens + r2.usage.prompt_tokens
             merged.usage.completion_tokens = r1.usage.completion_tokens + r2.usage.completion_tokens
             merged.usage.total_tokens = merged.usage.prompt_tokens + merged.usage.completion_tokens
             
        return merged

    async def create(
        self,
        stage_config: StageConfig,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[BaseModel]] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        profile = self._config.resolve_profile(stage_config.model_profile)
        params = self._build_params(profile, response_model)

        # Stage-level overrides take precedence
        if stage_config.generation_config:
            params.update(stage_config.generation_config)

        # Runtime overrides take highest precedence
        if generation_overrides:
            params.update(generation_overrides)

        # Handle provider-specific structured output configuration
        # If response_model is present, automatically configure structured output based on provider
        if response_model is not None:
            if profile.provider == "gemini":
                # Gemini requires response_mime_type for structured output
                params["response_mime_type"] = "application/json"
            # For OpenAI and other providers, response_format is already set in _build_params
            # Remove response_mime_type if it was manually specified for non-Gemini providers
            elif profile.provider != "gemini" and "response_mime_type" in params:
                logger.debug(
                    f"Removing response_mime_type for {profile.provider} provider "
                    f"(not supported, using response_format instead)"
                )
                params.pop("response_mime_type")

        # Handle tools if specified in stage config
        tools = None
        tool_executor = None
        if stage_config.tools:
            registry = get_registry()
            tool_definitions = registry.get_tools(stage_config.tools)
            if tool_definitions:
                tools = registry.to_openai_format(tool_definitions)
                tool_executor = ToolExecutor(tool_definitions)
                logger.debug(f"Loaded {len(tool_definitions)} tools: {[t.name for t in tool_definitions]}")
            else:
                logger.warning(f"No tools found for references: {stage_config.tools}")

        final_params = {k: v for k, v in params.items() if v is not None}
        
        # Extract semantic overrides for token suppression
        # These are domain-specific flags that trigger two-step generation in completion_mode
        suppress_token = final_params.pop("_suppress_token", None)
        suppress_for_tokens = final_params.pop("_suppress_for_tokens", 0)

        # Get rate limiter for this model (initialization happens here, which may help with first-call issues)
        rate_limiter = None
        if self._config.rate_limits:
            rate_limit_config = self._config.rate_limits.get(profile.model)
            if rate_limit_config:
                rate_limiter = await RateLimiterRegistry.get(profile.model, rate_limit_config)

        # Retry loop with rate limit handling
        max_retries = self._config.retry_config.num_retries
        cooldown = self._config.retry_config.cooldown_seconds

        for attempt in range(max_retries + 1):
            try:
                # Acquire rate limit permit (waits if needed for request slot)
                if rate_limiter:
                    await rate_limiter.acquire()

                # Handle completion mode vs chat completion mode
                if profile.completion_mode:
                    # Convert messages to single prompt string using chat template
                    prompt = self._messages_to_prompt(
                        messages,
                        vllm_model=profile.model,
                        thinking_block_token=profile.thinking_block_token
                    )
                    
                    # Dynamic max_tokens adjustment to fit context window
                    try:
                        hf_model_name = self._extract_hf_model_name(profile.model)
                        tokenizer = self._get_tokenizer(hf_model_name)
                        input_tokens = len(tokenizer.encode(prompt))
                        
                        # Default to 32k for Qwen/vLLM if not specified
                        # Ideally this should be in config, but hardcoding for stability for now
                        max_context_window = getattr(profile, 'context_window', 32768)
                        safety_buffer = 200
                        
                        requested_max = final_params.get("max_tokens", 2048)
                        available_space = max_context_window - input_tokens - safety_buffer
                        
                        if available_space < 100:
                            logger.warning(f"Context nearly full! Input: {input_tokens}, Available: {available_space}")
                            adjusted_max = available_space if available_space > 0 else 10
                        else:
                            adjusted_max = min(requested_max, available_space)
                            
                        if adjusted_max != requested_max:
                            logger.info(
                                f"Adjusting max_tokens to fit context: "
                                f"Input={input_tokens}, Requested={requested_max}, "
                                f"Limit={max_context_window}, New max_tokens={adjusted_max}"
                            )
                            final_params["max_tokens"] = adjusted_max
                        
                        # CRITICAL: Enforce minimum generation floor to prevent short responses
                        # If context is too full, fail fast instead of generating 30-50 tokens
                        MIN_GENERATION_TOKENS = 256
                        if final_params["max_tokens"] < MIN_GENERATION_TOKENS:
                            logger.warning(
                                f"max_tokens ({final_params['max_tokens']}) is too small for reliable generation. "
                                f"Context may be too full. Consider reducing input size."
                            )
                            # Still allow it, but log warning
                            
                    except Exception as e:
                        logger.warning(f"Failed to calculate dynamic max_tokens: {e}")

       
                    
                    # Use OpenAI client for completion endpoint
                    timeout = profile.get_timeout()
                    openai_client = AsyncOpenAI(
                        api_key=profile.api_key,
                        base_url=profile.api_base,
                        timeout=timeout,
                    )
                    
                    # Make API call using completion endpoint
                    # Log prompt for debugging (show last 500 chars to see ending)
                    logger.debug(
                        f"Completion prompt (last 500 chars): {repr(prompt[-500:]) if len(prompt) > 500 else repr(prompt)}"
                    )
                    logger.info(
                        f"Completion prompt length: {len(prompt)} chars, "
                        f"input_tokens: {input_tokens}, "
                        f"max_tokens: {final_params.get('max_tokens', 'N/A')}, "
                        f"context_window: {max_context_window}"
                    )
                    
                    _pv = 800
                    if len(prompt) > _pv * 2:
                        logger.debug(
                            "Completion prompt preview: head=%s ... tail=%s",
                            repr(prompt[:_pv]),
                            repr(prompt[-_pv:]),
                        )
                    else:
                        logger.debug("Completion prompt preview: %s", repr(prompt))

                    # Handle vLLM-specific parameters (like top_k) via extra_body
                    # OpenAI SDK doesn't support top_k as a direct parameter, but vLLM accepts it via extra_body
                    extra_body_params = {}
                    params_for_api = dict(final_params)
                    
                    if profile.provider == "vllm":
                        # Extract vLLM-specific parameters that need to go in extra_body
                        vllm_extra_params = ["top_k"]
                        for param in vllm_extra_params:
                            if param in params_for_api:
                                extra_body_params[param] = params_for_api.pop(param)
                    else:
                        # Filter out unsupported parameters for standard OpenAI API
                        unsupported_params = ["top_k"]
                        for param in unsupported_params:
                            params_for_api.pop(param, None)
                    
                    # Check if two-step generation (suppression) is needed
                    if suppress_token and suppress_for_tokens > 0:
                        logger.info(f"Executing two-step generation to suppress '{suppress_token}' for {suppress_for_tokens} tokens")
                        
                        # === Step 1: Suppressed Burst ===
                        step1_params = params_for_api.copy()
                        step1_params["max_tokens"] = suppress_for_tokens
                        step1_params["stop"] = None  # Ignore stop tokens to get full burst
                        
                        # Compute & Apply Logit Bias
                        try:
                            hf_model_name = self._extract_hf_model_name(profile.model)
                            tokenizer = self._get_tokenizer(hf_model_name)
                            
                            # Handle single token or list of tokens
                            tokens_to_suppress = suppress_token if isinstance(suppress_token, list) else [suppress_token]
                            
                            # Aggregate logit bias for all tokens
                            aggregated_bias = {}
                            for token in tokens_to_suppress:
                                bias = self._compute_logit_bias_for_token(tokenizer, token)
                                aggregated_bias.update(bias)
                            
                            if aggregated_bias:
                                # Merge with existing bias
                                existing = step1_params.get("logit_bias", {})
                                if existing:
                                    existing = existing.copy()
                                    existing.update(aggregated_bias)
                                    step1_params["logit_bias"] = existing
                                else:
                                    step1_params["logit_bias"] = aggregated_bias
                        except Exception as e:
                            logger.warning(f"Failed to compute logit bias for suppression: {e}")
                        
                        # Execute Step 1
                        response_step1 = await openai_client.completions.create(
                            model=profile.model,
                            prompt=prompt,
                            **step1_params,
                            extra_body=extra_body_params if extra_body_params else None,
                        )
                        
                        step1_text = response_step1.choices[0].text
                        logger.debug(f"Step 1 output ({len(step1_text)} chars): {repr(step1_text)}")

                        # === Step 2: Continuation ===
                        step2_params = params_for_api.copy()
                        # Adjust max_tokens
                        original_max = final_params.get("max_tokens", 2048)
                        # Estimate tokens consumed in step 1 (using tokenizer or heuristic)
                        try:
                            consumed = len(tokenizer.encode(step1_text))
                        except:
                            consumed = int(len(step1_text) / 3.0)
                        
                        remaining = max(1, original_max - consumed)
                        step2_params["max_tokens"] = remaining
                        
                        # Execute Step 2 (without suppression bias, essentially standard params)
                        response_step2 = await openai_client.completions.create(
                            model=profile.model,
                            prompt=prompt + step1_text,
                            **step2_params,
                            extra_body=extra_body_params if extra_body_params else None,
                        )
                        
                        # Merge Results
                        response = self._merge_completion_responses(response_step1, response_step2)
                        
                    else:
                        # === Standard Generation ===
                        response = await openai_client.completions.create(
                            model=profile.model,
                            prompt=prompt,
                            **params_for_api,
                            extra_body=extra_body_params if extra_body_params else None,
                        )
                else:
                    # Use chat completion
                    # For vLLM provider, use OpenAI SDK directly to preserve thinking tokens
                    # Direct OpenAI SDK preserves thinking tokens for vLLM
                    # Log messages for debugging
                    logger.debug(
                        f"Chat completion messages: {len(messages)} messages, "
                        f"last_message_role={messages[-1].get('role') if messages else 'N/A'}, "
                        f"last_message_preview={repr(messages[-1].get('content', '')[:200]) if messages else 'N/A'}"
                    )
                    logger.info(
                        f"Chat completion: model={profile.model}, "
                        f"max_tokens={final_params.get('max_tokens', 'N/A')}"
                    )
                    
                    if profile.provider == "vllm":
                        # Use OpenAI SDK directly for vLLM to preserve thinking tokens
                        timeout = profile.get_timeout()
                        openai_client = AsyncOpenAI(
                            api_key=profile.api_key,
                            base_url=profile.api_base,
                            timeout=timeout,
                        )
                        
                        # Handle vLLM-specific parameters (like top_k) via extra_body
                        # OpenAI SDK doesn't support top_k as a direct parameter, but vLLM accepts it via extra_body
                        extra_body_params = {}
                        params_for_api = dict(final_params)
                        
                        # Extract vLLM-specific parameters that need to go in extra_body
                        vllm_extra_params = ["top_k"]
                        for param in vllm_extra_params:
                            if param in params_for_api:
                                extra_body_params[param] = params_for_api.pop(param)
                        
                        # Handle tool calling loop for vLLM (OpenAI SDK)
                        response = await self._handle_tool_calling_loop(
                            profile=profile,
                            messages=messages,
                            final_params=params_for_api,
                            tools=tools,
                            tool_executor=tool_executor,
                            rate_limiter=rate_limiter,
                            openai_client=openai_client,
                        )
                    else:
                        # Provider router for Gemini, OpenAI, DeepSeek
                        # Handle tool calling loop automatically
                        response = await self._handle_tool_calling_loop(
                            profile=profile,
                            messages=messages,
                            final_params=final_params,
                            tools=tools,
                            tool_executor=tool_executor,
                            rate_limiter=rate_limiter,
                            openai_client=None,  # use provider router
                        )
                    
                # Extract token stats and record input tokens
                if rate_limiter:
                    token_stats = self.extract_token_stats(response)
                    usage = token_stats.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    if prompt_tokens:
                        await rate_limiter.record_input_tokens(prompt_tokens)

                # Success - process response and check for empty content
                if not hasattr(response, 'choices') or len(response.choices) == 0:
                    raise EmptyResponseException("Response has no choices")
                
                # Extract content based on response type
                content = None
                if profile.completion_mode:
                    # Text completion mode
                    choice = response.choices[0]
                    content = getattr(choice, 'text', None)
                else:
                    # Chat completion mode
                    choice = response.choices[0]
                    message = getattr(choice, 'message', None)
                    if message:
                        content = getattr(message, 'content', None)
                        
                # Check for empty content
                if content is None or (isinstance(content, str) and not content.strip()):
                    # Check if this is a reasoning-only response (Gemini-specific)
                    usage = getattr(response, 'usage', None)
                    if usage:
                        completion_details = getattr(usage, 'completion_tokens_details', None)
                        if completion_details:
                            reasoning_tokens = getattr(completion_details, 'reasoning_tokens', 0)
                            text_tokens = getattr(completion_details, 'text_tokens', 0)
                            if reasoning_tokens > 0 and text_tokens == 0:
                                # Model did reasoning but produced no output - raise exception to trigger retry
                                raise EmptyResponseException(
                                    f"Model produced {reasoning_tokens} reasoning tokens but no text output"
                                )
                    
                    # Generic empty response - raise exception to trigger retry
                    raise EmptyResponseException(
                        f"Model returned empty content (completion_mode={profile.completion_mode})"
                    )
                
                # Process structured responses if applicable
                # Note: completion_mode doesn't support structured responses (text completions only)
                if response_model and not profile.completion_mode:
                    parsed = self._extract_structured_response(response)
                    if parsed is not None:
                        setattr(response, "parsed_output", parsed)
                
                return response

            except ProviderRateLimitError as err:
                # Rate limit hit - wait and retry
                if attempt < max_retries:
                    # Calculate wait time
                    wait_time = cooldown if cooldown > 0 else 60.0  # Default 60s if not configured

                    logger.warning(
                        f"Rate limit hit for {profile.model} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Max retries reached - raise exception
                    logger.error(
                        f"Rate limit exceeded for {profile.model} after {max_retries + 1} attempts. "
                        "Stopping execution."
                    )
                    raise RateLimitException(
                        f"Rate limit exceeded after {max_retries + 1} attempts: {str(err)}"
                    ) from err

            except (ProviderAPIError, OpenAIError) as err:
                # Check if this is a rate limit error (429 status code)
                status_code = getattr(err, "status_code", None)
                if status_code == 429:
                    # Treat 429 as rate limit error - retry with cooldown
                    if attempt < max_retries:
                        wait_time = cooldown if cooldown > 0 else 60.0  # Default 60s if not configured
                        
                        logger.warning(
                            f"Rate limit hit (429) for {profile.model} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Max retries reached - raise exception
                        logger.error(
                            f"Rate limit exceeded for {profile.model} after {max_retries + 1} attempts. "
                            "Stopping execution."
                        )
                        raise RateLimitException(
                            f"Rate limit exceeded after {max_retries + 1} attempts: {str(err)}"
                        ) from err
                
                # Check if this is a transient server error (5xx) that should be retried
                if self._is_transient_error(err):
                    if attempt < max_retries:
                        # Calculate wait time (exponential backoff for server errors)
                        wait_time = cooldown * (2 ** attempt) if cooldown > 0 else min(60.0 * (2 ** attempt), 300.0)
                        # Cap backoff at max_backoff_seconds if configured
                        max_backoff = getattr(self._config.retry_config, 'max_backoff_seconds', None)
                        if max_backoff is not None:
                            wait_time = min(wait_time, max_backoff)
                        
                        logger.warning(
                            f"Transient server error for {profile.model} (attempt {attempt + 1}/{max_retries + 1}): {str(err)}. "
                            f"Waiting {wait_time:.1f}s before retry."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Max retries reached - raise exception
                        logger.error(
                            f"Server error persisted for {profile.model} after {max_retries + 1} attempts. "
                            "Stopping execution."
                        )
                        raise APIServerException(
                            f"Server error persisted after {max_retries + 1} attempts: {str(err)}"
                        ) from err
                else:
                    # Non-transient API error - don't retry
                    if isinstance(err, ProviderAPIError):
                        raise APIServerException(str(err)) from err
                    else:
                        raise WorkflowException(str(err)) from err

            except ProviderContentPolicyError as err:
                raise ContentBlockedException(str(err)) from err

            except EmptyResponseException as err:
                # Empty response (including reasoning-only responses) - retry
                if attempt < max_retries:
                    wait_time = cooldown if cooldown > 0 else 5.0  # Shorter wait for empty responses
                    
                    logger.warning(
                        f"Empty response for {profile.model} (attempt {attempt + 1}/{max_retries + 1}): {str(err)}. "
                        f"Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Max retries reached - raise exception
                    logger.error(
                        f"Empty response persisted for {profile.model} after {max_retries + 1} attempts. "
                        "Stopping execution."
                    )
                    raise EmptyResponseException(
                        f"Empty response persisted after {max_retries + 1} attempts: {str(err)}"
                    ) from err

        # Should never reach here, but just in case
        raise RateLimitException(f"Failed after {max_retries + 1} attempts")

    def extract_token_stats(self, response: Any) -> Dict[str, Any]:
        usage = self._to_serializable(getattr(response, "usage", None))
        return {
            "usage": usage,
        }

    def _extract_structured_response(self, response: Any) -> Optional[Any]:
        try:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                return None
            if isinstance(message, dict):
                return message.get("parsed")
            return getattr(message, "parsed", None)
        except (IndexError, AttributeError):
            logger.debug("Structured response not found in provider payload.")
        return None

    def _extract_tool_calls(self, response: Any, is_completion_mode: bool) -> Optional[List[Any]]:
        """Extract tool calls from LLM response."""
        if is_completion_mode:
            # Completion mode doesn't support tool calling
            return None
        
        try:
            choice = response.choices[0]
            message = getattr(choice, 'message', None)
            if message:
                tool_calls = getattr(message, 'tool_calls', None)
                if tool_calls and len(tool_calls) > 0:
                    return tool_calls
        except (AttributeError, IndexError):
            pass
        return None
    
    async def _handle_tool_calling_loop(
        self,
        profile: ModelProfile,
        messages: List[Dict[str, Any]],
        final_params: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]],
        tool_executor: Optional[ToolExecutor],
        rate_limiter: Optional[Any],
        openai_client: Optional[AsyncOpenAI] = None,
    ) -> Any:
        """
        Handle tool calling loop - automatically executes tools and continues conversation.
        
        Returns final response after all tool calls are resolved.
        """
        if not tools or not tool_executor:
            # No tools - make single API call
            return await self._make_api_call(
                profile=profile,
                messages=messages,
                final_params=final_params,
                tools=tools,
                rate_limiter=rate_limiter,
                openai_client=openai_client,
            )
        
        # Tool calling loop
        max_tool_iterations = 10  # Prevent infinite loops
        conversation_messages = messages.copy()
        
        for tool_iteration in range(max_tool_iterations):
            # Make API call
            response = await self._make_api_call(
                profile=profile,
                messages=conversation_messages,
                final_params=final_params,
                tools=tools,
                rate_limiter=rate_limiter,
                openai_client=openai_client,
            )
            
            # Extract tool calls
            tool_calls = self._extract_tool_calls(response, profile.completion_mode)
            
            if not tool_calls:
                # No tool calls - return final response
                logger.debug(f"Tool calling loop completed after {tool_iteration} iterations (final answer)")
                return response
            
            # Execute tools
            logger.info(f"Tool calling iteration {tool_iteration + 1}: executing {len(tool_calls)} tool calls")
            tool_results = await tool_executor.execute_all(tool_calls)
            
            # Add assistant's message (with tool calls) to conversation
            choice = response.choices[0]
            message = getattr(choice, 'message', None)
            
            # Convert message to dict format for conversation
            if hasattr(message, 'model_dump'):
                assistant_message = message.model_dump()
            elif hasattr(message, 'dict'):
                assistant_message = message.dict()
            elif isinstance(message, dict):
                assistant_message = message.copy()
            else:
                # Fallback: construct message dict manually
                assistant_message = {
                    "role": "assistant",
                    "content": getattr(message, 'content', None),
                }
                # Add tool_calls if present
                if tool_calls:
                    if hasattr(tool_calls[0], 'model_dump'):
                        assistant_message["tool_calls"] = [tc.model_dump() for tc in tool_calls]
                    elif hasattr(tool_calls[0], 'dict'):
                        assistant_message["tool_calls"] = [tc.dict() for tc in tool_calls]
                    else:
                        assistant_message["tool_calls"] = tool_calls
            
            conversation_messages.append(assistant_message)
            
            # Add tool results to conversation
            conversation_messages.extend(tool_results)
            
            # Continue loop to get final answer
        
        # Max iterations reached - return last response
        logger.warning(f"Tool calling loop reached max iterations ({max_tool_iterations})")
        return response
    
    async def _make_api_call(
        self,
        profile: ModelProfile,
        messages: List[Dict[str, Any]],
        final_params: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]],
        rate_limiter: Optional[Any],
        openai_client: Optional[AsyncOpenAI] = None,
    ) -> Any:
        """Make a single API call (used by tool calling loop)."""
        # Acquire rate limit permit if needed
        if rate_limiter:
            await rate_limiter.acquire()
        
        # Extract timeout and remove from params (to avoid double-passing)
        timeout = profile.get_timeout()
        api_params = final_params.copy()
        api_params.pop("timeout", None)  # Remove timeout from params if present
        
        if tools:
            api_params["tools"] = tools
        
        if profile.completion_mode:
            # Completion mode - convert messages to prompt
            prompt = self._messages_to_prompt(
                messages,
                vllm_model=profile.model,
                thinking_block_token=profile.thinking_block_token
            )
            
            if not openai_client:
                openai_client = AsyncOpenAI(
                    api_key=profile.api_key,
                    base_url=profile.api_base,
                    timeout=timeout,
                )
            
            # Note: Completion mode doesn't support tools
            if tools:
                logger.warning("Tools are not supported in completion_mode, ignoring")
            
            response = await openai_client.completions.create(
                model=profile.model,
                prompt=prompt,
                **{k: v for k, v in api_params.items() if k != "tools"},
            )
        elif profile.provider == "vllm" or openai_client:
            # vLLM via OpenAI SDK or explicit OpenAI client
            if not openai_client:
                openai_client = AsyncOpenAI(
                    api_key=profile.api_key,
                    base_url=profile.api_base,
                    timeout=timeout,
                )
            
            # Handle vLLM-specific parameters
            extra_body_params = {}
            params_for_api = api_params.copy()
            
            vllm_extra_params = ["top_k"]
            for param in vllm_extra_params:
                if param in params_for_api:
                    extra_body_params[param] = params_for_api.pop(param)
            
            response = await openai_client.chat.completions.create(
                model=profile.model,
                messages=messages,
                **params_for_api,
                extra_body=extra_body_params if extra_body_params else None,
            )
        else:
            response = await route_chat_completion(
                profile=profile,
                messages=messages,
                api_params=api_params,
                tools=tools,
                timeout=timeout,
            )
        
        # Record input tokens
        if rate_limiter:
            token_stats = self.extract_token_stats(response)
            usage = token_stats.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            if prompt_tokens:
                await rate_limiter.record_input_tokens(prompt_tokens)
        
        return response
    


    async def generate_image(
        self,
        stage_config: StageConfig,
        prompt: str,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> ImageGenerationResult:
        from .providers.router import route_image_generation

        profile = self._config.resolve_profile(stage_config.model_profile)
        if profile.task != "image_generation":
            raise ValueError(
                f"generate_image requires task=image_generation, got {profile.task!r}"
            )
        timeout = profile.get_timeout()
        params: Dict[str, Any] = {
            "temperature": profile.temperature,
            **profile.extra_params,
        }
        if generation_overrides:
            params.update(generation_overrides)
        return await route_image_generation(
            profile=profile,
            prompt=prompt,
            api_params=params,
            timeout=timeout,
        )

    def _build_params(
        self,
        profile: ModelProfile,
        response_model: Optional[Type[BaseModel]],
    ) -> Dict[str, Any]:
        # Start with temperature and extra_params, but exclude timeout (handled separately)
        extra_params_clean = {k: v for k, v in profile.extra_params.items() if k != "timeout"}
        params: Dict[str, Any] = {"temperature": profile.temperature, **extra_params_clean}

        if profile.provider == "gemini":
            # Use max_tokens directly without capping
            params["max_output_tokens"] = profile.max_tokens
        else:
            params["max_tokens"] = profile.max_tokens

        if profile.provider == "gemini" and profile.thinking:
            params["thinking"] = profile.thinking

        if profile.provider == "openai" and profile.reasoning_effort:
            params["reasoning_effort"] = profile.reasoning_effort
            params.setdefault("allowed_openai_params", ["reasoning_effort"])

        if response_model is not None:
            params["response_format"] = response_model

        return params

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [LiteLLMClient._to_serializable(item) for item in value]
        if isinstance(value, dict):
            return {k: LiteLLMClient._to_serializable(v) for k, v in value.items()}
        if hasattr(value, "model_dump"):
            return LiteLLMClient._to_serializable(value.model_dump())
        if hasattr(value, "dict"):
            return LiteLLMClient._to_serializable(value.dict())
        if hasattr(value, "__dict__"):
            return LiteLLMClient._to_serializable(vars(value))
        return str(value)


def create_llm_client(config: LLMConfig) -> LiteLLMClient:
    return LiteLLMClient(config)


