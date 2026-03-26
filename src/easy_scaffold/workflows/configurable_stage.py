# src/easy_scaffold/workflows/configurable_stage.py
import json
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, get_args
from pydantic import BaseModel, ValidationError
from rich.pretty import pprint  # type: ignore
from rich.console import Console  # type: ignore
from rich.theme import Theme  # type: ignore

custom = Theme(
    {
        "repr.str": "bold yellow",
        "repr.number": "bright_cyan",
        "repr.brace": "white",
        "repr.bool_true": "bold green",
        "repr.bool_false": "bold red",
    }
)
console = Console(theme=custom)

from easy_scaffold.common.custom_exceptions import (
    APIServerException,
    ContentBlockedException,
    EmptyResponseException,
    RateLimitException,
    WorkflowException,
)
from easy_scaffold.common.custom_exceptions import StageExecutionError
from easy_scaffold.common.utils import (
    get_from_nested_dict,
    import_from_string,
)
from easy_scaffold.configs.pydantic_models import (
    StageConfig,
    MessageTemplate,
)
logger = logging.getLogger(__name__)


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith('`'):
        lines = s.splitlines()
        if lines and lines[0].startswith('`'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '`':
            lines = lines[:-1]
        s = '\n'.join(lines).strip()
    return s


def _assistant_content_to_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for b in content:
            if isinstance(b, dict) and b.get('type') == 'text':
                parts.append(str(b.get('text') or ''))
        return ''.join(parts) if parts else None
    return str(content)


def _messages_to_single_prompt_text(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for m in messages:
        if m.get('role') not in ('user', 'system', 'assistant'):
            continue
        txt = _assistant_content_to_text(m.get('content'))
        if txt and txt.strip():
            chunks.append(txt.strip())
    return '\n\n'.join(chunks).strip()


class ConfigurableStage:
    """A stage configured from a StageConfig object, using the native genai SDK."""

    def __init__(self, config: StageConfig, llm_client: Any, llm_config: Any, blob_store: Any = None):
        self._config = config
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._blob_store = blob_store
        self._response_model: Optional[Type[BaseModel]] = None
        self._base_model_class: Optional[Type[BaseModel]] = None  # Base class for structured output (without List wrapper)
        
        if self._config.response_model:
            response_class = import_from_string(self._config.response_model)
            self._base_model_class = response_class  # Always store base class
            if self._config.response_is_list:
                self._response_model = List[response_class]  # Keep List wrapper for our parsing logic
            else:
                self._response_model = response_class


    @property
    def name(self) -> str:
        return self._config.name

    def _load_template(self, template_config: MessageTemplate) -> str:
        """Loads template content from a file path or a raw string."""
        if template_config.template_path:
            path = Path(template_config.template_path)
            if not path.exists():
                raise FileNotFoundError(f"Prompt template not found: {path}")
            return path.read_text(encoding="utf-8")
        elif template_config.template:
            return template_config.template
        raise WorkflowException(f"Message template for stage '{self.name}' must have 'template_path' or 'template'.")

    def _build_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Builds the list of messages for the LLM call."""
        messages = []
        
        def format_template(template_str: str, ctx: Dict[str, Any]) -> str:
            for key, value_path in self._config.input_mapping.items():
                val = get_from_nested_dict(ctx, value_path)
                # Log if value is None or empty to help debug template issues
                if val is None:
                    logger.warning(f"Template variable '{key}' (path: '{value_path}') is None in context")
                elif isinstance(val, str) and not val.strip():
                    logger.warning(f"Template variable '{key}' (path: '{value_path}') is empty string in context")
                
                # Special logging for partial_solution to debug content loss
                if key == "partial_solution" and isinstance(val, str):
                    logger.info(
                        f"Template variable 'partial_solution': length={len(val)}, "
                        f"first_200_chars={repr(val[:200]) if val else 'EMPTY'}, "
                        f"last_100_chars={repr(val[-100:]) if len(val) > 100 else repr(val)}"
                    )
                
                template_str = template_str.replace(f"{{{key}}}", str(val) if val is not None else "")
            return template_str

        for msg_template in self._config.messages:
            template_content = self._load_template(msg_template)
            formatted_content = format_template(template_content, context)
            messages.append({"role": msg_template.role, "content": formatted_content})
            
        return messages
    
    def _is_empty_structured_output(self, parsed_model: Any, output_data: Dict[str, Any]) -> bool:
        """
        Check if structured output is effectively empty.
        
        Returns True if:
        - parsed_model is None (parsing failed)
        - output_data is empty (no fields extracted)
        - parsed model has only default/None values
        """
        if parsed_model is None:
            return True
        
        # Check if output_data is empty (no fields extracted via output_mapping)
        if not output_data:
            return True
        
        # Check if parsed model has only default/None values
        if isinstance(parsed_model, BaseModel):
            # Get model dict excluding defaults
            try:
                model_dict = parsed_model.model_dump(exclude_defaults=True)
                # If all fields are defaults, model_dict will be empty
                if not model_dict:
                    # Double-check: if output_mapping exists but output_data is empty,
                    # it means no fields were successfully extracted
                    if self._config.output_mapping and not output_data:
                        return True
            except Exception:
                # If we can't check, assume it's not empty
                pass
        
        # For list responses, check if list is empty
        if isinstance(parsed_model, list) and len(parsed_model) == 0:
            return True
        
        return False

    def _extract_and_parse_structured_output(self, response: Any) -> Optional[Any]:
        """
        Extract JSON string from response and convert to Pydantic model instance.
        
        Handles:
        - JSON string in message.content (Gemini/OpenAI)
        - Already parsed output from the LLM client
        - Dict or Pydantic instance fallbacks
        
        Returns:
            Pydantic model instance(s) or None if extraction fails
        """
        if self._response_model is None:
            return None
        
        # Extract base model class (handle List[Model] case)
        model_class = self._response_model
        if hasattr(model_class, '__origin__') and getattr(model_class, '__origin__') is list:
            args = get_args(model_class)
            if args:
                model_class = args[0]
            else:
                return None
        
        # Try to get parsed_output first (if the client already parsed it)
        parsed = getattr(response, "parsed_output", None)
        if parsed is not None:
            # Already a Pydantic instance
            if isinstance(parsed, BaseModel):
                result = parsed
            # Already a dict - convert to Pydantic
            elif isinstance(parsed, dict):
                try:
                    result = model_class(**parsed)
                except ValidationError as e:
                    # Validation failed - treat as parsing failure for retry logic
                    logger.debug(f"Validation failed for {model_class.__name__} (from parsed_output): {e}")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to convert parsed dict to {model_class.__name__}: {e}")
                    return None
            else:
                return None
            
            # Normalize: if response_is_list is true, ensure result is a list
            if self._config.response_is_list and not isinstance(result, list):
                return [result]
            elif not self._config.response_is_list and isinstance(result, list):
                return result[0] if result else None
            return result
        
        # Extract JSON string from message.content
        try:
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                return None
            
            # Handle both dict and object message types
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "")
            
            text_out = None
            if isinstance(content, str) and content.strip():
                text_out = content
            elif isinstance(content, list):
                text_out = _assistant_content_to_text(content)
            if not text_out or not str(text_out).strip():
                return None
            text_out = _strip_json_fences(str(text_out))
            try:
                parsed_dict = json.loads(text_out)
            except json.JSONDecodeError:
                return None
            
            # Convert dict to Pydantic model
            try:
                if isinstance(parsed_dict, list):
                    # Handle list of models - catch ValidationError for each item
                    result = []
                    for item in parsed_dict:
                        if isinstance(item, dict):
                            try:
                                result.append(model_class(**item))
                            except ValidationError as e:
                                # If any item fails validation, fail the whole list
                                logger.debug(f"Validation failed for {model_class.__name__} in list: {e}")
                                return None
                        else:
                            result.append(item)
                elif isinstance(parsed_dict, dict):
                    result = model_class(**parsed_dict)
                else:
                    logger.warning(f"Unexpected parsed JSON structure: {type(parsed_dict)}")
                    return None
            except ValidationError as e:
                # Validation failed - treat as parsing failure for retry logic
                logger.debug(f"Validation failed for {model_class.__name__}: {e}")
                return None
            
            # Normalize: if response_is_list is true, ensure result is a list
            if self._config.response_is_list and not isinstance(result, list):
                return [result]
            elif not self._config.response_is_list and isinstance(result, list):
                return result[0] if result else None
            return result

        except (IndexError, AttributeError, json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to extract structured output: {e}")
            return None

    async def execute(
        self,
        context: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        logger.info(f"--- Running Stage: {self.name} ---")
        profile = self._llm_config.resolve_profile(self._config.model_profile)

        inputs_for_log = {
            key: str(get_from_nested_dict(context, path))
            for key, path in self._config.input_mapping.items()
        }

        if profile.task == "image_generation":
            if self._blob_store is None:
                raise WorkflowException(
                    "image_generation stages require blob_store in Hydra config."
                )
            messages_ig = self._build_messages(context)
            prompt = _messages_to_single_prompt_text(messages_ig)
            if not prompt:
                raise WorkflowException("image_generation requires non-empty text in messages")
            res = await self._llm_client.generate_image(
                stage_config=self._config,
                prompt=prompt,
                generation_overrides=overrides,
            )
            ext = "png" if "png" in (res.mime_type or "").lower() else "jpg"
            mt = res.mime_type or f"image/{ext}"
            key = f"generated/{self.name}/{uuid.uuid4().hex}.{ext}"
            ref = await self._blob_store.put_bytes(key, res.image_bytes, mt)
            if not self._config.output_key:
                raise WorkflowException("image_generation stage must set output_key")
            output_data = {self._config.output_key: ref.model_dump()}
            context.update(output_data)
            if console:
                console.print("--------------------------------")
                console.print("Output Data:")
                pprint(output_data, indent_guides=True, expand_all=True, console=console)
                console.print("--------------------------------")
            return {
                "outputs": output_data,
                "raw_output": ref.model_dump(),
                "token_stats": None,
                "inputs": inputs_for_log,
                "finish_reason": None,
            }

        messages = self._build_messages(context)
        if self._config.media_attachments:
            if self._blob_store is None:
                raise WorkflowException(
                    f"Stage '{self.name}' has media_attachments but blob_store is not configured."
                )
            from easy_scaffold.media.attachments import apply_media_attachments

            messages = await apply_media_attachments(
                messages,
                self._config.media_attachments,
                context,
                self._blob_store,
                get_from_nested_dict,
            )

        # Temporary: pprint messages being sent
        if console:
            console.print("=" * 80)
            console.print(f"[bold cyan]Stage: {self.name} - Messages Being Sent:[/bold cyan]")
            console.print("=" * 80)
            pprint(messages, indent_guides=True, expand_all=True, console=console)
            console.print("=" * 80)
        else:
            logger.info(f"Stage {self.name} messages: {messages}")
        
        # Retry configuration for empty structured outputs (from LLMConfig)
        parsing_retry_config = self._llm_config.parsing_retry_config
        max_retries = parsing_retry_config.num_retries
        retry_delay = parsing_retry_config.cooldown_seconds
        
        # Initialize output_data early to ensure it's always defined, even if exceptions occur
        output_data = {}
        raw_output = None
        
        try:
            # Retry loop for empty structured outputs
            for attempt in range(max_retries + 1):
                # Pass base model class for structured output (not List wrapper)
                model_for_llm = self._base_model_class if self._base_model_class else self._response_model
                response = await self._llm_client.create(
                    stage_config=self._config,
                    messages=messages,
                    response_model=model_for_llm,
                    generation_overrides=overrides,
                )
                
                # Temporary: pprint response received
                if console:
                    console.print("=" * 80)
                    console.print(f"[bold green]Stage: {self.name} - Response Received:[/bold green]")
                    console.print("=" * 80)
                    pprint(response, indent_guides=True, expand_all=True, console=console)
                    console.print("=" * 80)
                else:
                    logger.info(f"Stage {self.name} response: {response}")

                # Reset output_data for each retry attempt
                output_data = {}
                raw_output = None
                if self._config.response_model:
                    
                    # Extract and parse structured output to Pydantic model
                    parsed_model = self._extract_and_parse_structured_output(response)
                    
                    if parsed_model is not None:
                        # Normalize to list for consistent handling
                        responses = parsed_model if isinstance(parsed_model, list) else [parsed_model]
                        # Ensure raw_output matches response_is_list expectation
                        if self._config.response_is_list and not isinstance(parsed_model, list):
                            raw_output = [parsed_model]
                        elif not self._config.response_is_list and isinstance(parsed_model, list):
                            raw_output = parsed_model[0] if parsed_model else None
                        else:
                            raw_output = parsed_model
                        
                        # Extract fields using output_mapping
                        if self._config.output_mapping:
                            for pydantic_field, context_key in self._config.output_mapping.items():
                                values = []
                                for r in responses:
                                    if isinstance(r, BaseModel) and hasattr(r, pydantic_field):
                                        values.append(getattr(r, pydantic_field))
                                
                                if values:
                                    if len(values) == 1:
                                        output_data[context_key] = values[0]
                                    else:
                                        if all(isinstance(v, str) for v in values):
                                            output_data[context_key] = "\n\n---\n\n".join(values)
                                        else:
                                            output_data[context_key] = values
                        
                        # If output_key is specified and we have a parsed model, store it as dict
                        if self._config.output_key and parsed_model is not None:
                            if isinstance(parsed_model, BaseModel):
                                output_data[self._config.output_key] = parsed_model.model_dump()
                            elif isinstance(parsed_model, list) and len(parsed_model) > 0 and isinstance(parsed_model[0], BaseModel):
                                output_data[self._config.output_key] = [item.model_dump() for item in parsed_model]
                            else:
                                output_data[self._config.output_key] = parsed_model
                        
                        # For list responses, also include the full list in outputs for logging
                        if self._config.response_is_list and raw_output is not None:
                            # Convert to list of dicts for JSON serialization
                            # Use generic "items" key instead of "clusters" for non-clustering stages
                            list_key = "clusters" if "clustering" in self.name.lower() or "cluster" in self.name.lower() else "items"
                            if isinstance(raw_output, list):
                                output_data[list_key] = [
                                    item.model_dump() if hasattr(item, "model_dump") else item
                                    for item in raw_output
                                ]
                            else:
                                output_data[list_key] = [raw_output]
                    else:
                        # Parsing/validation failed - parsed_model is None
                        # This could be due to JSON parsing failure, ValidationError, or other issues
                        raw_output = response
                        logger.warning(
                            f"Failed to parse/validate structured output for stage '{self.name}'. "
                            f"This may be due to JSON parsing failure, validation error, or other issues. "
                            f"Using raw response."
                        )
                    
                    # Check if structured output is empty and retry if needed
                    # This includes validation failures (parsed_model is None) and empty outputs
                    if self._is_empty_structured_output(parsed_model, output_data):
                        if attempt < max_retries:
                            logger.warning(
                                f"Empty or invalid structured output for stage '{self.name}' (attempt {attempt + 1}/{max_retries + 1}). "
                                f"This may be due to validation failure or empty response. Retrying in {retry_delay}s..."
                            )
                            import asyncio
                            await asyncio.sleep(retry_delay)
                            continue  # Retry the LLM call
                        else:
                            # Max retries reached - log error and raise exception
                            logger.error(
                                f"Empty structured output for stage '{self.name}' after {max_retries + 1} attempts. "
                                f"Raising EmptyResponseException."
                            )
                            # Raise EmptyResponseException to signal this to the workflow
                            raise EmptyResponseException(
                                f"Stage '{self.name}' returned empty structured output after {max_retries + 1} attempts"
                            )
                    else:
                        # Non-empty output - break out of retry loop
                        break
                else:
                    # No structured output expected - handle response and break
                    # Handle the response based on its structure. Gemini responses have a `text`
                    # attribute for direct string output, while OpenAI-like ones use `choices`.
                    output_text = ""
                    if hasattr(response, "text"):
                        output_text = response.text
                    elif hasattr(response, "choices"):
                        try:
                            first_choice = response.choices[0]
                            # Handle completion API responses (has .text attribute)
                            if hasattr(first_choice, "text"):
                                output_text = str(first_choice.text)
                            # Handle chat completion API responses (has .message attribute)
                            else:
                                message = getattr(first_choice, "message", {})
                                if isinstance(message, dict):
                                    content = message.get("content", "")
                                    if isinstance(content, list):
                                        parts = []
                                        for item in content:
                                            if isinstance(item, dict):
                                                parts.append(str(item.get("text", "")))
                                            else:
                                                parts.append(str(item))
                                        output_text = " ".join(parts).strip()
                                    else:
                                        output_text = str(content)
                                elif hasattr(message, "content"):
                                    output_text = str(message.content)
                                else:
                                    output_text = ""
                        except (IndexError, AttributeError):
                            logger.warning("Unexpected response structure for choices.")
                    else:
                        logger.warning("Could not determine response structure. Defaulting to empty string.")

                    raw_output = output_text
                    output_data = {}
                    if self._config.output_key:
                        output_data[self._config.output_key] = output_text
                    
                    # Break out of retry loop after handling non-structured output
                    break

                # The result processor logic is now removed from the stage.
                # It's the responsibility of the calling workflow class.
            
            if console:
                console.print("--------------------------------")
                console.print("Output Data:")
                pprint(output_data, indent_guides=True, expand_all=True, console=console)
                console.print("--------------------------------")
            else:
                logger.debug("Stage output data: %s", output_data)

            token_payload = self._llm_client.extract_token_stats(response) or {}
            if isinstance(token_payload, dict) and not any(token_payload.values()):
                token_payload = None
            
            # Extract finish_reason if available
            finish_reason = None
            if hasattr(response, "choices") and len(response.choices) > 0:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
            
            context.update(output_data)

            return {
                "outputs": output_data,
                "raw_output": raw_output,
                "token_stats": token_payload,
                "inputs": inputs_for_log,
                "finish_reason": finish_reason,
            }

        except ContentBlockedException as err:
            raise StageExecutionError(
                status="blocked",
                message=str(err),
                inputs=inputs_for_log,
                cause=err,
            ) from err
        except APIServerException as err:
            raise StageExecutionError(
                status="api_error",
                message=str(err),
                inputs=inputs_for_log,
                cause=err,
            ) from err
        except RateLimitException as err:
            raise StageExecutionError(
                status="rate_limited",
                message=str(err),
                inputs=inputs_for_log,
                cause=err,
            ) from err
        except EmptyResponseException as err:
            raise StageExecutionError(
                status="empty_response",
                message=str(err),
                inputs=inputs_for_log,
                cause=err,
            ) from err
        except WorkflowException as err:
            raise StageExecutionError(
                status="failed",
                message=str(err),
                inputs=inputs_for_log,
                cause=err,
            ) from err
        except Exception as e:
            # Handle non-retryable, immediate exceptions
            error_message = str(e)
            status = "failed"
            raise StageExecutionError(
                status=status,
                message=f"Error in stage {self.name}: {error_message}",
                inputs=inputs_for_log,
                cause=WorkflowException(error_message),
            ) from e


class StageFactory:
    """Creates and caches configurable stage instances."""

    def __init__(self, stage_configs: List[StageConfig], llm_client: Any, llm_config: Any, blob_store: Any = None):
        self._stage_configs = {stage.name: stage for stage in stage_configs}
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._blob_store = blob_store
        self._stages: Dict[str, ConfigurableStage] = {}

    def create_stage(self, name: str) -> ConfigurableStage:
        if name not in self._stages:
            if name not in self._stage_configs:
                raise ValueError(f"Stage '{name}' not defined in workflow configuration.")
            stage_config = self._stage_configs[name]
            self._stages[name] = ConfigurableStage(stage_config, self._llm_client, self._llm_config, blob_store=self._blob_store)
        return self._stages[name]


