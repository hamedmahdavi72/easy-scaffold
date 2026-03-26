### What is a Stage?

A stage is just a single LLM call. That's it. Each one does one thing—generate, refine, verify, correct, whatever. Workflows string stages together to build more complex behavior.

---

### How Stages Work

You configure stages in YAML—prompts, mappings, which model to use, what format you want back. The stage runtime handles all the LLM calls, parsing responses, and updating the workflow context.

#### Stage Configuration

Here's what you can configure in your stage YAML files:

- `name`: Unique identifier (e.g., `InitialDraft`, `GenerateVerification`).
- `messages`: Ordered list of message templates (`system`, `user`, `assistant`, `model`).
- `input_mapping`: Maps template variables to values inside the workflow context.
- `response_model`: (Optional) Python path to a Pydantic model for structured responses.
- `response_is_list`: (Optional) Expect a list of models instead of a single instance.
- `output_mapping`: (When using `response_model`) Maps model attributes into the workflow context.
- `output_key`: (When *not* using `response_model`) Key where the raw string output is stored.
- `model_profile`: (Optional) Name of the LLM profile defined in `configs/llm/default.yaml` to override the default model.
- `generation_config`: (Optional) Per-stage generation overrides (e.g., alternative temperature).
- `tools`: (Optional) List of tool references for function calling (e.g., `["math.calculate", "code.execute"]`).

---

### What You Can Do

Stages support a bunch of features:

#### Model Profile Selection

Pick a different model for each stage:

```yaml
- name: QuickAnalysis
  model_profile: "gemini_flash_lite"  # Use cheaper/faster model
  messages:
    - role: "user"
      template: "Analyze: {problem_text}"
  input_mapping:
    problem_text: "problem.statement"
  output_key: "analysis"
```

Model profiles live in `configs/llm/default.yaml`. You can set:
- Provider (`gemini`, `openai`, `deepseek`, `vllm`)
- Temperature and token limits
- Completion mode (for models that need prompt-based input instead of messages)
- Provider-specific features like Gemini thinking blocks or OpenAI reasoning effort
- Timeouts per model
- Context window size (for dynamic token limit adjustment)

#### Structured Output

Get validated, typed responses using Pydantic models:

```yaml
- name: StructuredAnalysis
  response_model: "easy_scaffold.configs.pydantic_models.SolutionAnalysis"
  messages:
    - role: "user"
      template_path: "configs/prompts/analysis.md"
  input_mapping:
    problem: "problem.statement"
  output_mapping:
    is_complete: "analysis.is_complete"
    score: "analysis.score"
    reasoning: "analysis.reasoning"
```

The stage automatically validates responses against your Pydantic schema. If parsing fails, it retries (configurable). You can get a single model or a list (`response_is_list: true`). Map specific fields to context keys, or dump the whole model.

#### Template Variables

Use `{variable_name}` in your templates. The stage fills them from the workflow context:

```yaml
- name: GenerateSolution
  messages:
    - role: "system"
      template: "You are a math problem solver."
    - role: "user"
      template_path: "configs/prompts/solve.md"  # Contains: "Solve: {problem_statement}"
  input_mapping:
    problem_statement: "problem.text"  # Nested path support
    current_attempt: "draft.text"
  output_key: "solution"
```

You can load templates from files (`template_path`) or write them inline (`template`). File-based templates are easier to reuse.

Input mapping supports nested paths like `"problem.statement"` to access `context["problem"]["statement"]`. Values get converted to strings automatically. The stage warns you if a variable is missing or empty.

#### Output Handling

Two ways to handle outputs:

With structured output (`response_model`), map specific fields:
```yaml
output_mapping:
  field_name: "context_key"
```

Without structured output, store the raw string:
```yaml
output_key: "raw_solution"
```

You can't use both at the same time—pick one.

#### Completion Mode

Some models need completion API instead of chat completion:

```yaml
# In configs/llm/default.yaml
models:
  custom_vllm:
    provider: "vllm"
    model: "meta-llama/Llama-3.1-70B-Instruct"
    completion_mode: true  # Use completion API
    thinking_block_token: "<think>"  # Optional: token marking thinking blocks
    temperature: 0.7
    max_tokens: 4096
```

Useful for vLLM servers, models that need prompt-based input, or custom deployments. Note that tools aren't supported in completion mode, and messages get converted to a single prompt string.

#### Generation Config Overrides

Override model settings for a specific stage:

```yaml
- name: CreativeGeneration
  model_profile: "gemini_pro"
  generation_config:
    temperature: 1.2  # Override default temperature
    max_tokens: 5000  # Override token limit
  messages:
    - role: "user"
      template: "Create: {prompt}"
  input_mapping:
    prompt: "user_request"
  output_key: "generated_text"
```

You can override `temperature`, `max_tokens`, or other parameters supported by the provider.

#### Tool Calling

Enable function calling (you'll need to register tools first):

```yaml
- name: CalculatorStage
  tools: ["math.calculate", "math.solve_equation"]
  messages:
    - role: "user"
      template: "Calculate: {expression}"
  input_mapping:
    expression: "math_problem"
  output_key: "result"
```

Tools need to be registered in the tool manager. Also, completion mode doesn't support tools.

#### **Multiple Message Roles**

Build conversation context with multiple messages:

```yaml
- name: ConversationalStage
  messages:
    - role: "system"
      template: "You are a helpful assistant."
    - role: "user"
      template: "Problem: {problem}"
    - role: "assistant"
      template: "Previous attempt: {previous_solution}"
    - role: "user"
      template: "Now improve it considering: {feedback}"
  input_mapping:
    problem: "problem.text"
    previous_solution: "attempt.text"
    feedback: "critique.text"
  output_key: "improved_solution"
```

Supported roles: `system`, `user`, `assistant` (for context), and `model` (Gemini-specific).

#### Provider-Specific Features

**Gemini**: Enable thinking blocks with `thinking.type: "enabled"` and set a token budget. Use `role: "model"` for model responses in context.

**OpenAI**: Set `reasoning_effort: "high"` for o1/o3 models. Response format is automatic when using `response_model`.

**DeepSeek**: Standard chat completion, supports structured outputs.

**vLLM**: Supports completion mode and custom API endpoints via `api_base`.

#### How Stages Execute

When you call a stage, it:
1. Builds messages by pulling values from context and filling templates
2. Calls the LLM (handles rate limiting, retries, etc.)
3. Parses the response (validates against Pydantic schema if you have one)
4. Updates context with outputs
5. Returns a result dictionary:
  ```python
  {
      "outputs": {...},           # data added to context
      "raw_output": <object>,     # Pydantic instance or string
      "token_stats": {...},       # Token usage details (if available)
      "inputs": {...},            # values used in prompt templates
  }
  ```

The workflow logs this automatically and updates the context.

`StageFactory` creates and caches stage instances. Workflows get stages via `self._stage_factory.create_stage("StageName")`.

---

### Adding a New Stage

1. Define the stage in YAML (e.g., `configs/stages/imo25_agent.yaml`):
   ```yaml
   - name: SummarizeAttempt
     model_profile: gpt5_mini        # optional override
     messages:
       - role: system
         template: |
           You are a concise mathematical summarizer.
       - role: user
         template_path: prompts/summarize_attempt.txt
     input_mapping:
       attempt_text: "draft_solution.text"
     output_key: "summary"
   ```

2. **Reference the stage in a workflow**:
   ```python
   stage = self._stage_factory.create_stage("SummarizeAttempt")
   result = await stage.execute(context)
   context.update(result["outputs"])
   summary_text = result["outputs"].get("summary", "")
   ```

3. Check token usage if you want (optional):
   ```python
   token_stats = result.get("token_stats")
   if token_stats:
       self._logger.debug(f"Stage tokens: {token_stats}")
   ```

That’s it—no additional Python wiring is required unless you need custom prompt templates or structured response models.

---

### Best Practices

- Keep `input_mapping` granular and rely on canonical field names defined in workflow payloads.
- Use `response_model` for any structured output you want validated; otherwise, stick to `output_key`.
- Prefer referencing a `model_profile` instead of hardcoding provider-specific params inside stage YAML.
- When multiple stages share behavior, extract prompt templates into files for reuse (`prompts/` directory).

With these patterns, adding new behaviors to the agent is primarily a configuration exercise. Workflows remain concise, readable, and focused on orchestration logic.***
