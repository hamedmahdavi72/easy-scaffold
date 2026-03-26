# Adding a New Workflow

Workflows are where you decide how stages fit together. Each workflow takes a `WorkItem` (your problem/data) and calls stages in whatever order makes sense—loops, conditionals, whatever you need.

---

## How Workflows Work

### The Big Picture

Think of a workflow as a state machine:

1. You get a `WorkItem` with your data
2. Build a context dictionary from the payload
3. Call stages (sequentially or conditionally). Each stage reads from context, calls an LLM, writes results back
4. Use context values to decide what to do next (loop, branch, exit)
5. Return the final context (gets saved to the database)

### Key Ideas

- Stages are stateless—each call is independent. All state lives in the context dictionary.
- Context is mutable—stages update it, and later stages see those changes.
- Workflows control the flow—your Python code decides when and how stages run.
- Stages handle LLM details—prompt formatting, API calls, parsing—you don't worry about it.

### Example Flow

```
WorkItem (payload) 
    ↓
Initialize Context: {"problem": "...", "attempt": 1}
    ↓
Stage 1: GenerateSolution
    → Reads: context["problem"]
    → Calls LLM
    → Writes: context["solution"] = "..."
    ↓
Stage 2: VerifySolution
    → Reads: context["problem"], context["solution"]
    → Calls LLM
    → Writes: context["is_correct"] = True
    ↓
Decision: if context["is_correct"]:
    → Return context
    else:
    → Stage 3: FixSolution
        → Reads: context["solution"], context["errors"]
        → Writes: context["solution"] = "..."
    → Loop back to Stage 2
```

---

## Context Variables

### What is Context?

Context is just a Python dictionary. It's the workflow's shared memory:

- Starts with data from `WorkItem.payload` (via `model_dump()`)
- Grows as stages add their outputs
- Sticks around for the whole workflow
- Ends up as the return value (saved to MongoDB)

### Context Lifecycle

```python
# 1. Initialization (in _run method)
context: Dict[str, Any] = work_item.payload.model_dump()
# context = {"problem_id": "123", "statement": "Solve x^2 = 4", ...}

# 2. Stage execution updates context
result = await stage.execute(context)
context.update(result["outputs"])
# context now includes: {"problem_id": "...", "statement": "...", "solution": "x = ±2", ...}

# 3. Workflow reads context for decisions
if context.get("is_correct"):
    return context

# 4. Final context is returned and persisted
return context  # Saved to MongoDB via binding resolver
```

### Context Variable Access Patterns

**Reading**: Stages use `input_mapping` (e.g., `"problem_statement": "problem.statement"` accesses `context["problem"]["statement"]`). Workflows read directly: `context.get("solution")`. Nested paths work fine.

**Writing**: Stages write via `output_mapping` or `output_key` (automatic). Workflows write directly: `context["attempt"] = 1`. Updates are in-place.

**Common keys**: Input data (`problem`, `problem_statement`), stage outputs (`solution`, `is_correct`), workflow state (`attempt`, `status`), metadata (`run_log`, `document_id`).

### Tips

- Use descriptive keys (`"solution_text"` not `"s"`)
- Initialize workflow state early
- Use `.get()` with defaults for optional values
- Comment what keys your stages expect/produce

---

## How Stages Call LLMs

### The Abstraction

You don't call LLM APIs directly. Instead:

1. **Configure** the stage in YAML (prompt, model, structured output)
2. **Execute** the stage with a context dictionary
3. **Receive** parsed, validated results

### What Actually Happens

When you call `stage.execute(context)`:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Build Messages                                          │
│    - Load template from file or use inline template        │
│    - Replace {variables} using input_mapping                │
│    - Create message list: [{"role": "system", ...}, ...]   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Resolve Model Profile                                   │
│    - Use model_profile from stage config or default         │
│    - Merge generation_config overrides                      │
│    - Get API keys, temperature, max_tokens, etc.           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Call LLM client (provider router)                        │
│    - Handle rate limiting                                   │
│    - Support completion_mode vs chat completion             │
│    - Handle tool calling (if tools specified)               │
│    - Retry on transient errors                              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Parse Response                                           │
│    - If response_model: Parse JSON → Pydantic model         │
│      - Validate against schema                              │
│      - Retry if parsing fails (configurable)                │
│    - If no response_model: Extract text content             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Map Outputs                                              │
│    - Extract fields via output_mapping                      │
│    - Or store entire response via output_key                │
│    - Return: {"outputs": {...}, "raw_output": ..., ...}    │
└─────────────────────────────────────────────────────────────┘
```

### Example

**You write:**
```python
result = await self._execute_stage("GenerateSolution", context, ...)
context.update(result["outputs"])
solution = context["solution"]
```

**What happens under the hood:**
1. Stage loads template: `"Solve: {problem_statement}"`
2. Replaces variables: `"Solve: Find all x such that x^2 = 4"`
3. Calls LLM: `route_chat_completion` / OpenAI SDK depending on provider
4. LLM returns: `{"choices": [{"message": {"content": "x = ±2"}}]}`
5. Stage extracts: `output_data = {"solution": "x = ±2"}`
6. Context gets updated: `context["solution"] = "x = ±2"`

### Error Handling

Stages handle common errors automatically: rate limits (retry with backoff), API errors (retry transient ones, fail fast on permanent), parsing failures (retry if validation fails), empty responses (retry if configured).

Errors come wrapped in `StageExecutionError` with status codes like `rate_limited`, `api_error`, `empty_response`.

### Token Usage

Every stage returns token stats. They're logged automatically to MongoDB for cost tracking.

---

## Why Workflows Are Python Classes

- **Full Control Flow** – Use `for`/`while`, conditionals, early exits, and exception handling.
- **Direct Composition** – Call utility functions, repositories, or other Python modules as needed.
- **Unit Testable** – Each workflow is a normal class with clear inputs and outputs.
- **Configuration Driven Inputs** – Hydra injects runtime settings (`self._config`) so workflows stay configurable.

## Step-by-Step: Creating a Custom Workflow

### 1. Create the Workflow File

Add a new file under `src/easy_scaffold/workflows/agents/`, e.g. `simple_grader.py`.

### 2. Implement the Workflow Class

Inherit from `AbstractWorkflow` and implement `_run`. The base class handles logging lifecycle, run status updates, and stage logging.

```python
# src/easy_scaffold/workflows/agents/simple_grader.py
from typing import Any, Dict

from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import WorkItem, ProblemPayload
from easy_scaffold.db.pydantic_models import RunLog


class SimpleGraderWorkflow(AbstractWorkflow[ProblemPayload]):
    """
    Simple workflow that verifies and grades a solution.
    """

    async def _run(
        self,
        work_item: WorkItem[ProblemPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        # Initialize context from payload
        # This creates the initial state that stages will read from
        context: Dict[str, Any] = work_item.payload.model_dump()
        # context now contains: {"problem_id": "...", "statement": "..."}
        
        # Add workflow metadata
        context["run_log"] = run_log
        context["document_id"] = document_id

        # Step 1: Generate a verification report
        # The stage reads context["statement"] via input_mapping
        # and writes context["verification_report"] via output_key
        verification_result = await self._execute_stage(
            "GenerateVerification", 
            context, 
            run_log, 
            document_id, 
            workflow_name
        )
        # context now includes: {"problem_id": "...", "statement": "...", "verification_report": "..."}

        # Step 2: Interpret verification output
        # This stage reads context["verification_report"] and writes context["is_correct"]
        interpret_result = await self._execute_stage(
            "InterpretVerification", 
            context, 
            run_log, 
            document_id, 
            workflow_name
        )
        # context now includes: {"...", "is_correct": True/False}

        # Decision making based on context
        is_correct = context.get("is_correct", False)
        context["final_grade"] = "Pass" if is_correct else "Fail"
        context["status"] = "completed"
        
        # Return final context (will be persisted to MongoDB)
        return context
```

**Key Points:**
- Use `self._execute_stage()` instead of calling `stage.execute()` directly (handles logging)
- Context is updated automatically by `_execute_stage()` via `context.update(result["outputs"])`
- Read context values after stage execution to make decisions
- Return the final context dictionary (it gets persisted to MongoDB)

### 3. Wire It Up

Point to your workflow class in the binding YAML:

```yaml
# configs/workflow/simple_grader.yaml
workflow:
  class: "easy_scaffold.workflows.agents.simple_grader.SimpleGraderWorkflow"
  payload_model: "easy_scaffold.workflows.workflow_models.ProblemPayload"
```

Then select the binding via Hydra:

```powershell
$env:PYTHONPATH=".\src"; python -m easy_scaffold.cli workflow=simple_grader
```

### 4. Understanding Stage Results

When you call `self._execute_stage()`, it returns a dictionary with:

```python
result = await self._execute_stage("StageName", context, run_log, document_id, workflow_name)

# Result structure:
{
    "outputs": {
        "field1": "value1",  # Mapped outputs (via output_mapping or output_key)
        "field2": "value2"
    },
    "raw_output": <Pydantic instance or string>,  # The actual parsed model or raw text
    "token_stats": {
        "prompt_tokens": 150,
        "completion_tokens": 75,
        "total_tokens": 225
    },
    "inputs": {
        "template_var1": "value used",  # Values that were substituted into templates
        "template_var2": "value used"
    },
    "finish_reason": "stop"  # Why generation stopped (if available)
}
```

**Important:** `_execute_stage()` automatically updates the context with `result["outputs"]`, so you don't need to call `context.update()` manually.

**Accessing Results:**
- **For decisions**: Read from `context` after stage execution: `is_correct = context.get("is_correct")`
- **For processing**: Access `raw_output` if you need the full Pydantic model: `parsed_model = result["raw_output"]`
- **For logging**: Token stats are automatically logged, but you can access them: `tokens = result.get("token_stats")`

**Example with Structured Output:**
```python
# Stage returns structured Pydantic model
result = await self._execute_stage("AnalyzeSolution", context, ...)

# Access the parsed model directly
analysis_model = result["raw_output"]  # Pydantic instance
if analysis_model.is_complete:
    score = analysis_model.score

# Or read from context (if output_mapping was configured)
is_complete = context.get("is_complete")  # Same value, extracted via mapping
```

---

## Tips

- Access runtime settings via `self._config` (Hydra injects workflow-level config)
- Need richer inputs? Define a new payload model in `workflow_models.py`
- Reusing stages is fine—the factory caches instances
- Logging is automatic (inputs, outputs, token stats). Exceptions mark the run as failed

Workflows stay simple and focused. Everything else (data loading, prompts, LLM config) is declarative and reusable.
