<div align="center">

![Easy Scaffold Logo](logo.png)

---

[![GitHub Stars](https://img.shields.io/github/stars/hamedmahdavi72/easy-scaffold?style=social)](https://github.com/hamedmahdavi72/easy-scaffold/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/hamedmahdavi72/easy-scaffold?style=social)](https://github.com/hamedmahdavi72/easy-scaffold/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/hamedmahdavi72/easy-scaffold)](https://github.com/hamedmahdavi72/easy-scaffold/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/hamedmahdavi72/easy-scaffold)](https://github.com/hamedmahdavi72/easy-scaffold/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A minimal and hackable codebase for implementing agentic scaffolds with LLMs. This framework is designed for evaluation, data generation, and future reinforcement learning research. Write your agent logic in Python, configure LLM calls in YAML, and modify anything that doesn't fit your needs.

</div>

## Quick Links

- **Current version:** **0.2.0** (see [CHANGELOG.md](CHANGELOG.md) for history)
- **Installation** - See [Installation](#installation) below
- **Tutorials** - [Adding a New Stage](tutorials/add_new_stage_tutorial.md) | [Adding a New Workflow](tutorials/add_new_workflow_tutorial.md)
- **Example Workflows** - [Agentic Grader](#example-implementations) | [IMO 2025 Agent](#example-implementations)

## What This Is

Easy Scaffold is a hackable foundation for building agentic systems. The codebase is intentionally small and straightforward. You can read through the core components in an afternoon and understand how everything fits together. This makes it easy to adapt for evaluation pipelines, data generation workflows, or RL training loops.

The philosophy is simple: agents are code, LLM calls are configuration. Your workflow classes contain the actual decision making and control flow. Stage definitions in YAML describe how to call language models, what prompts to use, and how to map inputs and outputs. This separation keeps your agent logic clean while making it easy to experiment with different prompts and models.

Whether you're evaluating model performance, generating training data, or building RL environments, the framework stays out of your way. The abstractions are thin enough that you can modify the core components directly when needed.

## Core Design Principles

### Agents as Code

Your agent workflows are plain Python classes. You write loops, conditionals, error handling, and any control flow you need. There are no rigid declarative pipelines forcing you into a specific pattern. If you need to retry a stage three times with exponential backoff, you write that logic directly in your workflow class.

```python
class MyWorkflow(BaseWorkflow):
    async def _run(self, work_item: WorkItem) -> Dict[str, Any]:
        for attempt in range(3):
            result = await self.stage("GenerateSolution", {...})
            if result["is_valid"]:
                return result
        raise Exception("Failed after retries")
```

### YAML Based Stages

LLM calls are defined as stages in YAML files. Each stage specifies a prompt template, input mappings, output handling, and optionally which model to use. This means you can iterate on prompts without touching Python code. You can create variants of stages (different prompts, different models) just by adding new YAML entries. The system handles loading prompts, calling the LLM, parsing responses, and mapping outputs back to your workflow.

```yaml
- name: AnalyzeSolution
  model_profile: "gpt5_2"  # Switch models easily
  messages:
    - role: "user"
      template_path: "prompts/analysis.md"
  input_mapping:
    problem: "work_item.problem.text"
    solution: "work_item.solution.text"
  output_mapping:
    score: "analysis_score"
```

### Schema Agnostic Data Flow

Workflows operate on canonical data models like `WorkItem`. How you get data into these models is handled by binding configurations. A binding describes how to query MongoDB, which fields to extract, and how to map them into your workflow's expected format. This means you can point the same workflow at different collections or schemas just by changing the binding configuration.

```yaml
binding:
  workflow:
    class: "easy_scaffold.workflows.agents.my_workflow.MyWorkflow"
    payload_model: "easy_scaffold.workflows.workflow_models.ProblemPayload"
  query:
    collection: "problems"
    template:
      difficulty: "hard"
  bindings:
    inputs:
      problem_id:
        from: "_id"
      problem_statement:
        from: "statement"
    outputs:
      final_score:
        to: "results.score"
```

### Multi Provider LLM Support

The framework routes chat completions through a small provider layer: `google-genai` for Gemini, the official `anthropic` SDK for Claude (`provider: anthropic`), the OpenAI Python SDK for OpenAI and OpenAI-compatible APIs (e.g. DeepSeek), and the same SDK for vLLM and completion-style calls. Set `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_BASE_URL` or `api_base` in YAML) for Anthropic. Model profiles are defined once in YAML with API keys, temperature settings, token limits, and provider-specific features like reasoning effort or thinking budgets. Stages reference these profiles by name, so switching models is as simple as changing a string in your stage config.

```yaml
# configs/llm/default.yaml
models:
  gemini_pro:
    provider: "gemini"
    model: "gemini/gemini-2.5-pro"
    temperature: 0.1
    thinking:
      type: "enabled"
      budget_tokens: 32768
  gpt5_2:
    provider: "openai"
    model: "openai/gpt-5.2"
    reasoning_effort: "high"
  claude_sonnet:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
```

### Images, vision, and image generation

Stages can attach images from blob storage with `media_attachments` (paths resolved via your configured `blob_store`). Messages are built with OpenAI-style `image_url` parts (data URLs); Gemini and Anthropic map those to native vision inputs. For OpenAI, if the prompt includes images and you use a structured Pydantic output, the client uses `chat.completions.create` and the stage parses JSON from the reply.

For **image generation**, set the model profile `task: image_generation` (OpenAI provider only), configure `blob_store` in Hydra, and set the stage `output_key` to the context key that should receive an `ImageRef` document (`key`, `content_type`, etc.). Install `boto3` with the optional extra: `pip install ".[s3]"` when using the S3 blob store.

### Comprehensive Logging

Every workflow run and stage execution is logged to MongoDB. You get stage inputs and outputs, model parameters, token usage, timing information, and any errors that occurred. This makes it easy to debug issues, analyze performance, and understand what your agents are actually doing. The detailed logs are particularly useful for evaluation and data generation workflows where you need to track what the model produced and why.

```python
# Logs automatically capture:
{
  "stage": "GenerateSolution",
  "inputs": {"problem": "..."},
  "outputs": {"solution": "..."},
  "token_stats": {"prompt_tokens": 1500, "completion_tokens": 800},
  "model": "gpt-5.2",
  "timestamp": "2025-01-24T10:30:00Z"
}
```

## Installation

Easy Scaffold requires Python 3.12 or later.

```bash
# Clone the repository
git clone https://github.com/hamedmahdavi72/easy-scaffold.git
cd easy-scaffold

# Install uv if you haven't already
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

Set up your environment variables in a `.env` file:

```bash
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
MONGO_CONNECTION_STRING=your_connection_string
DB_NAME=AoPS
```

### AWS and Secrets Manager (recommended)

On AWS, **do not rely on a `.env` file in the image**. Map [Secrets Manager](https://docs.aws.amazon.com/secretsmanager/) (or SSM Parameter Store) to **process environment variables** in your task or function definition (ECS task definition, Lambda configuration, App Runner, etc.). Use the **same variable names** as in your Hydra YAML (e.g. `OPENAI_API_KEY`, `MONGO_CONNECTION_STRING`). The CLI loads a local `.env` with `override=False`, so any value your platform already set is **never** overwritten by the file.

Optionally set **`EASY_SCAFFOLD_LOAD_DOTENV=0`** in production so the app never attempts to open `.env` (useful if you want to avoid any file-based loading).


## Architecture Overview

![Architecture Diagram](diagram.png)

```
easy-scaffold/
├── configs/
│   ├── llm/                  # Model profiles and retry settings
│   ├── stages/               # Stage definitions (prompts, mappings)
│   ├── workflow/             # Workflow bindings and runtime config
│   └── prompts/              # Prompt templates (Markdown files)
├── src/
│   └── easy_scaffold/
│       ├── workflows/
│       │   ├── agents/       # Workflow implementations
│       │   ├── configurable_stage.py  # Stage execution engine
│       │   ├── orchestrator.py        # Workflow orchestration
│       │   └── binding_resolver.py    # Data binding logic
│       ├── llm_client/       # Provider router, client, rate limiting
│       ├── db/                # MongoDB repositories
│       └── tools/             # Tool calling infrastructure
└── tutorials/                # Step-by-step guides
```

The core components are:

- **Workflows** are Python classes that implement your agent logic. They call stages, process outputs, make decisions, and coordinate multi-step processes.

- **Stages** are configured in YAML and executed by `ConfigurableStage`. A stage loads a prompt template, maps inputs, calls an LLM, and handles structured or free-form outputs.

- **Bindings** connect your data sources to workflows. They specify MongoDB queries, field mappings, and where to write results back.

- **Model Profiles** define reusable LLM configurations. Each profile specifies a provider, model name, API key, and generation parameters.

## How It Works

A workflow is a Python class that processes data by calling LLMs and making decisions. Here's what happens: you get a document from your database (like a math problem), call LLM stages to analyze or generate content, check the results, and decide what to do next—maybe call another stage, retry, or finish.

Let's say you want to build a workflow that analyzes a solution and either grades it (if complete) or improves it (if incomplete). Here's how you'd write that:

```python
class MyAgentWorkflow(AbstractWorkflow):
    async def _run(self, work_item: WorkItem, run_log: RunLog, document_id: str, workflow_name: str) -> Dict[str, Any]:
        # Start with the data from the document
        context = work_item.payload.model_dump()
        
        # Call a stage to analyze the solution
        await self._execute_stage("SolutionAnalysis", context, run_log, document_id, workflow_name)
        
        # Check the result and decide what to do
        if context.get("is_complete"):
            # Solution is complete, grade it
            await self._execute_stage("FinalGrading", context, run_log, document_id, workflow_name)
        else:
            # Solution needs work, improve it
            await self._execute_stage("ImprovementStage", context, run_log, document_id, workflow_name)
        
        context["status"] = "completed"
        return context
```

The `SolutionAnalysis` stage is defined in YAML. It tells the framework which prompt to use, which model to call, and how to map the results back into your context:

```yaml
- name: SolutionAnalysis
  response_model: "easy_scaffold.configs.pydantic_models.SolutionCompletenessEvaluation"
  messages:
    - role: "user"
      template_path: "configs/prompts/solution_analysis.md"
  input_mapping:
    problem_statement: "problem_statement"  # Reads from context["problem_statement"]
    solution_text: "solution_text"          # Reads from context["solution_text"]
  output_mapping:
    is_complete: "is_complete"               # Writes to context["is_complete"]
    completeness_score: "score"              # Writes to context["score"]
```

When you call `_execute_stage("SolutionAnalysis", ...)`, the framework:
1. Loads the prompt template from `configs/prompts/solution_analysis.md`
2. Fills in `{problem_statement}` and `{solution_text}` from your context
3. Calls the LLM with that prompt
4. Parses the response into a Pydantic model
5. Extracts `is_complete` and `completeness_score` and writes them to your context
6. Returns the result (which includes token stats, raw output, etc.)

Your workflow code then reads `context["is_complete"]` to decide whether to grade or improve the solution.

## Configuration

Configuration is organized into logical files:

- **LLM Profiles** (`configs/llm/default.yaml`) define model configurations. Each profile specifies provider, model name, API keys (via environment variables), temperature, max tokens, and provider-specific features.

- **Stage Definitions** (`configs/stages/*.yaml`) describe how to execute LLM calls. Each stage has a name, prompt template path, input/output mappings, and optionally a model profile override.

- **Workflow Bindings** (`configs/workflow/*.yaml`) connect data sources to workflows. They specify MongoDB queries, field mappings, and output persistence.

- **Prompt Templates** (`configs/prompts/**/*.md`) are Markdown files with template variables. The system loads these and substitutes values from your input mappings.

## Running Workflows

With the package installed in editable mode, you can run workflows directly:

```bash
# Make sure your virtual environment is activated
python -m easy_scaffold.cli workflow=my_workflow

# Or use uv to run it directly
uv run python -m easy_scaffold.cli workflow=my_workflow
```

You can override configuration values via command line arguments:

```powershell
$env:PYTHONPATH=".\src"; python -m easy_scaffold.cli \
  workflow=grader_agent \
  llm.default_model=gpt5_2 \
  workflow.limit=100
```

## Features

### Tool Calling

The framework includes infrastructure for LLM tool calling. Define tools as Python functions decorated with `@tool`, and the LLM client automatically handles the full tool calling loop. Tools can execute code in sandboxed environments (Docker, Modal, RestrictedPython) for safe execution.

### Structured Outputs

Stages can specify Pydantic response models. The system automatically handles structured output formatting for providers that support it (Gemini, Anthropic/Claude, OpenAI-style `parse`, and others) and falls back gracefully where the API does not.

### Rate Limiting

Built-in rate limiting per model profile. Configure requests per minute and tokens per minute limits, and the system automatically throttles calls to stay within bounds.

### Retry Logic

Configurable retry policies with exponential backoff. Handle transient errors gracefully without manual retry logic in your workflows.

### Caching

Caching is handled through bindings and agent code. Use bindings to load cached data from your database, then check in your workflow whether cached results exist before running expensive stages. This gives you full control over cache invalidation and when to use cached vs fresh data.

```python
# In your workflow
if payload.cached_clustering:
    clusters = payload.cached_clustering  # Loaded via binding
else:
    clusters = await self.stage("SolutionClustering", {...})
    # Save to database for next run
```

## Example Implementations

This codebase includes workflow implementations that demonstrate the framework's capabilities for evaluation and data generation:

### Agentic Grader (`grader_agent` workflow)

A multi-stage grading system for mathematical competition proofs that extracts reference solutions, generates problem-specific rubrics, and assigns detailed scores with error analysis. This workflow is useful for evaluation tasks where you need consistent, detailed assessment of model outputs. The implementation follows the methodology described in [RefGrader: Automated Grading of Mathematical Competition Proofs using Agentic Workflows](https://arxiv.org/abs/2510.09021).

### IMO 2025 Agent (`imo25_agent` workflow)

A verification-and-refinement pipeline for solving International Mathematical Olympiad problems. The workflow generates solution drafts, performs iterative self-improvement, and uses verification loops to ensure correctness. This is useful for both evaluation (testing model capabilities) and data generation (producing high-quality solution datasets). This implementation is based on the approach in [Winning Gold at IMO 2025 with a Model-Agnostic Verification-and-Refinement Pipeline](https://arxiv.org/abs/2507.15855).

These implementations showcase how the framework can be used to build evaluation pipelines and data generation workflows. The same architecture can be extended for RL training loops where you need to run agents, collect trajectories, and update policies.

## Getting Started

1. **Set up environment variables** for your LLM API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

2. **Define model profiles** in `configs/llm/default.yaml` for the providers you want to use

3. **Create prompt templates** in `configs/prompts/` as Markdown files with template variables

4. **Define stages** in `configs/stages/*.yaml` that reference your prompts and specify input/output mappings

5. **Write a workflow class** that implements your agent logic, calling stages as needed

6. **Create a binding** in `configs/workflow/*.yaml` that connects your data source to your workflow

7. **Run it** via the CLI

## Tutorials

Step-by-step guides are available in the `tutorials/` directory:

- **[Adding a New Stage](tutorials/add_new_stage_tutorial.md)** - Learn how to create new LLM stages by defining YAML configurations. Covers prompt templates, input/output mappings, structured responses, and best practices.

- **[Adding a New Workflow](tutorials/add_new_workflow_tutorial.md)** - Build custom agent workflows by implementing Python classes that orchestrate stages. Includes examples of control flow, error handling, and stage composition.

These tutorials walk through the complete process from configuration to implementation, with working examples you can adapt for your use case.

## Why This Design

This codebase prioritizes clarity and hackability over features. The abstractions are thin. You can see exactly what's happening at each layer. When you need to customize something for evaluation, data generation, or RL experiments, the code is straightforward enough that you can modify it directly rather than working around framework limitations.

YAML stages mean non-engineers can contribute prompts. Agents as code means you have full control over logic. Schema-agnostic bindings mean you can reuse workflows across different data sources. Multi-provider support means you're not locked into one vendor.

The goal is to give you a hackable foundation for building agentic scaffolds. Whether you're running evaluation benchmarks, generating training datasets, or experimenting with RL training loops, the framework stays minimal and adaptable. Use what you need, ignore what you don't, and modify anything that doesn't fit your use case.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
