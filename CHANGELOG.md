# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-26

### Added

- Hydra presets for **Amazon DocumentDB**: `configs/db/documentdb.yaml` and `configs/main_documentdb.yaml` (same `MongoRepository` + Motor as MongoDB).
- **Anthropic (Claude)** chat support via the official `anthropic` SDK (`provider: anthropic`), including tool calling and structured JSON outputs (`output_config`), with OpenAI-shaped responses for the existing LLM client.
- **Vision / multimodal:** user messages may include OpenAI-style `image_url` parts (data URLs); Gemini and Anthropic providers map them to native image inputs. OpenAI chat uses `chat.completions.create` (not `parse`) when images are present so structured outputs can still be parsed in the stage.
- **Image generation stages:** model profiles with `task: image_generation` (OpenAI only) call the Images API; bytes are stored via optional Hydra `blob_store` (e.g. S3) and an `ImageRef` is written to `output_key`.
- **`media_attachments` on stages:** load images from blob storage into the prompt for vision models.
- **Media layer:** `ImageRef`, abstract blob store, S3 implementation (`pip install .[s3]` for `boto3`), and Hydra example under `configs/blob/`.
- **Environment bootstrap (pattern A):** CLI loads `.env` with `override=False` so platform-injected secrets win; optional `EASY_SCAFFOLD_LOAD_DOTENV=0` skips file load (e.g. AWS).

### Changed

- **LLM stack:** removed LiteLLM; chat routing uses a small in-tree provider layer (`google-genai` for Gemini, OpenAI Python SDK for OpenAI, DeepSeek-compatible bases, and existing vLLM/completion paths).
- Completion-mode prompt inspection uses **logging** instead of printing to stdout.

### Changed (developer ergonomics)

- Replaced redundant import-time `try` / `except ImportError` for packages already declared in `pyproject.toml` (e.g. `transformers`, `rich`, built-in tool modules); Unix `resource` import gated by platform instead of `ImportError`.

## [0.1.0] - 2026-01-25

### Added

- Initial **easy-scaffold** package: Hydra-driven workflows, configurable stages, MongoDB repository, multi-provider LLM configuration, tutorials, and example workflow configs.
