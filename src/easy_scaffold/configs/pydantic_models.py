# src/easy_scaffold/configs/pydantic_models.py
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easy_scaffold.workflows.workflow_models import WorkflowBindingConfig


class DbConfig(BaseModel):
    name: str
    connection_string: str
    data_db_name: str
    logs_db_name: str
    data_collection: str
    logs_collection: str


class MessageTemplate(BaseModel):
    role: Literal["system", "user", "assistant", "model"]
    template_path: Optional[str] = None
    template: Optional[str] = None




class MediaAttachment(BaseModel):
    """Attach an image from context (ImageRef dict) to a message by index."""

    source: str = Field(
        description="Nested context path to an ImageRef-compatible dict (same as input_mapping paths)"
    )
    message_index: int = Field(
        default=0,
        ge=0,
        description="Index into stage.messages / built message list to augment with the image",
    )

class VerificationOutput(BaseModel):
    verdict: str = Field(
        description="The verdict, typically 'yes' or 'no', indicating if the condition was met."
    )


# Grader workflow models
class SolutionClusterItem(BaseModel):
    """Represents a single cluster of solutions."""
    class_id: str = Field(description="Cluster identifier like 'C1', 'C2', etc.")
    main_steps: List[str] = Field(description="Ordered list of strategic insights/main steps")
    representative_post_id: int = Field(description="Post index/ID (1-based) of the chosen representative solution from the input list")


class SolutionClusteringOutput(BaseModel):
    """Output from solution clustering stage."""
    clusters: List[SolutionClusterItem] = Field(description="List of solution clusters, each representing solutions with the same strategic approach")


class SimilarityMatchOutput(BaseModel):
    """Output from similarity assessment stage."""
    closest_rep_id: str = Field(description="Cluster ID that best matches the student solution (e.g., 'C1', 'C2')")
    justification: str = Field(description="Explanation of why this representative best matches the student's approach")


class SolutionCompletenessEvaluation(BaseModel):
    """Evaluation of solution completeness and cleanliness."""
    is_complete: bool = Field(description="Whether solution is complete and cleanly written")
    completeness_score: float = Field(description="Score 0-1 for completeness")
    cleanliness_score: float = Field(description="Score 0-1 for cleanliness")
    has_errors: bool = Field(description="Whether the solution contains mathematical errors")
    errors: List[ErrorItem] = Field(
        default_factory=list,
        description="List of errors found in the solution. Empty list if has_errors=False."
    )
    reasoning: str = Field(description="Explanation of the evaluation, including error analysis if applicable")


class SolutionEleganceEvaluation(BaseModel):
    """Evaluation of solution creativity and elegance."""
    elegance_score: float = Field(description="Score 0-1 for elegance")
    creativity_score: float = Field(description="Score 0-1 for creativity")
    overall_score: float = Field(description="Combined score 0-1 (weighted average)")
    reasoning: str = Field(description="Explanation of the evaluation")


class BestOfKSelectionOutput(BaseModel):
    """Output from best-of-k selection stage."""
    selected_index: int = Field(
        description="Index (0-based) of the selected best solution from the provided list"
    )
    reasoning: str = Field(
        description="Detailed explanation of why this solution was selected as the best"
    )
    comparison_notes: Optional[str] = Field(
        default=None,
        description="Brief comparison notes about all solutions considered"
    )


class RelaxedCompletenessEvaluation(BaseModel):
    """Evaluation for 'legit and fixable' solutions."""
    is_legit_and_fixable: bool = Field(description="Whether solution is legit and can be fixed")
    fixability_score: float = Field(description="Score 0-1 for fixability")
    issues: List[str] = Field(description="List of issues found that need fixing")
    reasoning: str = Field(description="Explanation of the evaluation")


class SolutionFixerOutput(BaseModel):
    """Output from solution fixer."""
    fixed_solution: str = Field(description="The fixed solution text")
    changes_made: str = Field(description="Description of what was fixed")
    reasoning: str = Field(description="Explanation of the fixes")


class ErrorItem(BaseModel):
    """Represents an error found in a student solution."""
    type: str = Field(description="Error type from predefined categories")
    description: str = Field(description="Explanation of the error")
    location: str = Field(description="Precise part of the solution where the error occurs")


class OverallAssessment(BaseModel):
    """Overall assessment of a graded solution."""
    score: int = Field(ge=0, le=7, description="Final score out of 7 points")
    rationale: str = Field(description="Concise rationale for the score")


class GradingResultOutput(BaseModel):
    """Complete grading result output."""
    overall_assessment: OverallAssessment
    solution_structure_analysis: str = Field(description="Main steps vs substeps and high-level logic assessment")
    substep_error_analysis: List[ErrorItem] = Field(default_factory=list, description="Systematic identification of errors")
    cross_solution_consistency: str = Field(description="Comparison against reference solution, contradictions identified")
    error_propagation_analysis: str = Field(description="Propagation chains using structured syntax")
    rubric_milestone_assessment: str = Field(description="Detailed evaluation against rubric criteria with justification")
    clarity_structure_notation: str = Field(description="Comments on clarity, organization, and notation consistency")
    constructive_feedback: str = Field(description="Suggestions for improvements or summary of core reason for failure")


class MilestoneItem(BaseModel):
    """A milestone in the marking scheme."""
    points: str = Field(description="Point value as integer or range (e.g., '1-2' or '3')")
    description: str = Field(description="Description of the logical truth required")


class DesignedMarkingScheme(BaseModel):
    """The designed marking scheme."""
    summary: str = Field(description="Brief description of the logic path required")
    milestones: List[MilestoneItem] = Field(description="List of milestones with point allocations")


class ErrorFeedback(BaseModel):
    """An error found in the student solution."""
    location: str = Field(description="Quote from text where error occurs")
    issue: str = Field(description="Specific logic error")
    severity: Literal["High", "Medium", "Low"] = Field(description="Severity of the error")


class FeedbackDetail(BaseModel):
    """Detailed feedback on the student solution."""
    achieved_milestones: List[str] = Field(description="List of milestones from the scheme the student hit")
    missed_milestones: List[str] = Field(description="List of milestones missed")
    errors: List[ErrorFeedback] = Field(default_factory=list, description="List of errors found")


class GradeResult(BaseModel):
    """The grade result."""
    score: int = Field(ge=0, le=7, description="Final score out of 7 points")
    classification: str = Field(description="Classification of the solution (e.g., 'Substantial Progress')")


class GraderJudgeOutput(BaseModel):
    """Output from grader judge prompt (auto-rubric generation & grading)."""
    designed_marking_scheme: DesignedMarkingScheme = Field(description="The rubric designed from the problem/reference solution")
    overall_assessment: GradeResult = Field(description="The final grade and classification")
    feedback: FeedbackDetail = Field(description="Detailed feedback on achieved/missed milestones and errors")


class StageConfig(BaseModel):
    name: str
    messages: List[MessageTemplate]
    response_model: Optional[str] = None
    response_is_list: bool = False
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    output_key: Optional[str] = None
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    model_profile: Optional[str] = None
    tools: Optional[List[str]] = None  # List of tool references like "math.calculate" or "math"
    media_attachments: Optional[List[MediaAttachment]] = Field(
        default=None,
        description="Resolve ImageRef from context and attach as vision input to a message",
    )

    @model_validator(mode="after")
    def check_exclusive_output_fields(self):
        if self.output_key is not None and self.output_mapping:
            raise ValueError(
                "'output_key' and 'output_mapping' cannot be used at the same time."
            )
        if self.response_model is None and self.output_key is None:
            raise ValueError(
                "Must provide 'output_key' when 'response_model' is not set."
            )
        if self.response_model is not None and not self.output_mapping and self.output_key is None:
            raise ValueError(
                "Must provide 'output_mapping' or 'output_key' when 'response_model' is set."
            )
        return self


class ModelProfile(BaseModel):
    provider: Literal["gemini", "openai", "deepseek", "vllm", "anthropic"]
    task: Literal["chat", "image_generation"] = Field(
        default="chat",
        description="chat: language model; image_generation: image API (e.g. OpenAI Images)",
    )
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float
    max_tokens: int
    thinking: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    completion_mode: bool = Field(default=False, description="Use completion API instead of chat completion")
    thinking_block_token: Optional[str] = Field(default=None, description="Opening token for thinking block (e.g., '`<think>`')")
    context_window: int = Field(
        default=32768, 
        description="Total context window size for the model. Used for dynamic token limit adjustment."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Request timeout in seconds. If None, falls back to extra_params.timeout or provider-specific defaults."
    )
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    def get_timeout(self) -> float:
        """
        Get timeout value with fallback logic.
        
        Returns:
            Timeout in seconds. Defaults: 600s for most models, 1200s for long-context models (>100k tokens).
        """
        # Explicit timeout takes precedence
        if self.timeout is not None:
            return self.timeout
        
        # Backward compatibility: check extra_params
        if "timeout" in self.extra_params:
            return float(self.extra_params["timeout"])
        
        # Provider-specific defaults based on context window
        # Long-context models (>100k tokens) get higher timeout
        if self.max_tokens > 100000:
            return 1200.0  # 20 minutes for long-context models
        
        # Default timeout for most models
        return 600.0  # 10 minutes

    @model_validator(mode="after")
    def _anthropic_chat_only(self) -> "ModelProfile":
        if self.provider == "anthropic" and self.completion_mode:
            raise ValueError(
                "provider 'anthropic' does not support completion_mode; use chat completion only."
            )
        return self

    @model_validator(mode="after")
    def _image_generation_constraints(self) -> "ModelProfile":
        if self.task == "image_generation":
            if self.provider != "openai":
                raise ValueError(
                    "task 'image_generation' is only supported for provider 'openai' in this release."
                )
            if self.completion_mode:
                raise ValueError("task 'image_generation' cannot be used with completion_mode.")
        return self


class RetrySettings(BaseModel):
    num_retries: int = 0
    cooldown_seconds: float = 0.0
    max_backoff_seconds: Optional[float] = Field(default=None, description="Cap exponential backoff at this value (seconds)")


class RateLimitConfig(BaseModel):
    """Rate limit configuration for a specific model."""
    requests_per_minute: int
    input_tokens_per_minute: int


class LLMConfig(BaseModel):
    default_model: str
    models: Dict[str, ModelProfile]
    retry_config: RetrySettings = Field(default_factory=RetrySettings)
    parsing_retry_config: RetrySettings = Field(
        default_factory=lambda: RetrySettings(num_retries=2, cooldown_seconds=1.0),
        description="Retry configuration for parsing/validation failures (empty structured output)"
    )
    rate_limits: Optional[Dict[str, RateLimitConfig]] = None

    @model_validator(mode="after")
    def _validate_default_model(self) -> "LLMConfig":
        if self.default_model not in self.models:
            raise ValueError(
                f"default_model '{self.default_model}' is not defined in llm.models"
            )
        return self

    def resolve_profile(self, name: Optional[str] = None) -> ModelProfile:
        key = name or self.default_model
        if key not in self.models:
            raise ValueError(f"Unknown model profile '{key}'")
        return self.models[key]


class StageLoggingConfig(BaseModel):
    """Configuration for controlling stage-level logging within a workflow."""
    enabled: bool = Field(default=True, description="Global toggle for stage logging (default: True)")
    exclude_stages: Optional[List[str]] = Field(
        default=None,
        description="List of stage names to exclude from logging (blacklist pattern)"
    )
    include_stages: Optional[List[str]] = Field(
        default=None,
        description="List of stage names to include in logging (whitelist pattern, mutually exclusive with exclude_stages)"
    )

    @model_validator(mode="after")
    def check_exclusive_patterns(self):
        """Ensure exclude_stages and include_stages are not both specified."""
        if self.exclude_stages and self.include_stages:
            raise ValueError(
                "Cannot use both 'exclude_stages' and 'include_stages' in logging config. "
                "Use one or the other."
            )
        return self


class RetryConfig(BaseModel):
    """Configuration for retrying transient errors."""
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff multiplier")
    initial_delay: float = Field(default=1.0, description="Initial delay in seconds before first retry")


class ErrorHandlingRetryConfig(BaseModel):
    """Configuration for retry behavior by error type."""
    transient_errors: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        description="Retry configuration for transient errors (rate limits, API errors)"
    )
    semi_transient_errors: RetryConfig = Field(
        default_factory=lambda: RetryConfig(max_retries=2, backoff_factor=1.5, initial_delay=0.5),
        description="Retry configuration for semi-transient errors (empty responses)"
    )


class StageCriticalityConfig(BaseModel):
    """Configuration for stage criticality classification."""
    critical: List[str] = Field(
        default_factory=lambda: ["Generator", "GeneratorContinuation", "SolutionAnalyzer"],
        description="Critical stages that workflow cannot proceed without"
    )
    important: List[str] = Field(
        default_factory=lambda: ["Verifier", "ProgressAssessor", "HintGenerator"],
        description="Important stages that can use fallbacks if they fail"
    )
    optional: List[str] = Field(
        default_factory=lambda: ["Translator", "CoTShortener", "SolutionDetector"],
        description="Optional stages that can be skipped if they fail"
    )


class ErrorHandlingFallbacksConfig(BaseModel):
    """Configuration for fallback values when stages fail."""
    enabled: bool = Field(default=True, description="Whether fallbacks are enabled")
    verifier_fallback_correctness: bool = Field(
        default=False,
        description="Fallback correctness value when Verifier fails (assume incorrect)"
    )
    progress_assessor_fallback_pointer: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fallback progress pointer when ProgressAssessor fails"
    )
    hint_generator_fallback_hint: str = Field(
        default="Continue with the correct approach based on the reference solution.",
        description="Fallback hint text when HintGenerator fails"
    )


class ErrorHandlingConfig(BaseModel):
    """Configuration for workflow error handling."""
    retry_config: ErrorHandlingRetryConfig = Field(
        default_factory=ErrorHandlingRetryConfig,
        description="Retry configuration for different error types"
    )
    stage_criticality: StageCriticalityConfig = Field(
        default_factory=StageCriticalityConfig,
        description="Stage criticality classification"
    )
    fallbacks: ErrorHandlingFallbacksConfig = Field(
        default_factory=ErrorHandlingFallbacksConfig,
        description="Fallback configuration for important stages"
    )


class LoopDetectionConfig(BaseModel):
    """Configuration for detecting and handling infinite loops in generation."""
    enabled: bool = Field(default=False, description="Enable loop detection")
    min_length: int = Field(default=200, description="Minimum length of repetition to detect")


class WorkflowConfig(BaseModel):
    name: str
    # Parameters for the workflow class can be defined here
    max_runs: int = 10
    max_verification_turns: int = 30
    success_threshold: int = 5
    failure_threshold: int = 10
    limit: Optional[int] = None  # Limit the number of documents to process
    binding: WorkflowBindingConfig
    logging: Optional[StageLoggingConfig] = Field(
        default=None,
        description="Stage logging configuration. If not specified, all stages are logged."
    )
    error_handling: Optional[ErrorHandlingConfig] = Field(
        default=None,
        description="Error handling configuration. If not specified, uses defaults."
    )
    loop_detection: Optional[LoopDetectionConfig] = Field(
        default=None,
        description="Configuration for detecting and handling infinite loops in generation"
    )

    model_config = ConfigDict(extra="allow")


class AppConfig(BaseModel):
    max_concurrent_tasks: int


class LoggingConfig(BaseModel):
    level: str


class MainConfig(BaseModel):
    db: DbConfig
    workflow: WorkflowConfig
    stages: List[StageConfig]
    app: AppConfig
    llm: LLMConfig
    logging: LoggingConfig

    # Hydra instantiation config
    repository: Any


