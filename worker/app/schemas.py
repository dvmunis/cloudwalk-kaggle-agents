"""Request/response contracts. These are what the n8n agents produce and consume."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    drop: list[str] = Field(default_factory=list, description="Columns to remove")
    frequency_encode: list[str] = Field(default_factory=list, description="Add <col>_freq with value counts")
    as_category: list[str] = Field(default_factory=list, description="Treat these numeric columns as categorical")
    add_null_count: bool = Field(False, description="Add a per-row count of missing values")
    digit_features: list[str] = Field(
        default_factory=list,
        description="Numeric columns that get <col>_d0/_d1/_frac/_round100: digits and roundness of the value",
    )
    target_encode: list[str] = Field(
        default_factory=list,
        description="Add <col>_te: smoothed target mean per value, computed out-of-fold inside the CV",
    )


class TrainConfig(BaseModel):
    metric: str = Field(description="Competition metric, e.g. roc_auc, rmse, map@3")
    model: Literal["lightgbm", "xgboost", "blend"] = "lightgbm"
    params: dict[str, Any] = Field(default_factory=dict, description="Overrides for the model's default params")
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    target_transform: Literal["none", "log1p"] = "none"
    n_folds: int = Field(5, ge=2, le=10)
    sample_frac: float = Field(1.0, gt=0, le=1, description="Train on a fraction of rows for fast iterations")
    seed: int = 42
    external_data: list[str] = Field(
        default_factory=list,
        description="Accepted external datasets whose rows are added to the training folds (never to validation)",
    )
    blend_runs: list[str] = Field(
        default_factory=list,
        description="model 'blend': earlier runs whose out-of-fold and test predictions are averaged",
    )
    note: str = Field("", description="Why this experiment is being run (kept in the experiment log)")


class AttemptCreate(BaseModel):
    competition: str
    goal: str = ""
    execution_id: str | None = Field(None, description="n8n execution that runs this attempt")


class ExecutionFailure(BaseModel):
    node: str | None = None
    error: str


class EventCreate(BaseModel):
    agent: str = Field(description="Who produced the event, e.g. planner, evaluator, orchestrator")
    kind: Literal["decision", "tool_call", "tool_result", "message", "error"] = "decision"
    content: Any
    reasoning: str | None = None
    context: Any = Field(None, description="What the agent saw when it decided (its full input)")
    meta: dict[str, Any] | None = Field(None, description="LLM call details: model, attempts, elapsed_ms, token_usage")


class AttemptFinish(BaseModel):
    status: Literal["completed", "failed"] = "completed"
    summary: dict[str, Any] = Field(default_factory=dict)


class DiscoverRequest(BaseModel):
    queries: list[str] = Field(default_factory=list, description="Kaggle dataset searches, e.g. from the planner")


class SubmitRequest(BaseModel):
    run_id: str
    message: str = ""
    wait_for_score: bool = True
