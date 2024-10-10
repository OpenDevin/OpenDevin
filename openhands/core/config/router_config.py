from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ModelConfig:
    model_name: str
    litellm_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Remove None values from litellm_params
        self.litellm_params = {
            k: v for k, v in self.litellm_params.items() if v is not None
        }


@dataclass
class RouterConfig:
    default_model: str | None = field(default=None)
    model_list: List[ModelConfig] = field(default_factory=list)
    routing_strategy: str = 'simple-shuffle'
    cooldown_time: Optional[float] = 1
    num_retries: Optional[int] = 8
    retry_after: int = 15
    timeout: Optional[float] = 120
    allowed_fails: Optional[int] = None
    default_fallbacks: list[str] | None = None
    cache_responses: bool = False
    cache_kwargs: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(
        default_factory=lambda: {
            'exceptions_to_retry': ['RateLimitError'],
            'max_retries': 8,
            'retry_after': 15,
            'retry_after_multiplier': 2,
        }
    )
