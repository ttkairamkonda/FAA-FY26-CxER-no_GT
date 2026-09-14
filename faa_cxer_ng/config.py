"""Central configuration — every script's default file locations, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class Config:
    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR

    conversations_path: Path = field(default=None)  # type: ignore[assignment]
    results_path: Path = field(default=None)  # type: ignore[assignment]

    judge_model: str = DEFAULT_JUDGE_MODEL

    def __post_init__(self):
        if self.conversations_path is None:
            self.conversations_path = self.data_dir / "synthetic_conversations.json"
        if self.results_path is None:
            self.results_path = self.data_dir / "analysis_results.json"


def default_config() -> Config:
    return Config()
