"""Configuration for the released inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ModelConfig:
    history_frames: int

    def __post_init__(self) -> None:
        if self.history_frames < 1:
            raise ValueError("history_frames must be at least 1")


@dataclass(frozen=True)
class DataConfig:
    validation_root: Path
    sample_submission: Path


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    source: Path


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(
            **{
                key: _resolve(path.parent, value)
                for key, value in raw["data"].items()
            }
        ),
        source=path,
    )
