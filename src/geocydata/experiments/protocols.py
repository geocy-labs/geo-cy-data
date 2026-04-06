"""Named benchmark protocol presets for GeoCYData experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolPreset:
    """Named sweep preset that freezes a benchmark protocol."""

    name: str
    description: str
    target_name: str
    seeds: tuple[int, ...]
    n_samples: int
    include: tuple[str, ...]
    test_size: float = 0.2
    split_strategy: str = "deterministic_random_train_validation_split"

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the preset."""

        return {
            "name": self.name,
            "description": self.description,
            "target_name": self.target_name,
            "seeds": list(self.seeds),
            "n_samples": self.n_samples,
            "include": list(self.include),
            "test_size": self.test_size,
            "split_strategy": self.split_strategy,
        }


PROTOCOL_PRESETS: dict[str, ProtocolPreset] = {
    "paper_v1_fast": ProtocolPreset(
        name="paper_v1_fast",
        description="Single-seed paper-style smoke benchmark for quick checks.",
        target_name="hypersurface_fs_scalar",
        seeds=(7,),
        n_samples=120,
        include=("fermat_quartic", "cefalu_lambda_0_75", "cefalu_lambda_1_0"),
    ),
    "paper_v1_default": ProtocolPreset(
        name="paper_v1_default",
        description="Default paper-style benchmark protocol with the core multiseed matrix.",
        target_name="hypersurface_fs_scalar",
        seeds=(7, 11, 19),
        n_samples=200,
        include=("fermat_quartic", "cefalu_lambda_0_75", "cefalu_lambda_1_0"),
    ),
    "paper_v1_multiseed": ProtocolPreset(
        name="paper_v1_multiseed",
        description="Explicit multiseed paper benchmark protocol for the core geometry matrix.",
        target_name="hypersurface_fs_scalar",
        seeds=(7, 11, 19),
        n_samples=200,
        include=("fermat_quartic", "cefalu_lambda_0_75", "cefalu_lambda_1_0"),
    ),
}


HARD_EVALUATION_SLICES: dict[str, dict[str, object]] = {
    "cefalu_hard_v1": {
        "name": "cefalu_hard_v1",
        "description": "Harder Cefalu neighborhood around lambda=1.0 for robustness-focused evaluation.",
        "include": ["cefalu_lambda_0_99", "cefalu_lambda_1_00", "cefalu_lambda_1_01"],
    }
}


def list_protocol_presets() -> list[str]:
    """List the registered protocol preset names."""

    return sorted(PROTOCOL_PRESETS)


def list_hard_evaluation_slices() -> list[str]:
    """List the registered harder evaluation slices."""

    return sorted(HARD_EVALUATION_SLICES)


def resolve_protocol_preset(name: str) -> ProtocolPreset:
    """Resolve one named protocol preset."""

    try:
        return PROTOCOL_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(list_protocol_presets())
        raise ValueError(f"Unknown protocol preset '{name}'. Available presets: {available}.") from exc


def resolve_hard_evaluation_slice(name: str) -> dict[str, object]:
    """Resolve one named harder evaluation slice."""

    try:
        return HARD_EVALUATION_SLICES[name]
    except KeyError as exc:
        available = ", ".join(list_hard_evaluation_slices())
        raise ValueError(f"Unknown hard evaluation slice '{name}'. Available slices: {available}.") from exc
