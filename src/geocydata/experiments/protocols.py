"""Named benchmark protocol presets for GeoCYData experiments."""

from __future__ import annotations

from dataclasses import dataclass

from geocydata.registry.cases import NEAR_SINGULAR_SLICES, PAPER1_CORE_CASE_IDS, PAPER2_HARD_REGIME_CASE_IDS


@dataclass(frozen=True)
class ProtocolPreset:
    """Named sweep preset that freezes a benchmark protocol."""

    name: str
    description: str
    target_name: str
    seeds: tuple[int, ...]
    n_samples: int
    include: tuple[str, ...]
    benchmark_version: str = "phase9"
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
            "benchmark_version": self.benchmark_version,
            "test_size": self.test_size,
            "split_strategy": self.split_strategy,
        }


PROTOCOL_PRESETS: dict[str, ProtocolPreset] = {
    "paper_v1_fast": ProtocolPreset(
        name="paper_v1_fast",
        description="Single-seed paper-style smoke benchmark for quick checks.",
        target_name="hypersurface_fs_scalar",
        seeds=(7,),
        n_samples=64,
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
    "globalcy_paper1_core": ProtocolPreset(
        name="globalcy_paper1_core",
        description="Paper 1 GlobalCY core geometry matrix across the first fixed Cefalu cases.",
        target_name="hypersurface_fs_scalar",
        seeds=(7,),
        n_samples=200,
        include=PAPER1_CORE_CASE_IDS,
    ),
    "globalcy_paper1_near_0_75": ProtocolPreset(
        name="globalcy_paper1_near_0_75",
        description="Paper 1 near-singular Cefalu slice around lambda=0.75.",
        target_name="hypersurface_fs_scalar",
        seeds=(7,),
        n_samples=200,
        include=NEAR_SINGULAR_SLICES["cefalu_near_lambda_0_75"],
    ),
    "globalcy_paper1_near_1_0": ProtocolPreset(
        name="globalcy_paper1_near_1_0",
        description="Paper 1 near-singular Cefalu slice around lambda=1.0.",
        target_name="hypersurface_fs_scalar",
        seeds=(7,),
        n_samples=200,
        include=NEAR_SINGULAR_SLICES["cefalu_near_lambda_1_0"],
    ),
    "cefalu_hard_regime_sweep_v1": ProtocolPreset(
        name="cefalu_hard_regime_sweep_v1",
        description="Paper II hard-regime Cefalu benchmark preset for GlobalCY II.",
        target_name="hypersurface_fs_scalar",
        seeds=(7, 11, 19),
        n_samples=200,
        include=PAPER2_HARD_REGIME_CASE_IDS,
        benchmark_version="paper2_hard_regime_v1",
    ),
}


HARD_EVALUATION_SLICES: dict[str, dict[str, object]] = {
    "cefalu_hard_v1": {
        "name": "cefalu_hard_v1",
        "description": "Harder Cefalu neighborhood around lambda=1.0 for robustness-focused evaluation.",
        "include": ["cefalu_lambda_0_99", "cefalu_lambda_1_0", "cefalu_lambda_1_01"],
    },
    "globalcy_near_lambda_0_75": {
        "name": "globalcy_near_lambda_0_75",
        "description": "Fixed near-singular slice around lambda=0.75 for GlobalCY Paper 1.",
        "include": list(NEAR_SINGULAR_SLICES["cefalu_near_lambda_0_75"]),
    },
    "globalcy_near_lambda_1_0": {
        "name": "globalcy_near_lambda_1_0",
        "description": "Fixed near-singular slice around lambda=1.0 for GlobalCY Paper 1.",
        "include": list(NEAR_SINGULAR_SLICES["cefalu_near_lambda_1_0"]),
    },
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
