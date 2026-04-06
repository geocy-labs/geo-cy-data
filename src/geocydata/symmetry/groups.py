"""Discrete symmetry groups for supported benchmark families."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product


@dataclass(frozen=True)
class SymmetryAction:
    """A signed coordinate permutation on homogeneous coordinates."""

    action_id: str
    permutation: tuple[int, int, int, int]
    signs: tuple[int, int, int, int]


def cefalu_symmetry_actions() -> list[SymmetryAction]:
    """Return the default coordinate-permutation and sign-flip actions for the Cefalu family."""

    actions: list[SymmetryAction] = []
    for permutation in permutations(range(4)):
        for signs in product((-1, 1), repeat=4):
            sign_label = "".join("p" if sign > 0 else "m" for sign in signs)
            perm_label = "".join(str(index) for index in permutation)
            actions.append(
                SymmetryAction(
                    action_id=f"perm_{perm_label}__sign_{sign_label}",
                    permutation=tuple(permutation),
                    signs=tuple(signs),
                )
            )
    return actions

