"""Ground-truth color mixing rule (8-color, 3-bit RGB)."""

from typing import Dict, FrozenSet

BLACK = (0, 0, 0)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
YELLOW = (1, 1, 0)
MAGENTA = (1, 0, 1)
CYAN = (0, 1, 1)
WHITE = (1, 1, 1)

COLORS = (BLACK, RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN, WHITE)
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}
NUM_COLORS = 8

COLOR_NAMES = {
    BLACK: "K", RED: "R", GREEN: "G", BLUE: "B",
    YELLOW: "Y", MAGENTA: "M", CYAN: "C", WHITE: "W",
}


def edge_key(c_A, c_B):
    return frozenset({COLOR_TO_IDX[c_A], COLOR_TO_IDX[c_B]})


def mix(c_A, c_B):
    s = (c_A[0] + c_B[0], c_A[1] + c_B[1], c_A[2] + c_B[2])
    if s[0] > 1 or s[1] > 1 or s[2] > 1:
        return BLACK
    return s


def _build_truth():
    truth = {}
    for i, c_A in enumerate(COLORS):
        for c_B in COLORS[i:]:
            truth[edge_key(c_A, c_B)] = COLOR_TO_IDX[mix(c_A, c_B)]
    return truth


TRUE_MIXING_RULE = _build_truth()
ALL_EDGES = list(TRUE_MIXING_RULE.keys())


def edge_label(edge):
    idxs = sorted(edge)
    if len(idxs) == 1:
        idxs = [idxs[0], idxs[0]]
    return f"{COLOR_NAMES[COLORS[idxs[0]]]}+{COLOR_NAMES[COLORS[idxs[1]]]}"


EDGE_LABELS = {e: edge_label(e) for e in ALL_EDGES}

assert len(TRUE_MIXING_RULE) == 36
