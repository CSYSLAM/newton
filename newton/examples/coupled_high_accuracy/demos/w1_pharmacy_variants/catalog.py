"""Four generic medicine-carton constructions, with estimated effective material parameters."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Carton:
    key: str
    name: str
    product: str
    size: tuple
    cells: tuple
    thickness: float
    young: float
    yield_strain: float
    peak_force: float
    construction: str
    color: tuple
    behavior: str

    def report(self):
        return asdict(self)


CARTONS = (
    Carton(
        "tablets",
        "小号片剂盒",
        "片剂",
        (0.072, 0.038, 0.102),
        (10, 6, 14),
        0.00050,
        10e6,
        0.060,
        2.5,
        "reinforced_ends",
        (27, 94, 156),
        "轻压回弹",
    ),
    Carton(
        "capsules",
        "细高胶囊盒",
        "胶囊",
        (0.062, 0.040, 0.145),
        (9, 6, 20),
        0.00045,
        8e6,
        0.050,
        9.0,
        "side_overlap",
        (15, 109, 98),
        "中度压凹",
    ),
    Carton(
        "granules",
        "宽面颗粒盒",
        "颗粒",
        (0.102, 0.046, 0.115),
        (14, 7, 16),
        0.00038,
        6.5e6,
        0.040,
        8.5,
        "scored_corners",
        (176, 109, 24),
        "局部折痕",
    ),
    Carton(
        "patches",
        "长扁贴剂盒",
        "贴剂",
        (0.112, 0.042, 0.085),
        (15, 6, 12),
        0.00033,
        5.5e6,
        0.035,
        23.0,
        "thin_wall",
        (135, 56, 99),
        "明显压瘪",
    ),
)
PAIRS = ((0, 1), (2, 3))
DURATION = 11.5
TABLE_Z = 0.86


def centers(cases):
    return [[0.47, 0.28 if i == 0 else -0.28, TABLE_Z + 0.0005 + c.size[2] / 2] for i, c in enumerate(cases)]
