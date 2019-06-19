from enum import Enum, IntEnum


class Connect4Stats():
    height = 6
    width = 7
    area = 42


class Side(IntEnum):
    o = 0
    x = 1

    @classmethod
    def as_str(cls, side):
        return 'o' if side == Side.o else 'x'


class Result(Enum):
    o_win = 1.0
    x_win = 0.0
    draw = 0.5


def same_side(result: Result, side: Side):
    if result == Result.o_win and side == Side.o:
        return True
    elif result == Result.x_win and side == Side.x:
        return True
    return False


def value_to_side(value: float, side: Side) -> float:
    return value if side == Side.o else (1.0 - value)
