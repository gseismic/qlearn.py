from enum import IntEnum


class ExitCode(IntEnum):
    UNDEFINED = -1
    SUCC = 0
    EXIT_SUCC = 0
    EXIT_REACH_MAX_ITER = 1
