BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

EMPTY = 0
HEAD = 1
TAIL = 2
CONDUCTOR = 3

VALUE_TO_RGB = {
    0: BLACK,
    1: BLUE,
    2: RED,
    3: YELLOW
}

# TODO:
# 2) Errors:
# a. If trying to load from a non-existent file
# b. If trying to load from an incompatible file (e.g. not a CSV)
# c. If trying to save to a file location that produces an error (already exists?, in a folder that doesn't exist?)
# d. Array out of bounds errors e.g. if coord outside state array in get next value function
