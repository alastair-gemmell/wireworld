import numpy as np
import pygame

BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

EMPTY = 0
HEAD = 1
TAIL = 2
CONDUCTOR = 3

SCALE_FACTOR = 30
FRAMES_PER_SECOND = 4

VALUE_TO_RGB = {
    0: BLACK,
    1: BLUE,
    2: RED,
    3: YELLOW
}

DATA_FILE_1 = 'states/initial_state_1.csv'
DATA_FILE_2 = 'states/squares.csv'
DATA_FILE_3 = 'states/xor-generators.csv'
DATA_FILE_4 = 'states/diode.csv'
INTERRUPTED_FILE = 'states/interrupted.csv'


# TODO:
# 1) Doc strings / Comments
#
# 2) Errors:
# a. If trying to load from a non-existent file
# b. If trying to load from an incompatible file (e.g. not a CSV)
# c. If trying to save to a file location that produces an error (already exists?, in a folder that doesn't exist?)
# d. Array out of bounds errors e.g. if coord outside state array in get next value function
#
# 3) Unit Tests:
# a. load function?
# b. Save function?
# c. get neighbours function?
# d. get next value function?
# e. get next state function?
# f. get display array function?

def main():
    sim = WireworldSimulation()
    sim.start()


class WireworldSimulation:
    def __init__(self):
        initial_state = load_state(INTERRUPTED_FILE)

        width = initial_state.shape[0]
        height = initial_state.shape[1]

        pygame.init()
        self.fps = FRAMES_PER_SECOND
        self.state = initial_state
        self.screen = pygame.display.set_mode((width * SCALE_FACTOR, height * SCALE_FACTOR))
        self.clock = pygame.time.Clock()

    def start(self):
        while True:
            self._loop()

    def stop(self):
        save_state(self.state, INTERRUPTED_FILE)
        pygame.quit()
        exit(0)

    def _loop(self):
        self._poll()
        self._update()
        self._draw()
        self._render()
        self.clock.tick(self.fps)

    def _poll(self):
        ev = pygame.event.poll()
        if ev.type == pygame.QUIT:
            self.stop()

    def _update(self):
        self.state = get_next_state(self.state)

    def _draw(self):
        display_array = get_display_array_from_state(self.state)
        self.screen.fill(BLACK)
        pygame.surfarray.blit_array(self.screen, display_array)

    def _render(self):
        # Now that everything is drawn render it by flipping the buffers
        pygame.display.flip()


def load_state(file):
    return np.rot90(np.genfromtxt(file, delimiter=','), 3)


def save_state(state, file):
    np.savetxt(file, np.rot90(state), delimiter=',', fmt='%i')


def get_display_array_from_state(state):
    # Create an RGB array by adding extra dimension of size 3 and filling with RGB values
    rgb_array = np.repeat(state[:, :, np.newaxis], 3, axis=2)
    for ix, iy in np.ndindex(state.shape):
        rgb_array[ix, iy] = VALUE_TO_RGB[state[ix, iy]]

    # scale out the array so we don't just have one pixel per cell using
    # https://stackoverflow.com/questions/7525214/how-to-scale-a-numpy-array
    return np.kron(rgb_array, np.ones((SCALE_FACTOR, SCALE_FACTOR, 1)))


def get_next_state(current_state):
    next_state = np.zeros_like(current_state)
    for coord, val in np.ndenumerate(next_state):
        next_state[coord] = get_next_value(coord, current_state)
    return next_state


def get_next_value(coord, state_array):
    next_val = EMPTY
    if state_array[coord] == HEAD:
        next_val = TAIL
    elif state_array[coord] == TAIL:
        next_val = CONDUCTOR
    elif state_array[coord] == CONDUCTOR:
        if 1 <= len(get_neighbours(coord, state_array, HEAD)) <= 2:
            next_val = HEAD
        else:
            next_val = CONDUCTOR

    return next_val


def get_neighbours(coord, array, condition_value):
    neighbours = []
    rows = len(array)
    cols = len(array[0]) if rows else 0
    for i in range(max(0, coord[0] - 1), min(rows, coord[0] + 2)):
        for j in range(max(0, coord[1] - 1), min(cols, coord[1] + 2)):
            if (i, j) != coord and array[i][j] == condition_value:
                neighbours.append((i, j))
    return neighbours


if __name__ == "__main__":
    main()