import pygame

FRAMES_PER_SECOND = 4
SCALE_FACTOR = 30

EXAMPLE_INPUT_FILE = 'states/example.csv'
SQUARES_INPUT_FILE = 'states/squares.csv'
XOR_GENERATORS_INPUT_FILE = 'states/xor-generators.csv'
DIODE_INPUT_FILE = 'states/diode.csv'
INTERRUPTED_FILE = 'states/interrupted.csv'

FILE_TO_LOAD = INTERRUPTED_FILE

class WireworldSimulation:
    def __init__(self, eng):
        self.eng = eng
        initial_state = self.eng.load_state(FILE_TO_LOAD)

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
        self.eng.save_state(self.state, INTERRUPTED_FILE)
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
        self.state = self.eng.get_next_state(self.state)

    def _draw(self):
        display_array = self.eng.get_display_array_from_state(self.state, SCALE_FACTOR)
        pygame.surfarray.blit_array(self.screen, display_array)

    def _render(self):
        # Now that everything is drawn render it by flipping the buffers
        pygame.display.flip()
    