import wireworld as ww
import pygame

FRAMES_PER_SECOND = 4

DATA_FILE_1 = 'states/initial_state_1.csv'
DATA_FILE_2 = 'states/squares.csv'
DATA_FILE_3 = 'states/xor-generators.csv'
DATA_FILE_4 = 'states/diode.csv'
INTERRUPTED_FILE = 'states/interrupted.csv'


class WireworldSimulation:
    def __init__(self, eng):
        self.eng = eng
        initial_state = self.eng.load_state(INTERRUPTED_FILE)

        width = initial_state.shape[0]
        height = initial_state.shape[1]

        pygame.init()
        self.fps = FRAMES_PER_SECOND
        self.scale_factor = ww.DEFAULT_SCALE_FACTOR
        self.state = initial_state
        self.screen = pygame.display.set_mode((width * self.scale_factor, height * self.scale_factor))
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
        display_array = self.eng.get_display_array_from_state(self.state, self.scale_factor)
        self.screen.fill(ww.BLACK)
        pygame.surfarray.blit_array(self.screen, display_array)

    def _render(self):
        # Now that everything is drawn render it by flipping the buffers
        pygame.display.flip()
    