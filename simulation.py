import pygame
import sys
import wireworld.gui as gui
from pgu import timer

FRAMES_PER_SECOND = 4
SCALE_FACTOR = 30

EXAMPLE_INPUT_FILE = 'states/example.csv'
SQUARES_INPUT_FILE = 'states/squares.csv'
XOR_GENERATORS_INPUT_FILE = 'states/xor-generators.csv'
DIODE_INPUT_FILE = 'states/diode.csv'
INTERRUPTED_FILE = 'states/interrupted.csv'

FILE_TO_LOAD = XOR_GENERATORS_INPUT_FILE


class WireworldSimulation:
    """
    A wireworld simulation using pygame

    """

    def __init__(self, eng):
        self.eng = eng
        initial_state = self.eng.load_state(FILE_TO_LOAD)

        width = initial_state.shape[0]
        height = initial_state.shape[1]

        pygame.init()
        self.fps = FRAMES_PER_SECOND
        self.state = initial_state
        # self.screen = pygame.display.set_mode((width * SCALE_FACTOR, height * SCALE_FACTOR))
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = timer.Clock() #pygame.time.Clock()

        self.updates = []
        self.gui = gui.MainGui(self.screen)
        self.gui.engine = self

    def start(self):
        while True:
            self._loop()

    def stop(self):
        """
        Stops the simulation, but saves off the state to file first

        """
        self.eng.save_state(self.state, INTERRUPTED_FILE)
        pygame.quit()
        sys.exit(0)

    def pause(self):
        self.clock.pause()

    def resume(self):
        self.clock.resume()

    def _loop(self):
        self._poll()
        self._update()
        #self._draw()
        self._render()
        self.clock.tick(self.fps)

    def _poll(self):
        ev = pygame.event.poll()
        if ev.type == pygame.QUIT:
            self.stop()
        else:
            # Pass the event off to pgu
            self.gui.event(ev)

    def _update(self):
        self.state = self.eng.get_next_state(self.state)

    def _draw(self, dest, rect):
        display_array = self.eng.get_display_array_from_state(self.state, SCALE_FACTOR)
        display_surf = pygame.surfarray.make_surface(display_array)
        dest.blit(display_surf, rect)
        self.updates.append(rect)


    def _render(self):
        # Give pgu a chance to update the display

        rect = self.gui.get_render_area()
        self._draw(self.screen, rect)

        lst = self.gui.update()
        if lst:
            self.updates += lst
        pygame.display.update(self.updates)
        pygame.time.wait(10)

        # Now that everything is drawn render it by flipping the buffers
        # pygame.display.flip()
