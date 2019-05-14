#!/usr/bin/env python

# self is not needed if you have PGU installed
import sys
sys.path.insert(0, "..")

import math
import pygame
from pgu import gui, timer

import wireworld.engine as engine


SCALE_FACTOR = 30

EXAMPLE_INPUT_FILE = 'states/example.csv'
SQUARES_INPUT_FILE = 'states/squares.csv'
XOR_GENERATORS_INPUT_FILE = 'states/xor-generators.csv'
DIODE_INPUT_FILE = 'states/diode.csv'
INTERRUPTED_FILE = 'states/interrupted.csv'

FILE_TO_LOAD = XOR_GENERATORS_INPUT_FILE


class DrawingArea(gui.Widget):
    def __init__(self, width, height):
        gui.Widget.__init__(self, width=width, height=height)
        self.imageBuffer = pygame.Surface((width, height))

    def paint(self, surf):
        # Paint whatever has been captured in the buffer
        surf.blit(self.imageBuffer, (0, 0))

    # Call self function to take a snapshot of whatever has been rendered
    # onto the display over self widget.
    def save_background(self):
        disp = pygame.display.get_surface()
        self.imageBuffer.blit(disp, self.get_abs_rect())

class TestDialog(gui.Dialog):
    def __init__(self):
        title = gui.Label("Some Dialog Box")
        label = gui.Label("Close self window to resume.")
        gui.Dialog.__init__(self, title, label)

class MainGui(gui.Desktop):
    gameAreaHeight = 500
    gameArea = None
    menuArea = None
    # The game engine
    engine = None

    def __init__(self, disp):
        gui.Desktop.__init__(self)

        # Setup the 'game' area where the action takes place
        self.gameArea = DrawingArea(disp.get_width(),
                                    self.gameAreaHeight)
        # Setup the gui area
        self.menuArea = gui.Container(
            height=disp.get_height()-self.gameAreaHeight)

        tbl = gui.Table(height=disp.get_height())
        tbl.tr()
        tbl.td(self.gameArea)
        tbl.tr()
        tbl.td(self.menuArea)

        self.setup_menu()

        self.init(tbl, disp)

    def setup_menu(self):
        tbl = gui.Table(vpadding=5, hpadding=2)
        tbl.tr()

        dlg = TestDialog()

        def dialog_cb():
            dlg.open()

        btn = gui.Button("Modal dialog", height=50)
        btn.connect(gui.CLICK, dialog_cb)
        tbl.td(btn)

        # Add a button for pausing / resuming the game clock
        def pause_cb():
            if (self.engine.clock.paused):
                self.engine.resume()
            else:
                self.engine.pause()

        btn = gui.Button("Pause/resume clock", height=50)
        btn.connect(gui.CLICK, pause_cb)
        tbl.td(btn)

        # Add a slider for adjusting the game clock speed
        tbl2 = gui.Table()

        timeLabel = gui.Label("Clock speed")

        tbl2.tr()
        tbl2.td(timeLabel)

        slider = gui.HSlider(value=23,min=0,max=100,size=20,height=16,width=120)

        def update_speed():
            self.engine.clock.set_speed(slider.value/10.0)

        slider.connect(gui.CHANGE, update_speed)

        tbl2.tr()
        tbl2.td(slider)

        tbl.td(tbl2)

        self.menuArea.add(tbl, 0, 0)

    def open(self, dlg, pos=None):
        # Gray out the game area before showing the popup
        rect = self.gameArea.get_abs_rect()
        dark = pygame.Surface(rect.size).convert_alpha()
        dark.fill((0,0,0,150))
        pygame.display.get_surface().blit(dark, rect)
        # Save whatever has been rendered to the 'game area' so we can
        # render it as a static image while the dialog is open.
        self.gameArea.save_background()
        # Pause the gameplay while the dialog is visible
        running = not(self.engine.clock.paused)
        self.engine.pause()
        gui.Desktop.open(self, dlg, pos)
        while (dlg.is_open()):
            for ev in pygame.event.get():
                self.event(ev)
            rects = self.update()
            if (rects):
                pygame.display.update(rects)
        if (running):
            # Resume gameplay
            self.engine.resume()

    def get_render_area(self):
        return self.gameArea.get_abs_rect()


class GameEngine(object):
    def __init__(self, disp, eng):

        self.eng = eng
        initial_state = self.eng.load_state(FILE_TO_LOAD)

        self.state = initial_state
        self.disp = disp
        self.square = pygame.Surface((400,400)).convert_alpha()
        self.square.fill((255,0,0))
        self.app = MainGui(self.disp)
        self.app.engine = self

    # Pause the game clock
    def pause(self):
        self.clock.pause()

    # Resume the game clock
    def resume(self):
        self.clock.resume()

    def render(self, dest, rect):

        print("a", rect)
        self.state = self.eng.get_next_state(self.state)

        display_array = self.eng.get_display_array_from_state(self.state, SCALE_FACTOR)
        # display_surf = pygame.surfarray.make_surface(display_array)
        display_surf = pygame.Surface((display_array.shape[0], display_array.shape[1]))
        #dest.blit(display_surf, rect)
        pygame.surfarray.blit_array(display_surf, display_array)
        self.disp.blit(display_surf, (0,0))
        #self.updates.append(rect)

        print("b", rect)

        return rect

    def run(self):
        self.app.update()
        pygame.display.flip()

        self.font = pygame.font.SysFont("", 16)

        self.clock = timer.Clock() #pygame.time.Clock()
        done = False
        while not done:
            # Process events
            for ev in pygame.event.get():
                if (ev.type == pygame.QUIT or
                    ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    done = True
                else:
                    # Pass the event off to pgu
                    self.app.event(ev)
            # Render the game
            rect = self.app.get_render_area()
            updates = []
            self.disp.set_clip(rect)
            lst = self.render(self.disp, rect)
            print("c", lst)
            if lst:
                updates.append(lst)
                print("1", updates)
            self.disp.set_clip()

            # Cap it at 30fps
            self.clock.tick(1)

            # Give pgu a chance to update the display
            lst = self.app.update()
            if lst:
                print("2a", updates)
                updates += lst
                print("2b", updates)

            pygame.display.update(updates)
            pygame.time.wait(10)


###
disp = pygame.display.set_mode((800, 600))
ww_eng = engine.WireworldEngine()
eng = GameEngine(disp, ww_eng)
eng.run()

