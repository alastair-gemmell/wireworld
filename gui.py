import pygame
from pgu import gui


class DrawingArea(gui.Widget):
    def __init__(self, width, height):
        gui.Widget.__init__(self, width=width, height=height)
        self.imageBuffer = pygame.Surface((width, height))

    def paint(self, surf):
        # Paint whatever has been captured in the buffer
        surf.blit(self.imageBuffer, (0, 0))

    # Call this function to take a snapshot of whatever has been rendered
    # onto the display over this widget.
    def save_background(self):
        disp = pygame.display.get_surface()
        self.imageBuffer.blit(disp, self.get_abs_rect())


class TestDialog(gui.Dialog):
    def __init__(self):
        title = gui.Label("Some Dialog Box")
        label = gui.Label("Close this window to resume.")
        gui.Dialog.__init__(self, title, label)


class MainGui(gui.Desktop):
    gameAreaHeight = 500
    gameArea = None
    menuArea = None
    # The game engine
    engine = None

    def __init__(self, disp):
        gui.Desktop.__init__(self)

        self.toggle_pause_btn = gui.Switch("Pause", height=50)

        # Setup the 'game' area where the action takes place
        self.gameArea = DrawingArea(disp.get_width(),
                                    self.gameAreaHeight)
        # Setup the gui area
        self.menuArea = gui.Container(height=disp.get_height() - self.gameAreaHeight)

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
            print("modal dialog clicked!")
            self.engine.toggle_pause()
            # if self.toggle_pause_btn.value.value == "Pause":
            #     self.toggle_pause_btn.value = "Resume"
            # else:
            #     self.toggle_pause_btn.value = "Pause"

        self.toggle_pause_btn.connect(gui.CLICK, dialog_cb)
        tbl.td(self.toggle_pause_btn)

        # Add a slider for adjusting the game clock speed
        tbl2 = gui.Table()

        timeLabel = gui.Label("Clock speed")

        tbl2.tr()
        tbl2.td(timeLabel)

        slider = gui.HSlider(value=4,min=1,max=30,size=20,height=16,width=120)

        def update_speed():
            # if slider.value == 0:
            #     self.engine.pause
            # else:
            #     self.engine.resume
                self.engine.fps = slider.value

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
