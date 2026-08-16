import pystray
from pystray import MenuItem as Item
from PIL import Image, ImageDraw


class TrayApp:

    def __init__(
        self,
        pause_callback,
        resume_callback,
        pause_for_callback,
        quit_callback,
    ):
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self.pause_for_callback = pause_for_callback
        self.quit_callback = quit_callback

        self.paused = False

        self.icon = pystray.Icon(
            "ArcLock",
            self.create_icon(),
            "ArcLock",
            menu=pystray.Menu(
                Item(
                    self.status_text,
                    self.toggle_pause,
                ),

                pystray.Menu.SEPARATOR,

                Item(
                    "Pause 1 Hour",
                    self.pause_1h,
                ),

                Item(
                    "Pause 2 Hours",
                    self.pause_2h,
                ),

                Item(
                    "Pause 3 Hours",
                    self.pause_3h,
                ),

                Item(
                    "Pause Until Restart",
                    self.pause_until_restart,
                ),

                pystray.Menu.SEPARATOR,

                Item(
                    "Kill ArcLock",
                    self.quit_app,
                ),
            ),
        )

    # ---------------------------------------------------------
    # ICON
    # ---------------------------------------------------------

    @staticmethod
    def create_icon():

        image = Image.new(
            "RGB",
            (64, 64),
            (0, 0, 0),
        )

        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (8, 8, 56, 56),
            fill=(255, 255, 255),
        )

        return image

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def status_text(self, item):

        if self.paused:
            return "Paused"

        return "Enabled"

    # ---------------------------------------------------------
    # PAUSE / RESUME
    # ---------------------------------------------------------

    def toggle_pause(self, icon, item):

        if self.paused:

            self.paused = False
            self.resume_callback()

        else:

            self.paused = True
            self.pause_callback()

        icon.update_menu()

    # ---------------------------------------------------------
    # TIMED PAUSES
    # ---------------------------------------------------------

    def pause_1h(self, icon, item):

        self.paused = True
        self.pause_for_callback(3600)

        icon.update_menu()

    def pause_2h(self, icon, item):

        self.paused = True
        self.pause_for_callback(7200)

        icon.update_menu()

    def pause_3h(self, icon, item):

        self.paused = True
        self.pause_for_callback(10800)

        icon.update_menu()

    def pause_until_restart(self, icon, item):

        self.paused = True
        self.pause_for_callback(None)

        icon.update_menu()

    # ---------------------------------------------------------
    # QUIT
    # ---------------------------------------------------------

    def quit_app(self, icon, item):

        self.quit_callback()

        icon.stop()

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    def run(self):

        self.icon.run()