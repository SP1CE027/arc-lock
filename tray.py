import pystray
from pystray import MenuItem as Item
from PIL import Image, ImageDraw

from logs import show_logs


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
        self.status = "enabled"

        self.icon = pystray.Icon(
            "ArcLock",
            self.create_icon("enabled"),
            "ArcLock",
            menu=pystray.Menu(

                Item(
                    self.status_text,
                    self.toggle_pause,
                ),

                pystray.Menu.SEPARATOR,

                Item(
                    "View Logs",
                    show_logs,
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


    # ========================================================
    # ICON
    # ========================================================

    @staticmethod
    def create_icon(status):

        image = Image.new(
            "RGB",
            (64, 64),
            (0, 0, 0),
        )

        draw = ImageDraw.Draw(image)

        if status == "enabled":

            fill = (0, 200, 0)

        elif status == "paused":

            fill = (255, 200, 0)

        elif status == "error":

            fill = (220, 0, 0)

        else:

            fill = (255, 255, 255)

        draw.ellipse(
            (8, 8, 56, 56),
            fill=fill,
        )

        # Small white center gives the icon a more
        # recognizable ArcLock appearance.

        draw.ellipse(
            (22, 22, 42, 42),
            fill=(255, 255, 255),
        )

        return image


    # ========================================================
    # STATUS
    # ========================================================

    def set_status(self, status):

        if status not in (
            "enabled",
            "paused",
            "error",
        ):

            return

        self.status = status

        self.icon.icon = self.create_icon(
            status
        )

        self.icon.update_menu()


    def status_text(self, item):

        if self.paused:

            return "Paused"

        if self.status == "error":

            return "Error"

        return "Enabled"


    # ========================================================
    # PAUSE / RESUME
    # ========================================================

    def toggle_pause(
        self,
        icon,
        item,
    ):

        if self.paused:

            self.paused = False

            self.resume_callback(
                update_tray=False
            )

            self.set_status(
                "enabled"
            )

        else:

            self.paused = True

            self.pause_callback(
                update_tray=False
            )

            self.set_status(
                "paused"
            )


    # ========================================================
    # TIMED PAUSES
    # ========================================================

    def pause_1h(
        self,
        icon,
        item,
    ):

        self.paused = True

        self.pause_for_callback(
            3600
        )

        self.set_status(
            "paused"
        )


    def pause_2h(
        self,
        icon,
        item,
    ):

        self.paused = True

        self.pause_for_callback(
            7200
        )

        self.set_status(
            "paused"
        )


    def pause_3h(
        self,
        icon,
        item,
    ):

        self.paused = True

        self.pause_for_callback(
            10800
        )

        self.set_status(
            "paused"
        )


    def pause_until_restart(
        self,
        icon,
        item,
    ):

        self.paused = True

        self.pause_for_callback(
            None
        )

        self.set_status(
            "paused"
        )


    # ========================================================
    # RESUME FROM MAIN
    # ========================================================

    def set_enabled(self):

        self.paused = False

        self.set_status(
            "enabled"
        )


    # ========================================================
    # ERROR
    # ========================================================

    def set_error(self):

        self.set_status(
            "error"
        )


    # ========================================================
    # QUIT
    # ========================================================

    def quit_app(
        self,
        icon,
        item,
    ):

        self.quit_callback()

        icon.stop()


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        self.icon.run()