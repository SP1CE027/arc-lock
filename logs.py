import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
import threading
import queue


# ============================================================
# STATE
# ============================================================

_command_queue = queue.Queue()
_log_history = []

_started = False
_start_lock = threading.Lock()
_tk_thread = None


# ============================================================
# LOG WINDOW THREAD
# ============================================================

def _tk_worker():

    root = tk.Tk()

    root.title("ArcLock Logs")
    root.geometry("700x400")
    root.minsize(500, 250)

    text = ScrolledText(
        root,
        state="disabled",
        font=("Consolas", 10)
    )

    text.pack(
        fill="both",
        expand=True
    )

    # Hide instead of destroying
    root.protocol(
        "WM_DELETE_WINDOW",
        root.withdraw
    )

    # Start hidden
    root.withdraw()


    # --------------------------------------------------------
    # Process commands from other threads
    # --------------------------------------------------------

    def process_commands():

        try:

            while True:

                command = _command_queue.get_nowait()

                # --------------------------------------------
                # SHOW
                # --------------------------------------------

                if command == "show":

                    root.deiconify()
                    root.lift()
                    root.attributes(
                        "-topmost",
                        True
                    )
                    root.after(
                        100,
                        lambda: root.attributes(
                            "-topmost",
                            False
                        )
                    )
                    root.focus_force()


                # --------------------------------------------
                # HIDE
                # --------------------------------------------

                elif command == "hide":

                    root.withdraw()


                # --------------------------------------------
                # QUIT
                # --------------------------------------------

                elif command == "quit":

                    root.destroy()
                    return


                # --------------------------------------------
                # LOG MESSAGE
                # --------------------------------------------

                elif (
                    isinstance(command, tuple)
                    and command[0] == "log"
                ):

                    line = command[1]

                    text.configure(
                        state="normal"
                    )

                    text.insert(
                        "end",
                        line
                    )

                    text.see("end")

                    text.configure(
                        state="disabled"
                    )

        except queue.Empty:
            pass


        root.after(
            100,
            process_commands
        )


    # --------------------------------------------------------
    # Load logs that happened before the window was created
    # --------------------------------------------------------

    for line in _log_history:

        text.configure(
            state="normal"
        )

        text.insert(
            "end",
            line
        )

        text.configure(
            state="disabled"
        )


    process_commands()

    root.mainloop()


# ============================================================
# START LOG SYSTEM
# ============================================================

def start_log_window():

    global _started
    global _tk_thread

    with _start_lock:

        if _started:
            return

        _started = True

        _tk_thread = threading.Thread(
            target=_tk_worker,
            daemon=True
        )

        _tk_thread.start()


# ============================================================
# SHOW LOGS
# ============================================================

def show_logs(icon=None, item=None):

    if not _started:
        start_log_window()

    _command_queue.put("show")


# ============================================================
# LOG
# ============================================================

def log(message):

    timestamp = datetime.now().strftime(
        "%H:%M:%S"
    )

    line = (
        f"[{timestamp}] "
        f"{message}\n"
    )

    _log_history.append(line)

    _command_queue.put(
        ("log", line)
    )


# ============================================================
# STOP LOG SYSTEM
# ============================================================

def stop_log_window():

    if not _started:
        return

    _command_queue.put("quit")