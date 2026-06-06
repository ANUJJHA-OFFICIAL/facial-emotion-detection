import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys

# ── local modules ──────────────────────────────────
from train_mode import run_train_mode
from use_mode   import run_use_mode


# ──────────────────────────────────────────────────
#  Helper – run a blocking function in a thread so
#  the Tk event-loop stays alive.
# ──────────────────────────────────────────────────
def _run_in_thread(fn, on_done_msg: str, status_var: tk.StringVar,
                   btn_train, btn_use):
    """Disable buttons, launch fn in a daemon thread, re-enable when done."""

    def wrapper():
        btn_train.config(state="disabled")
        btn_use.config(state="disabled")
        try:
            fn(status_var)
        except Exception as exc:
            status_var.set(f"Error: {exc}")
        finally:
            status_var.set(on_done_msg)
            btn_train.config(state="normal")
            btn_use.config(state="normal")

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()


# ──────────────────────────────────────────────────
#  Main Application Window
# ──────────────────────────────────────────────────
class EmotionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🎭  Facial Emotion Detection")
        self.resizable(False, False)
        self._build_ui()
        self._center_window()

    # ── layout ────────────────────────────────────
    def _build_ui(self):
        BG      = "#0d1117"
        CARD    = "#161b22"
        ACCENT  = "#58a6ff"
        TEXT    = "#e6edf3"
        SUBTEXT = "#8b949e"
        GREEN   = "#3fb950"
        RED     = "#f85149"

        self.configure(bg=BG)

        # ── header ──
        hdr = tk.Frame(self, bg=BG, pady=20)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎭", font=("Segoe UI Emoji", 36),
                 bg=BG, fg=ACCENT).pack()
        tk.Label(hdr, text="Facial Emotion Detection",
                 font=("Segoe UI", 22, "bold"),
                 bg=BG, fg=TEXT).pack()
        tk.Label(hdr, text="Real-time deep-learning emotion recognition",
                 font=("Segoe UI", 10),
                 bg=BG, fg=SUBTEXT).pack(pady=(4, 0))

        # ── card ──
        card = tk.Frame(self, bg=CARD, padx=40, pady=30,
                        relief="flat", bd=0)
        card.pack(padx=30, pady=10, fill="both")

        # Emotion badges
        badge_frame = tk.Frame(card, bg=CARD)
        badge_frame.pack(pady=(0, 20))
        emotions = [("😊", "Happy"), ("😢", "Sad"), ("😠", "Angry"),
                    ("😐", "Neutral"), ("😲", "Surprise"), ("😨", "Fear")]
        for i, (ico, lbl) in enumerate(emotions):
            f = tk.Frame(badge_frame, bg="#21262d", padx=8, pady=4)
            f.grid(row=0, column=i, padx=4)
            tk.Label(f, text=ico, font=("Segoe UI Emoji", 14),
                     bg="#21262d").pack()
            tk.Label(f, text=lbl, font=("Segoe UI", 8),
                     bg="#21262d", fg=SUBTEXT).pack()

        # Divider
        tk.Frame(card, bg="#30363d", height=1).pack(fill="x", pady=14)

        # Buttons
        btn_cfg = dict(font=("Segoe UI", 13, "bold"),
                       width=22, pady=10, bd=0, cursor="hand2",
                       relief="flat")

        self.btn_train = tk.Button(
            card, text="🧠  Train Mode",
            bg=ACCENT, fg="#0d1117", activebackground="#79c0ff",
            command=self._start_train, **btn_cfg)
        self.btn_train.pack(pady=6)

        self.btn_use = tk.Button(
            card, text="📷  Use Mode",
            bg=GREEN, fg="#0d1117", activebackground="#56d364",
            command=self._start_use, **btn_cfg)
        self.btn_use.pack(pady=6)

        self.btn_retrain = tk.Button(
            card, text="🔄  Retrain Model",
            bg="#21262d", fg=TEXT, activebackground="#30363d",
            command=self._retrain, **btn_cfg)
        self.btn_retrain.pack(pady=6)

        self.btn_exit = tk.Button(
            card, text="✖  Exit",
            bg="#21262d", fg=RED, activebackground="#30363d",
            command=self._exit, **btn_cfg)
        self.btn_exit.pack(pady=6)

        # Status bar
        tk.Frame(self, bg="#30363d", height=1).pack(fill="x", padx=30)
        self.status_var = tk.StringVar(value="Ready — select a mode to begin.")
        status_bar = tk.Frame(self, bg=BG, pady=10)
        status_bar.pack(fill="x")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("Segoe UI", 10), bg=BG, fg=SUBTEXT).pack()

        # Progress bar (hidden until training)
        self.progress = ttk.Progressbar(self, mode="indeterminate",
                                        length=300)

        # Style progress bar
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TProgressbar",
                        troughcolor=CARD,
                        background=ACCENT,
                        bordercolor=CARD,
                        lightcolor=ACCENT,
                        darkcolor=ACCENT)

        # Footer
        tk.Label(self, text="Press Q inside webcam window to quit  •  "
                             "Built with OpenCV + TensorFlow",
                 font=("Segoe UI", 8), bg=BG, fg="#484f58").pack(pady=(0, 12))

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw    = self.winfo_screenwidth()
        sh    = self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    # ── button callbacks ──────────────────────────
    def _start_train(self):
        self.status_var.set("📷  Train Mode active — webcam opening…")
        _run_in_thread(
            run_train_mode,
            "Train Mode closed. Press T inside webcam to train model.",
            self.status_var,
            self.btn_train, self.btn_use)

    def _start_use(self):
        if not os.path.exists("model.keras"):
            messagebox.showerror(
                "Model Not Found",
                "No trained model found!\n\n"
                "Please run Train Mode first and press T to train.")
            return
        self.status_var.set("📷  Use Mode active — real-time detection running…")
        _run_in_thread(
            run_use_mode,
            "Use Mode closed.",
            self.status_var,
            self.btn_train, self.btn_use)

    def _retrain(self):
        from trainer import train_model
        self.status_var.set("🔄  Training model…")
        self.progress.pack(pady=4)
        self.progress.start(10)

        def do_train(sv):
            train_model(status_callback=sv.set)

        def wrapper():
            self.btn_train.config(state="disabled")
            self.btn_use.config(state="disabled")
            try:
                do_train(self.status_var)
            except Exception as exc:
                self.status_var.set(f"Training error: {exc}")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                self.status_var.set("✅  Training complete! Model saved.")
                self.btn_train.config(state="normal")
                self.btn_use.config(state="normal")

        threading.Thread(target=wrapper, daemon=True).start()

    def _exit(self):
        self.destroy()
        sys.exit(0)


# ──────────────────────────────────────────────────
if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
