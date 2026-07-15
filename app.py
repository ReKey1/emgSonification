"""cheezEMG desktop UI.

A small control panel for the sensor:
    * Connect to the board (serial) or run the synthetic generator.
    * Pick mains frequency  — 50 Hz eastern Japan / 60 Hz western Japan.
    * Watch the live signal (raw vs cleaned) and the envelope.
    * See skin-contact status  — important for the dry PCB pads.
    * Start / Stop a recording, saved into recordings/<title>/<timestamp>_<notes>/.
    * Toggle sonification on/off.
    * Live feature panel (shows the categorizer framework running).

Run:  python app.py
"""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception as e:  # pragma: no cover
    print("matplotlib with the TkAgg backend is required for the UI.\n"
          f"Import failed: {e}\nInstall with: pip install matplotlib", file=sys.stderr)
    raise

from emg.config import Config
from emg.pipeline import Pipeline
from emg.sonify import Sonifier, beep
from emg.source import open_source

REFRESH_MS = 33  # ~30 fps


class EmgApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cheezEMG")
        self.geometry("1120x700")
        self.minsize(940, 600)

        self.cfg = Config.load("config.json")
        self.cfg.enabled_features = ["example"]  # demo the framework; see emg/features.py
        self.pipeline = Pipeline(self.cfg)
        self.sonifier: Sonifier | None = None
        self._rec_t0 = 0.0
        self._counting = False              # a record countdown is in progress
        self._countdown_jobs: list[str] = []  # pending `after` ids, so we can cancel

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(REFRESH_MS, self._tick)

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #
    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        side = ttk.Frame(root)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self._build_controls(side)

        plots = ttk.Frame(root)
        plots.grid(row=0, column=1, sticky="nsew")
        self._build_plot(plots)

    def _build_controls(self, p: ttk.Frame) -> None:
        # --- Connection ---
        conn = ttk.LabelFrame(p, text="Connection", padding=8)
        conn.pack(fill="x")
        self.var_source = tk.StringVar(value=self.cfg.source)
        ttk.Radiobutton(conn, text="Serial", variable=self.var_source,
                        value="serial").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(conn, text="Synthetic", variable=self.var_source,
                        value="synthetic").grid(row=0, column=1, sticky="w")
        ttk.Label(conn, text="Port:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_port = tk.StringVar(value=self.cfg.serial_port)
        ttk.Entry(conn, textvariable=self.var_port, width=10).grid(
            row=1, column=1, sticky="w", pady=(4, 0))
        self.btn_conn = ttk.Button(conn, text="Connect", command=self._toggle_connect)
        self.btn_conn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # --- Mains frequency ---
        mains = ttk.LabelFrame(p, text="Mains hum notch (Japan)", padding=8)
        mains.pack(fill="x", pady=(8, 0))
        self.var_mains = tk.DoubleVar(value=self.cfg.mains_hz)
        ttk.Radiobutton(mains, text="50 Hz  (E. Japan / Tokyo)", variable=self.var_mains,
                        value=50.0, command=self._apply_mains).pack(anchor="w")
        ttk.Radiobutton(mains, text="60 Hz  (W. Japan / Osaka)", variable=self.var_mains,
                        value=60.0, command=self._apply_mains).pack(anchor="w")

        # --- Recording ---
        rec = ttk.LabelFrame(p, text="Recording", padding=8)
        rec.pack(fill="x", pady=(8, 0))
        ttk.Label(rec, text="Subject:").grid(row=0, column=0, sticky="w")
        self.var_title = tk.StringVar(value="subject")
        ttk.Entry(rec, textvariable=self.var_title, width=16).grid(row=0, column=1, sticky="ew")
        ttk.Label(rec, text="Dataset:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_notes = tk.StringVar()
        ttk.Entry(rec, textvariable=self.var_notes, width=16).grid(
            row=1, column=1, sticky="ew", pady=(4, 0))
        rec.columnconfigure(1, weight=1)
        self.var_countdown = tk.BooleanVar(value=True)
        ttk.Checkbutton(rec, text="3·2·1 countdown tones", variable=self.var_countdown).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.btn_rec = ttk.Button(rec, text="● Start Recording",
                                  command=self._toggle_record, state="disabled")
        self.btn_rec.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.lbl_rec = ttk.Label(rec, text="not recording", foreground="gray")
        self.lbl_rec.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(rec, text="Open recordings folder",
                   command=self._open_folder).grid(row=5, column=0, columnspan=2,
                                                    sticky="ew", pady=(4, 0))

        # --- Live signal peaks (strongest burst so far) ---
        pk = ttk.LabelFrame(p, text="Signal peaks (live)", padding=8)
        pk.pack(fill="x", pady=(8, 0))
        self.lbl_peaks = ttk.Label(pk, text="raw   —\nfiltered   —",
                                   justify="left", font=("TkFixedFont", 9))
        self.lbl_peaks.pack(anchor="w")
        ttk.Label(pk, text="raw & filtered from the same burst; resets on record",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(2, 0))

        # --- Sonification ---
        son = ttk.LabelFrame(p, text="Sonification", padding=8)
        son.pack(fill="x", pady=(8, 0))
        self.var_audio = tk.BooleanVar(value=False)
        ttk.Checkbutton(son, text="Sound on", variable=self.var_audio,
                        command=self._toggle_audio).pack(anchor="w")
        ttk.Label(son, text="Sensitivity:").pack(anchor="w", pady=(4, 0))
        self.var_gain = tk.DoubleVar(value=self.cfg.envelope_full_scale)
        ttk.Scale(son, from_=500, to=30, orient="horizontal", variable=self.var_gain,
                  command=self._apply_gain).pack(fill="x")
        self.lbl_audio = ttk.Label(son, text="", foreground="gray")
        self.lbl_audio.pack(anchor="w")

        # --- Features (categorizer framework) ---
        feat = ttk.LabelFrame(p, text="Signal qualities (live)", padding=8)
        feat.pack(fill="x", pady=(8, 0))
        self.lbl_features = ttk.Label(feat, text="—", justify="left", font=("TkFixedFont", 9))
        self.lbl_features.pack(anchor="w")
        ttk.Label(feat, text="Add your own in emg/features.py",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(2, 0))

        # --- Status ---
        st = ttk.LabelFrame(p, text="Status", padding=8)
        st.pack(fill="x", pady=(8, 0))
        self.lbl_status = ttk.Label(st, text="disconnected", justify="left")
        self.lbl_status.grid(row=0, column=0, sticky="w")
        cont = ttk.Frame(st)
        cont.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.contact_dot = tk.Canvas(cont, width=14, height=14, highlightthickness=0)
        self._dot = self.contact_dot.create_oval(2, 2, 12, 12, fill="gray", outline="")
        self.contact_dot.pack(side="left")
        self.lbl_contact = ttk.Label(cont, text="contact —")
        self.lbl_contact.pack(side="left", padx=(4, 0))

    def _build_plot(self, p: ttk.Frame) -> None:
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.ax1.set_title("Signal")
        self.ax1.set_ylabel("filtered")
        self.ax2.set_ylabel("envelope")
        self.ax2.set_xlabel("time (s)")
        (self.ln_raw,) = self.ax1.plot([], [], lw=0.6, color="#c9c9c9", label="raw (centered)")
        (self.ln_filt,) = self.ax1.plot([], [], lw=0.9, color="#1f77b4", label="filtered")
        (self.ln_env,) = self.ax2.plot([], [], lw=1.1, color="#d62728", label="envelope")
        self.ax1.legend(loc="upper right", fontsize=8)
        self.ax1.grid(alpha=0.25)
        self.ax2.grid(alpha=0.25)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=p)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    #  Actions
    # ------------------------------------------------------------------ #
    def _toggle_connect(self) -> None:
        if self.pipeline.running:
            self._disconnect()
            return
        self.cfg.source = self.var_source.get()
        self.cfg.serial_port = self.var_port.get().strip()
        self.cfg.mains_hz = self.var_mains.get()
        self.pipeline.reconfigure(self.cfg)
        try:
            source = open_source(self.cfg)
        except Exception as e:
            messagebox.showerror(
                "Connection failed",
                f"Could not open {self.cfg.source} source:\n{e}\n\n"
                "Tip: pick 'Synthetic' to run without hardware.")
            return
        self.pipeline.start(source)
        self.btn_conn.config(text="Disconnect")
        self.btn_rec.config(state="normal")

    def _disconnect(self) -> None:
        self._cancel_countdown()
        if self.pipeline.is_recording:
            self._stop_record()
        self.pipeline.stop()
        self.btn_conn.config(text="Connect")
        self.btn_rec.config(state="disabled")

    def _toggle_record(self) -> None:
        if self.pipeline.is_recording:
            self._stop_record()
            return
        if self._counting:
            return  # countdown already running — ignore extra clicks
        if not self.pipeline.running:
            messagebox.showwarning("Not connected", "Connect to a source first.")
            return
        if self.var_countdown.get():
            self._begin_countdown()
        else:
            self._begin_record()

    def _begin_countdown(self) -> None:
        """Play three spaced tones; recording starts on the third (Mario-Kart style)."""
        self._counting = True
        self.btn_rec.config(state="disabled")
        self._countdown(3)

    def _countdown(self, n: int) -> None:
        go = n <= 1  # the third (last) tone is the higher "go" — recording starts on it
        beep(880.0 if go else 587.0, 0.18)
        if go:
            self._counting = False
            self._countdown_jobs.clear()
            self.btn_rec.config(state="normal")
            self._begin_record()
            return
        self.lbl_rec.config(text=f"●  {n}…", foreground="#d68a00")
        self._countdown_jobs.append(self.after(1000, lambda: self._countdown(n - 1)))

    def _begin_record(self) -> None:
        path = self.pipeline.start_recording(self.var_title.get(), self.var_notes.get())
        self._rec_t0 = time.time()
        self.btn_rec.config(text="■ Stop Recording")
        self.lbl_rec.config(text=f"→ {os.path.basename(path)}", foreground="#d62728")

    def _stop_record(self) -> None:
        path = self.pipeline.stop_recording()
        self.btn_rec.config(text="● Start Recording")
        self.lbl_rec.config(text=f"saved: {os.path.basename(path)}", foreground="gray")

    def _cancel_countdown(self) -> None:
        for job in self._countdown_jobs:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._countdown_jobs.clear()
        self._counting = False

    def _toggle_audio(self) -> None:
        if self.var_audio.get():
            self.sonifier = Sonifier(self.cfg, lambda: self.pipeline.audio_level)
            if not self.sonifier.start():
                self.var_audio.set(False)
                self.lbl_audio.config(text=f"audio error: {self.sonifier.error}")
            else:
                self.lbl_audio.config(text="playing")
        else:
            if self.sonifier:
                self.sonifier.stop()
            self.lbl_audio.config(text="")

    def _apply_mains(self) -> None:
        self.cfg.mains_hz = self.var_mains.get()
        self.pipeline.reconfigure(self.cfg)

    def _apply_gain(self, _evt=None) -> None:
        self.cfg.envelope_full_scale = float(self.var_gain.get())

    def _open_folder(self) -> None:
        path = self.cfg.recordings_path()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # Windows
        except AttributeError:  # pragma: no cover - non-Windows fallback
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    # ------------------------------------------------------------------ #
    #  Refresh loop
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        try:
            self._update_plot()
            self._update_status()
        except Exception as e:  # never let the loop die
            print(f"UI tick error: {e}", file=sys.stderr)
        finally:
            self.after(REFRESH_MS, self._tick)

    def _update_plot(self) -> None:
        t, raw, filt, env = self.pipeline.plot_snapshot()
        if t.size < 2:
            return
        raw_c = raw - raw.mean()  # center raw so it overlays the filtered scale
        self.ln_raw.set_data(t, raw_c)
        self.ln_filt.set_data(t, filt)
        self.ln_env.set_data(t, env)
        self.ax1.set_xlim(t[0], t[-1])
        m = max(10.0, float(max(abs(filt.min()), abs(filt.max()), abs(raw_c).max())))
        self.ax1.set_ylim(-m, m)
        self.ax2.set_ylim(0, max(10.0, float(env.max()) * 1.1))
        self.canvas.draw_idle()

    def _update_status(self) -> None:
        p = self.pipeline
        if p.error:
            self.lbl_status.config(text=f"ERROR: {p.error}", foreground="#d62728")
        elif p.running:
            self.lbl_status.config(
                text=f"source: {p.source_name}\n"
                     f"rate:   {p.measured_rate:5.1f} Hz\n"
                     f"samples:{p.samples_seen}",
                foreground="black")
        else:
            self.lbl_status.config(text="disconnected", foreground="gray")

        ok = p.contact_ok and p.running
        self.contact_dot.itemconfig(self._dot, fill="#2ca02c" if ok else "#d62728")
        self.lbl_contact.config(text="contact OK" if ok else "contact POOR / off")

        if p.running:
            self.lbl_peaks.config(
                text=f"raw   {p.peak_raw:8.1f}\nfiltered   {p.peak_filtered:8.2f}")
        else:
            self.lbl_peaks.config(text="raw   —\nfiltered   —")

        if p.is_recording:
            self.lbl_rec.config(
                text=f"● REC {time.time() - self._rec_t0:5.1f}s "
                     f"({p.recorder.sample_count} samples)", foreground="#d62728")

        feats = p.compute_features()
        if feats:
            lines = []
            for name, r in feats.items():
                val = "—" if r.value is None else f"{r.value:8.2f}"
                lines.append(f"{name:16s} {val}")
            self.lbl_features.config(text="\n".join(lines))

    def _on_close(self) -> None:
        try:
            self._cancel_countdown()
            if self.sonifier:
                self.sonifier.stop()
            self.pipeline.stop()
        finally:
            self.destroy()


def main() -> int:
    EmgApp().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
