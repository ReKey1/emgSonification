"""cheezEMG recorded-session viewer.

A read-only companion to app.py: instead of watching the live signal, browse
sessions already saved under recordings/ and inspect them after the fact.

The tree mirrors how the data is organised — subject -> movement category -> rep:

    * Pick a CATEGORY to see all of its reps at once (overlay or 2x3 grid),
      normalised to %MVC and aligned at the movement cue, with the mean envelope
      and its spread — a direct read on inter-rep consistency.
    * Pick a single REP to see that one recording in full: raw vs cleaned signal,
      the envelope with the detected rep shaded and the cue marked, and a power
      spectrum with the mains harmonics flagged.
    * The `amp (MVC)` node is the subject's max-contraction reference; its robust
      max defines 100% MVC for every category under that subject.
    * Every metric shown is the same number score_cli.py writes (scores.csv for a
      category, reps.csv for a rep) — nothing here re-derives the maths.
    * Zoom / pan / measure with the matplotlib toolbar.
    * The `Data (CSV)` tab shows scores.csv and reps.csv themselves — the tables
      score_cli.py wrote — sortable by column and tracking the tree selection.

Run:  python view.py
"""

from __future__ import annotations

import csv
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
except Exception as e:  # pragma: no cover
    print("matplotlib with the TkAgg backend is required for the viewer.\n"
          f"Import failed: {e}\nInstall with: pip install matplotlib", file=sys.stderr)
    raise

import numpy as np
from scipy import signal as sp_signal

from emg.config import Config
from emg.scoring import (
    AMP_CATEGORY,
    COLUMN_LABELS,
    DatasetContext,
    REP_METRIC_COLUMNS,
    SCORE_COLUMN,
    _fmt,
    aggregate_category,
    find_subjects,
    load_dataset,
    rep_profile,
    rep_row,
    rep_window,
    robust_max,
    subject_mvc,
)

# Match the live plot in app.py so a recording looks the same reviewed as recorded.
COL_RAW = "#c9c9c9"
COL_FILT = "#1f77b4"
COL_ENV = "#d62728"
COL_REP = "#ffcc66"      # rep shading
COL_ONSET = "#2ca02c"    # movement-cue marker
COL_MAINS = "#d62728"    # mains-harmonic markers on the spectrum
COL_MEAN = "#1f77b4"     # mean envelope in the category overlay
COL_FAINT = "#8aa9c8"    # individual reps behind the mean
COL_MVC = "#999999"      # 100% MVC reference line

# Cue<->cross-correlation fusion for the category overlay (see Viewer._align_offsets).
ALIGN_ITERS = 3             # template/offset co-adaptation passes
ALIGN_CUE_PRECISION = 0.15  # weight of the cue anchor vs a confident correlation (c²)

COL_MATCH = "#fff2c4"    # CSV rows belonging to the current tree selection

# CSV sheets: keep the identity columns narrow-ish and the rest readable.
CSV_MIN_COL_PX = 62
CSV_MAX_COL_PX = 210
CSV_PX_PER_CHAR = 7

REP_META_FIELDS = [
    ("subject", "subject"),
    ("category", "category"),
    ("rep", "rep"),
    ("started", "started"),
    ("duration_s", "duration (s)"),
    ("n_samples", "samples"),
    ("sample_rate", "sample_rate (Hz)"),
    ("mains_hz", "mains (Hz)"),
    ("movement_onset_s", "movement onset (s)"),
]


def _sort_key(cell: str):
    """Sort blanks last, numbers numerically, everything else as text."""
    if cell == "":
        return (2, 0.0, "")
    try:
        return (0, float(cell), "")
    except ValueError:
        return (1, 0.0, cell.lower())


class CsvSheet(ttk.Frame):
    """One CSV rendered as a table: the file exactly as score_cli.py wrote it.

    Rows belonging to the current tree selection are highlighted and scrolled to;
    `only selection` narrows the table down to just those rows. Click a column
    heading to sort by it (numeric where the column is numeric).
    """

    def __init__(self, parent: ttk.Widget, path: Path, hint: str):
        super().__init__(parent)
        self.path = path
        self.hint = hint
        self.headers: List[str] = []
        self.rows: List[List[str]] = []
        self._match: Optional[Callable[[Dict[str, str]], bool]] = None
        self._sort_col: Optional[str] = None
        self._sort_desc = False

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        ttk.Label(bar, text=path.name,
                  font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.lbl_status = ttk.Label(bar, text="", foreground="gray",
                                    font=("TkDefaultFont", 8))
        self.lbl_status.pack(side="left", padx=(8, 0))
        self.var_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="only selection", variable=self.var_only,
                        command=self._refill).pack(side="right")

        self.tv = ttk.Treeview(self, show="headings", selectmode="browse", height=8)
        self.tv.grid(row=1, column=0, sticky="nsew")
        self.tv.tag_configure("match", background=COL_MATCH)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tv.yview)
        vsb.grid(row=1, column=1, sticky="ns")
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tv.xview)
        hsb.grid(row=2, column=0, sticky="ew")
        self.tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    # -------------------------------------------------------------- #
    def reload(self) -> None:
        """Re-read the CSV from disk (cheap — these files are small)."""
        self.headers, self.rows = [], []
        if self.path.exists():
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    self.headers = next(reader, []) or []
                    self.rows = [r for r in reader if any(c.strip() for c in r)]
            except Exception as e:
                self.headers, self.rows = [], []
                self.lbl_status.config(text=f"could not read: {e}")
                self._clear_columns()
                return
        self._build_columns()
        self._refill()

    def set_match(self, match: Optional[Callable[[Dict[str, str]], bool]]) -> None:
        """Predicate over a row (as a header->cell dict) marking the selection."""
        self._match = match
        self._refill()

    # -------------------------------------------------------------- #
    def _clear_columns(self) -> None:
        self.tv.delete(*self.tv.get_children())
        self.tv["columns"] = ()

    def _build_columns(self) -> None:
        self._clear_columns()
        self.tv["columns"] = tuple(self.headers)
        for i, h in enumerate(self.headers):
            widest = max([len(h)] + [len(r[i]) for r in self.rows[:200]
                                     if i < len(r)])
            width = min(CSV_MAX_COL_PX, max(CSV_MIN_COL_PX,
                                            widest * CSV_PX_PER_CHAR + 12))
            anchor = "w" if h in ("subject", "category", "rep", "started") else "e"
            self.tv.heading(h, text=h, command=lambda c=h: self._sort_by(c))
            self.tv.column(h, width=width, anchor=anchor, stretch=False)

    def _sort_by(self, column: str) -> None:
        self._sort_desc = not self._sort_desc if self._sort_col == column else False
        self._sort_col = column
        self._refill()

    def _ordered(self) -> List[List[str]]:
        if self._sort_col is None or self._sort_col not in self.headers:
            return list(self.rows)
        i = self.headers.index(self._sort_col)
        return sorted(self.rows, reverse=self._sort_desc,
                      key=lambda r: _sort_key(r[i] if i < len(r) else ""))

    def _is_match(self, row: List[str]) -> bool:
        if self._match is None:
            return False
        return self._match(dict(zip(self.headers, row)))

    def _refill(self) -> None:
        self.tv.delete(*self.tv.get_children())
        if not self.headers:
            self.lbl_status.config(
                text=f"not found — run `python score_cli.py` to write {self.path.name}")
            return

        only = self.var_only.get()
        first: Optional[str] = None
        shown = 0
        for row in self._ordered():
            hit = self._is_match(row)
            if only and not hit:
                continue
            item = self.tv.insert("", "end", values=row,
                                  tags=("match",) if hit else ())
            shown += 1
            if hit and first is None:
                first = item
        if first is not None:
            self.tv.see(first)

        total = len(self.rows)
        note = f"{shown} of {total} rows" if shown != total else f"{total} rows"
        if only and self._match is None:
            note += " — select a category or rep"
        self.lbl_status.config(text=f"{note}  ·  {self.hint}")


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("cheezEMG — session viewer")
        self.geometry("1240x840")
        self.minsize(1000, 660)

        self.cfg = Config.load("config.json")
        self.root_dir = self.cfg.recordings_path()
        self._nodes: Dict[str, tuple] = {}     # tree item id -> descriptor tuple
        self._mvc_cache: Dict[str, Optional[float]] = {}
        self._current: Optional[tuple] = None  # last-rendered descriptor

        self._build_widgets()
        self._scan()

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #
    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_browser(root)

        self.tabs = ttk.Notebook(root)
        self.tabs.grid(row=0, column=1, sticky="nsew")

        plots = ttk.Frame(self.tabs, padding=(0, 6, 0, 0))
        plots.rowconfigure(1, weight=1)
        plots.columnconfigure(0, weight=1)
        self.tabs.add(plots, text="Plots")
        self._build_toolbar(plots)
        self._build_plot(plots)
        self._build_panel(plots)

        data = ttk.Frame(self.tabs, padding=6)
        self.tabs.add(data, text="Data (CSV)")
        self._build_data_tab(data)

    def _build_browser(self, parent: ttk.Frame) -> None:
        side = ttk.Frame(parent)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        side.rowconfigure(1, weight=1)

        ttk.Label(side, text="Sessions", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.tree = ttk.Treeview(side, show="tree", height=34, selectmode="browse")
        self.tree.grid(row=1, column=0, sticky="ns")
        scroll = ttk.Scrollbar(side, orient="vertical", command=self.tree.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        ttk.Button(side, text="Refresh", command=self._scan).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.lbl_root = ttk.Label(side, text="", foreground="gray",
                                  font=("TkDefaultFont", 8), wraplength=210)
        self.lbl_root.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(bar, text="Category view:").pack(side="left")
        self.var_mode = tk.StringVar(value="overlay")
        for text, val in (("Overlay", "overlay"), ("Grid", "grid")):
            ttk.Radiobutton(bar, text=text, value=val, variable=self.var_mode,
                            command=self._on_mode_change).pack(side="left", padx=(4, 0))
        self.lbl_view = ttk.Label(bar, text="", foreground="gray")
        self.lbl_view.pack(side="left", padx=(12, 0))

    def _build_plot(self, parent: ttk.Frame) -> None:
        holder = ttk.Frame(parent)
        holder.grid(row=1, column=0, sticky="nsew")
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        toolbar = NavigationToolbar2Tk(self.canvas, holder)  # zoom / pan / measure
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._message("Select a session")

    def _build_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        meta = ttk.LabelFrame(panel, text="Selection", padding=6)
        meta.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.tv_meta = self._make_table(meta)

        mets = ttk.LabelFrame(panel, text="Metrics", padding=6)
        mets.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.tv_metrics = self._make_table(mets)

    def _build_data_tab(self, parent: ttk.Frame) -> None:
        """The two CSVs score_cli.py writes, stacked: categories over reps."""
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        panes = ttk.PanedWindow(parent, orient="vertical")
        panes.grid(row=0, column=0, sticky="nsew")

        self.sheet_scores = CsvSheet(panes, self.root_dir / "scores.csv",
                                     "one row per subject × category")
        self.sheet_reps = CsvSheet(panes, self.root_dir / "reps.csv",
                                   "one row per rep file")
        panes.add(self.sheet_scores, weight=1)
        panes.add(self.sheet_reps, weight=1)

    def _reload_csvs(self) -> None:
        for sheet in (self.sheet_scores, self.sheet_reps):
            sheet.reload()
        self._sync_csvs(self._current)

    def _sync_csvs(self, desc: Optional[tuple]) -> None:
        """Point both sheets at the rows for the current tree selection."""
        if desc is None or desc[0] == "amp":
            self.sheet_scores.set_match(None)
            self.sheet_reps.set_match(None)
            return
        subject, category = desc[1], desc[2]

        def cat_match(row: Dict[str, str]) -> bool:
            return row.get("subject") == subject and row.get("category") == category

        self.sheet_scores.set_match(cat_match)
        if desc[0] == "rep":
            rep_name = desc[3].name
            self.sheet_reps.set_match(
                lambda row: cat_match(row) and row.get("rep") == rep_name)
        else:
            self.sheet_reps.set_match(cat_match)

    def _make_table(self, parent: ttk.Widget) -> ttk.Treeview:
        tv = ttk.Treeview(parent, columns=("field", "value"), show="headings",
                          height=11, selectmode="none")
        for col, anchor, width in (("field", "w", 200), ("value", "e", 110)):
            tv.heading(col, text="")
            tv.column(col, anchor=anchor, width=width, stretch=True)
        tv.tag_configure("score", font=("TkDefaultFont", 9, "bold"))
        tv.pack(fill="both", expand=True)
        return tv

    # ------------------------------------------------------------------ #
    #  Browser: subject -> [amp, category -> rep]
    # ------------------------------------------------------------------ #
    def _scan(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._nodes.clear()
        self._mvc_cache.clear()
        self.lbl_root.config(text=str(self.root_dir))
        self._reload_csvs()

        groups = find_subjects(self.root_dir)
        if not groups:
            self.tree.insert("", "end", text="(no recordings found)")
            return

        for g in groups:
            snode = self.tree.insert("", "end", text=g.subject, open=False)
            if g.amp_dir is not None:
                node = self.tree.insert(snode, "end", text="amp (MVC)")
                self._nodes[node] = ("amp", g.subject, g.amp_dir)
            for category in sorted(g.categories):
                reps = g.categories[category]
                cnode = self.tree.insert(
                    snode, "end", text=f"{category}  ({len(reps)})")
                self._nodes[cnode] = ("category", g.subject, category, reps)
                for d in reps:
                    rnode = self.tree.insert(cnode, "end", text=d.name)
                    self._nodes[rnode] = ("rep", g.subject, category, d)

    def _mvc_for(self, subject: str) -> Optional[float]:
        if subject not in self._mvc_cache:
            amp_dir = None
            for desc in self._nodes.values():
                if desc[0] == "amp" and desc[1] == subject:
                    amp_dir = desc[2]
                    break
            self._mvc_cache[subject] = subject_mvc(amp_dir, self.root_dir)
        return self._mvc_cache[subject]

    def _on_select(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        desc = self._nodes.get(sel[0])
        if desc is None:
            return  # a subject node — nothing to render
        self._render(desc)

    def _on_mode_change(self) -> None:
        if self._current is not None and self._current[0] == "category":
            self._render(self._current)

    def _render(self, desc: tuple) -> None:
        self._current = desc
        self._sync_csvs(desc)
        kind = desc[0]
        try:
            if kind == "category":
                self._render_category(*desc[1:])
            else:  # "rep" or "amp"
                self._render_single(desc)
        except Exception as e:
            self._show_error(f"{desc}:\n{e}")
            return
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Single-recording view (one rep, or the amp/MVC file)
    # ------------------------------------------------------------------ #
    def _render_single(self, desc: tuple) -> None:
        kind, subject, category = desc[0], desc[1], desc[2]
        path = desc[3] if kind == "rep" else desc[2]
        if kind == "amp":
            category, path = AMP_CATEGORY, desc[2]
        mvc = self._mvc_for(subject)
        ctx = load_dataset(path, self.root_dir)

        self.fig.clf()
        ax_sig = self.fig.add_subplot(3, 1, 1)
        ax_env = self.fig.add_subplot(3, 1, 2, sharex=ax_sig)
        ax_spec = self.fig.add_subplot(3, 1, 3)
        self.fig.subplots_adjust(hspace=0.35, left=0.1, right=0.97, top=0.94, bottom=0.08)

        t = ctx.t
        raw_c = ctx.raw - float(ctx.raw.mean()) if ctx.raw.size else ctx.raw
        ax_sig.plot(t, raw_c, lw=0.6, color=COL_RAW, label="raw (centered)")
        ax_sig.plot(t, ctx.filtered, lw=0.9, color=COL_FILT, label="filtered")
        ax_env.plot(t, ctx.envelope, lw=1.1, color=COL_ENV, label="envelope")

        # Shade what the scorer actually measures: the detected burst for a rep, or
        # the amplitude-detected burst for the amp file, which has no cue. The cue
        # line is drawn separately below so the reaction-time gap is visible.
        win = rep_window(ctx)
        s, e = (win[0], win[1]) if win is not None else ctx.rep()
        if e > s and e <= t.size:
            ax_env.axvspan(t[s], t[min(e, t.size) - 1], color=COL_REP, alpha=0.35)
        onset = ctx.meta.get("movement_onset_t")
        if onset is not None:
            for ax in (ax_sig, ax_env):
                ax.axvline(float(onset), color=COL_ONSET, lw=1.2, ls="--")

        tag = "amp (MVC reference)" if kind == "amp" else f"{category} / {ctx.dataset}"
        ax_sig.set_title(f"{subject} — {tag}")
        ax_sig.set_ylabel("filtered")
        ax_env.set_ylabel("envelope")
        ax_env.set_xlabel("time (s)")
        ax_sig.legend(loc="upper right", fontsize=8)
        for ax in (ax_sig, ax_env):
            ax.grid(alpha=0.25)
        if t.size:
            ax_sig.set_xlim(float(t[0]), float(t[-1]))
        self._draw_spectrum(ax_spec, ctx)

        # tables
        if kind == "amp":
            self._fill_meta([("subject", subject), ("recording", ctx.dataset),
                             ("started", ctx.meta.get("started", "—")),
                             ("duration_s", ctx.meta.get("duration_s", "—")),
                             ("MVC ref (counts)",
                              _fmt(robust_max(ctx.envelope, ctx.fs)))])
            self._fill_metrics_rows([("note", "used as 100% MVC for this subject")])
            self.lbl_view.config(text="MVC reference recording")
        else:
            row = rep_row(ctx, subject, category, mvc)
            meta_src = dict(ctx.meta)
            meta_src.update({"subject": subject, "category": category, "rep": ctx.dataset})
            self._fill_meta([(lbl, self._meta_value(meta_src, key))
                             for key, lbl in REP_META_FIELDS])
            self._fill_metrics_from_row(row, REP_METRIC_COLUMNS)
            self.lbl_view.config(text="single rep")

    def _draw_spectrum(self, ax, ctx: DatasetContext) -> None:
        ax.set_ylabel("PSD")
        ax.set_xlabel("frequency (Hz)")
        ax.grid(alpha=0.25)
        f, fs = ctx.filtered, ctx.fs
        if f.size < 64 or fs <= 0:
            ax.text(0.5, 0.5, "insufficient data for spectrum", ha="center",
                    va="center", color="gray", transform=ax.transAxes)
            return
        nperseg = int(min(f.size, max(64, fs)))
        freqs, psd = sp_signal.welch(f, fs=fs, nperseg=nperseg)
        ax.semilogy(freqs, psd + 1e-12, lw=0.9, color=COL_FILT)
        nyq = fs / 2.0
        k = 1
        while ctx.cfg.mains_hz * k < 0.99 * nyq:
            ax.axvline(ctx.cfg.mains_hz * k, color=COL_MAINS, lw=0.8, ls=":",
                       alpha=0.7, label="mains + harmonics" if k == 1 else None)
            k += 1
        ax.set_xlim(0, nyq)
        ax.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------------ #
    #  Category view: all reps together
    # ------------------------------------------------------------------ #
    def _render_category(self, subject: str, category: str, rep_dirs: List[Path]) -> None:
        mvc = self._mvc_for(subject)
        reps: List[DatasetContext] = []
        rep_rows: List[dict] = []
        profiles: List[Optional[np.ndarray]] = []
        for d in rep_dirs:
            ctx = load_dataset(d, self.root_dir)
            reps.append(ctx)
            rep_rows.append(rep_row(ctx, subject, category, mvc))
            win = rep_window(ctx)
            s, e = (win[0], win[1]) if win is not None else ctx.rep()
            profiles.append(rep_profile(ctx.envelope[s:e]))

        self.fig.clf()
        if self.var_mode.get() == "grid":
            self._draw_grid(reps, subject, category, mvc)
        else:
            self._draw_overlay(reps, profiles, subject, category, mvc)

        # tables: aggregate = the scores.csv row for this category
        cat = aggregate_category(subject, category, mvc, rep_rows)
        self._fill_meta([("subject", subject), ("category", category),
                         ("n_reps", str(cat["n_reps"])),
                         ("MVC ref (counts)", _fmt(cat["mvc_reference"]))])
        self._fill_metrics_from_row(
            cat, (*REP_METRIC_COLUMNS, SCORE_COLUMN), score_key=SCORE_COLUMN)
        self.lbl_view.config(text=f"{len(reps)} reps — {self.var_mode.get()}")

    def _yscale(self, env: np.ndarray, mvc: Optional[float]) -> Tuple[np.ndarray, str]:
        if mvc and mvc > 0:
            return env / mvc * 100.0, "%MVC"
        return env, "envelope"

    def _onset_offset(self, ctx: DatasetContext) -> float:
        """Fallback offset: the movement cue, or the detected rep onset if absent."""
        onset = ctx.meta.get("movement_onset_t")
        if onset is None:
            s, _ = ctx.rep()
            onset = float(ctx.t[s]) if ctx.t.size else 0.0
        return float(onset)

    def _align_offsets(self, reps: List[DatasetContext]) -> List[float]:
        """Per-rep time offsets (s) that align the reps by *fusing* two estimates,
        odometry-style — neither one is trusted alone:

          * the **cue anchor** `a_i` (movement_onset_t, or the detected onset):
            absolute and drift-free, but noisy from reaction-time jitter;
          * the **cross-correlation** offset `x_i`: precise at lining up the shared
            burst shape, but can lock onto the wrong feature for an odd rep and has
            no absolute reference of its own.

        They influence each other in a short iterated loop (a complementary filter):
        each pass builds the mean envelope (the template) from the *current* fused
        offsets, cross-correlates every rep to it for `x_i` and a confidence `c_i`
        (normalised correlation peak), then fuses per rep by precision weighting

            o_i = (LAMBDA·a_i + c_i²·x_i) / (LAMBDA + c_i²)

        so a rep with a clean, confident match follows the correlation while a weak /
        ambiguous one is held near its cue. The cue terms keep the whole group pinned
        to an absolute frame (0 = the cue) instead of drifting. `ALIGN_ITERS` passes;
        the template and the offsets co-adapt until they settle.
        """
        fs = reps[0].fs or 500.0
        anchors = [self._onset_offset(r) for r in reps]  # absolute cue/onset prior
        offsets = list(anchors)                          # start from the cue
        grid = np.arange(-1.0, 2.5, 1.0 / fs)

        for _ in range(ALIGN_ITERS):
            stacks: List[np.ndarray] = []
            for r, off in zip(reps, offsets):
                if r.t.size < 2:
                    stacks.append(np.zeros_like(grid))
                    continue
                e = r.envelope - np.median(r.envelope)
                stacks.append(np.interp(grid, r.t - off, e, left=0.0, right=0.0))
            template = np.vstack(stacks).mean(axis=0) if stacks else np.array([])
            t_norm = float(np.linalg.norm(template))
            if t_norm < 1e-9:
                break

            fused: List[float] = []
            for off, s, a in zip(offsets, stacks, anchors):
                s_norm = float(np.linalg.norm(s))
                if s_norm < 1e-9:
                    fused.append(a)  # flat rep -> trust the cue entirely
                    continue
                corr = sp_signal.correlate(template, s, mode="full")
                lags = sp_signal.correlation_lags(template.size, s.size, mode="full")
                k = int(np.argmax(corr))
                lag = int(lags[k])
                conf = float(np.clip(corr[k] / (s_norm * t_norm), 0.0, 1.0))
                x = off - lag / fs                      # cross-correlation estimate
                w = conf * conf                          # precision (sharpened)
                fused.append((ALIGN_CUE_PRECISION * a + w * x)
                             / (ALIGN_CUE_PRECISION + w))
            offsets = fused
        return offsets

    def _draw_overlay(self, reps: List[DatasetContext],
                      profiles: List[Optional[np.ndarray]],
                      subject: str, category: str, mvc: Optional[float]) -> None:
        ax_real = self.fig.add_subplot(2, 1, 1)
        ax_norm = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(hspace=0.32, left=0.1, right=0.97, top=0.92, bottom=0.1)
        unit = "%MVC" if (mvc and mvc > 0) else "envelope"

        # Top: real-time envelopes aligned by fusing the cue anchor with feature
        # cross-correlation (see _align_offsets) — amplitude + timing spread. Each
        # rep's original cue is marked on its curve, so the leftover scatter of cues
        # around 0 shows the reaction-time jitter the fusion pulled out.
        offsets = self._align_offsets(reps)
        cue_lbl = True
        for ctx, off in zip(reps, offsets):
            y, _ = self._yscale(ctx.envelope, mvc)
            ax_real.plot(ctx.t - off, y, lw=0.9, color=COL_FAINT, alpha=0.7)
            cue = self._onset_offset(ctx)
            yc = float(np.interp(cue, ctx.t, ctx.envelope)) if ctx.t.size else 0.0
            yc = (yc / mvc * 100.0) if (mvc and mvc > 0) else yc
            ax_real.plot([cue - off], [yc], marker="o", ms=5, color=COL_ONSET,
                         alpha=0.85, zorder=5, label="rep cue" if cue_lbl else None)
            cue_lbl = False
        if mvc and mvc > 0:
            ax_real.axhline(100.0, color=COL_MVC, lw=1.0, ls="--", label="100% MVC")
        ax_real.axvline(0.0, color="#777777", lw=1.0, ls=":", label="aligned zero")
        ax_real.legend(loc="upper right", fontsize=8)
        ax_real.set_title(f"{subject} — {category}: {len(reps)} reps aligned (cue + features)")
        ax_real.set_ylabel(unit)
        ax_real.set_xlabel("aligned time (s)  ·  0 = fused reference")
        ax_real.grid(alpha=0.25)

        # Bottom: time-normalised rep profiles + mean ± SD — shape consistency.
        scale = (100.0 / mvc) if (mvc and mvc > 0) else 1.0
        valid = [p * scale for p in profiles if p is not None]
        x = np.linspace(0.0, 1.0, valid[0].size) if valid else np.array([])
        for p in valid:
            ax_norm.plot(x, p, lw=0.8, color=COL_FAINT, alpha=0.6)
        if len(valid) >= 2:
            mat = np.vstack(valid)
            mean, std = mat.mean(axis=0), mat.std(axis=0)
            ax_norm.fill_between(x, mean - std, mean + std, color=COL_MEAN, alpha=0.2,
                                 label="±1 SD")
            ax_norm.plot(x, mean, lw=1.8, color=COL_MEAN, label="mean")
            ax_norm.legend(loc="upper right", fontsize=8)
        ax_norm.set_title("time-normalised reps (consistency)")
        ax_norm.set_ylabel(unit)
        ax_norm.set_xlabel("rep progress (0–1)")
        ax_norm.grid(alpha=0.25)

    def _draw_grid(self, reps: List[DatasetContext],
                   subject: str, category: str, mvc: Optional[float]) -> None:
        n = len(reps)
        ncols = 3 if n > 4 else max(1, n)
        nrows = max(1, math.ceil(n / ncols))
        unit = "%MVC" if (mvc and mvc > 0) else "env"
        self.fig.suptitle(f"{subject} — {category}: {n} reps", fontsize=10)
        offsets = self._align_offsets(reps)
        ax0 = None
        for i, ctx in enumerate(reps):
            ax = self.fig.add_subplot(nrows, ncols, i + 1, sharey=ax0)
            ax0 = ax0 or ax
            t = ctx.t - offsets[i]
            y, _ = self._yscale(ctx.envelope, mvc)
            ax.plot(t, y, lw=0.9, color=COL_ENV)
            s, e = ctx.rep()
            if e > s and e <= t.size:
                ax.axvspan(t[s], t[min(e, t.size) - 1], color=COL_REP, alpha=0.3)
            ax.axvline(0.0, color="#777777", lw=0.8, ls=":")            # aligned zero
            ax.axvline(self._onset_offset(ctx) - offsets[i],            # this rep's cue
                       color=COL_ONSET, lw=0.9, ls="--")
            ax.set_title(f"rep {i + 1}", fontsize=8)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
        self.fig.supylabel(unit, fontsize=9)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.25, left=0.09,
                                 right=0.97, top=0.9, bottom=0.08)

    # ------------------------------------------------------------------ #
    #  Table fills
    # ------------------------------------------------------------------ #
    def _fill_meta(self, pairs: List[Tuple[str, str]]) -> None:
        self.tv_meta.delete(*self.tv_meta.get_children())
        for label, value in pairs:
            self.tv_meta.insert("", "end", values=(label, value))

    def _fill_metrics_from_row(self, row: dict, columns, score_key: str = "") -> None:
        self.tv_metrics.delete(*self.tv_metrics.get_children())
        for col in columns:
            tags = ("score",) if col == score_key else ()
            self.tv_metrics.insert("", "end",
                                   values=(COLUMN_LABELS.get(col, col), _fmt(row.get(col))),
                                   tags=tags)

    def _fill_metrics_rows(self, pairs: List[Tuple[str, str]]) -> None:
        self.tv_metrics.delete(*self.tv_metrics.get_children())
        for label, value in pairs:
            self.tv_metrics.insert("", "end", values=(label, value))

    @staticmethod
    def _meta_value(source: Dict[str, object], key: str) -> str:
        val = source.get(key)
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:g}"
        return str(val)

    # ------------------------------------------------------------------ #
    #  Placeholders / errors
    # ------------------------------------------------------------------ #
    def _message(self, msg: str) -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color="gray",
                transform=ax.transAxes)
        self.canvas.draw_idle()

    def _show_error(self, msg: str) -> None:
        self._message(msg)
        self.tv_meta.delete(*self.tv_meta.get_children())
        self.tv_metrics.delete(*self.tv_metrics.get_children())


def main() -> int:
    Viewer().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
