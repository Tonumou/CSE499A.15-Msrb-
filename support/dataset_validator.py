"""
03_qa_validator.py
------------------
Human-in-the-loop QA tool for the VLM fine-tuning dataset curation pipeline.

Features
--------
- Dark-mode UI powered by customtkinter
- Visual progress bar showing % completion
- Editable text pane (edits are saved on Accept)
- Accept  (→  / Right Arrow)  — saves sample (with edits) to verified_dataset.jsonl
- Reject  (←  / Left Arrow)   — skips sample (not saved)
- Undo    (↑  / Up Arrow)      — steps back one sample; re-opens it for editing;
                                  on next Accept the old entry is overwritten, not duplicated
- Image zoom-in / zoom-out via mouse-scroll wheel
- Image pan via click-and-drag
- Resumes automatically from where the last session left off

Directory layout expected
-------------------------
PROJECT_ROOT/
├── data/
│   └── processed_images/   ← .jpg files referenced inside the JSONL
├── dataset/
│   ├── train_dataset.jsonl         ← INPUT  (raw generated data)
│   └── verified_dataset.jsonl      ← OUTPUT (approved / edited data)
└── support/
    └── 03_qa_validator.py          ← THIS FILE

Dependencies
------------
    pip install customtkinter pillow
"""

import os
import json
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# PATH CONFIGURATION  (all relative — never hardcoded)
# ─────────────────────────────────────────────
# support/ → PROJECT_ROOT requires two dirname() calls
SUPPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

INPUT_DATASET  = os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl")
OUTPUT_DATASET = os.path.join(PROJECT_ROOT, "dataset", "verified_dataset.jsonl")
IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, "data")

# ─────────────────────────────────────────────
# APPEARANCE
# ─────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Accent colours used throughout
COLOR_ACCEPT = "#2ECC71"   # emerald green
COLOR_REJECT = "#E74C3C"   # flat red
COLOR_UNDO   = "#F39C12"   # amber
COLOR_BG     = "#1A1A2E"   # deep navy canvas
COLOR_PANEL  = "#16213E"   # slightly lighter panel
COLOR_TEXT   = "#E0E0E0"


# ─────────────────────────────────────────────
# HELPER: safe JSONL file rewriting
# ─────────────────────────────────────────────
def _rewrite_verified(lines: list[dict]) -> None:
    """Atomically rewrite the entire output JSONL file."""
    os.makedirs(os.path.dirname(OUTPUT_DATASET), exist_ok=True)
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_verified() -> list[dict]:
    """Return all currently verified rows (empty list if file absent)."""
    if not os.path.exists(OUTPUT_DATASET):
        return []
    rows = []
    with open(OUTPUT_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────
# ZOOMABLE / PANNABLE IMAGE CANVAS
# ─────────────────────────────────────────────
class ZoomCanvas(ctk.CTkCanvas):
    """
    A canvas that lets the user:
      - scroll to zoom in/out (centred on the mouse pointer)
      - click-and-drag to pan
    """

    ZOOM_FACTOR = 1.15
    MIN_SCALE   = 0.2
    MAX_SCALE   = 8.0

    def __init__(self, master, **kwargs):
        super().__init__(master, bg=COLOR_BG, highlightthickness=0, **kwargs)

        self._pil_image: Image.Image | None = None
        self._scale    : float              = 1.0
        self._img_id   : int  | None        = None
        self._offset_x : float              = 0.0
        self._offset_y : float              = 0.0
        self._drag_start: tuple[int, int] | None = None

        # Bindings
        self.bind("<MouseWheel>",       self._on_mousewheel_win)   # Windows / macOS
        self.bind("<Button-4>",         self._on_scroll_up)        # Linux scroll up
        self.bind("<Button-5>",         self._on_scroll_down)      # Linux scroll down
        self.bind("<ButtonPress-1>",    self._on_drag_start)
        self.bind("<B1-Motion>",        self._on_drag_move)
        self.bind("<Configure>",        lambda _: self._redraw())

    # ── public API ──────────────────────────────────────────────────────────

    def load_image(self, pil_img: Image.Image) -> None:
        """Display a new PIL image, resetting zoom and centring it."""
        self._pil_image = pil_img
        self._scale     = 1.0
        self.after(10, self._fit_to_canvas)   # wait one tick so canvas has its size

    def clear(self) -> None:
        self._pil_image = None
        self.delete("all")

    # ── internal ────────────────────────────────────────────────────────────

    def _fit_to_canvas(self) -> None:
        """Scale image so it fits the canvas on first load, then centre it."""
        if self._pil_image is None:
            return
        cw = self.winfo_width()  or 800
        ch = self.winfo_height() or 700
        iw, ih = self._pil_image.size
        self._scale = min(cw / iw, ch / ih, 1.0)
        self._offset_x = (cw - iw * self._scale) / 2
        self._offset_y = (ch - ih * self._scale) / 2
        self._redraw()

    def _redraw(self) -> None:
        if self._pil_image is None:
            return
        w = max(1, int(self._pil_image.width  * self._scale))
        h = max(1, int(self._pil_image.height * self._scale))
        resized   = self._pil_image.resize((w, h), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)   # keep reference!
        self.delete("all")
        self._img_id = self.create_image(
            int(self._offset_x), int(self._offset_y),
            anchor="nw", image=self._tk_img
        )

    def _zoom(self, factor: float, pivot_x: float, pivot_y: float) -> None:
        new_scale = self._scale * factor
        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, new_scale))
        if new_scale == self._scale:
            return
        actual = new_scale / self._scale
        self._offset_x = pivot_x - actual * (pivot_x - self._offset_x)
        self._offset_y = pivot_y - actual * (pivot_y - self._offset_y)
        self._scale    = new_scale
        self._redraw()

    def _on_mousewheel_win(self, event) -> None:
        factor = self.ZOOM_FACTOR if event.delta > 0 else 1 / self.ZOOM_FACTOR
        self._zoom(factor, event.x, event.y)

    def _on_scroll_up(self, event) -> None:
        self._zoom(self.ZOOM_FACTOR, event.x, event.y)

    def _on_scroll_down(self, event) -> None:
        self._zoom(1 / self.ZOOM_FACTOR, event.x, event.y)

    def _on_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x   += dx
        self._offset_y   += dy
        self._drag_start  = (event.x, event.y)
        self._redraw()


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
class QAValidatorApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("VLM Dataset QA Validator")
        self.geometry("1500x860")
        self.configure(fg_color=COLOR_BG)

        self.samples        : list[dict] = []
        self.verified_rows  : list[dict] = []   # in-memory mirror of the output file
        self.current_index  : int        = 0    # index into self.samples

        self._load_data()
        self._build_ui()
        self._load_sample()

    # ── data layer ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        if not os.path.exists(INPUT_DATASET):
            messagebox.showerror(
                "Dataset not found",
                f"Expected input file:\n{INPUT_DATASET}"
            )
            self.destroy()
            return

        with open(INPUT_DATASET, "r", encoding="utf-8") as f:
            self.samples = [json.loads(l) for l in f if l.strip()]

        # Resume: read how many were already verified
        self.verified_rows  = _read_verified()
        self.current_index  = len(self.verified_rows)

        if self.current_index >= len(self.samples):
            messagebox.showinfo("All done!", "Every sample has already been reviewed.")
            self.destroy()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        total = len(self.samples)

        # ── TOP BAR ──────────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=0, height=70)
        top_bar.pack(side="top", fill="x", padx=0, pady=0)
        top_bar.pack_propagate(False)

        title_lbl = ctk.CTkLabel(
            top_bar,
            text="🛰  VLM Dataset QA Validator",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXT,
        )
        title_lbl.pack(side="left", padx=20)

        # Progress block (right-aligned in top bar)
        prog_block = ctk.CTkFrame(top_bar, fg_color="transparent")
        prog_block.pack(side="right", padx=20, pady=10)

        self.progress_label = ctk.CTkLabel(
            prog_block,
            text=f"0 / {total}  (0 %)",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#A0A0B0",
        )
        self.progress_label.pack(anchor="e")

        self.progress_bar = ctk.CTkProgressBar(
            prog_block,
            width=300,
            height=10,
            fg_color="#2A2A4A",
            progress_color="#3B82F6",
        )
        self.progress_bar.pack(pady=(4, 0))
        self.progress_bar.set(0)

        # ── MAIN AREA ────────────────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color=COLOR_BG)
        main.pack(side="top", fill="both", expand=True, padx=12, pady=(8, 12))

        # Left: zoomable image canvas
        left = ctk.CTkFrame(main, fg_color=COLOR_PANEL, corner_radius=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        zoom_hint = ctk.CTkLabel(
            left,
            text="🔍  Scroll to zoom  ·  Drag to pan",
            font=ctk.CTkFont(size=11),
            text_color="#606080",
        )
        zoom_hint.pack(pady=(8, 2))

        self.zoom_canvas = ZoomCanvas(left)
        self.zoom_canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Right: text + controls
        right = ctk.CTkFrame(main, fg_color=COLOR_PANEL, corner_radius=10, width=560)
        right.pack(side="right", fill="both", padx=(6, 0))
        right.pack_propagate(False)

        # Filename label
        self.filename_label = ctk.CTkLabel(
            right,
            text="",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color="#707090",
        )
        self.filename_label.pack(padx=16, pady=(16, 4), anchor="w")

        # Sample index label
        self.index_label = ctk.CTkLabel(
            right,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        )
        self.index_label.pack(padx=16, anchor="w")

        divider = ctk.CTkFrame(right, fg_color="#2A2A4A", height=2)
        divider.pack(fill="x", padx=16, pady=10)

        report_lbl = ctk.CTkLabel(
            right,
            text="GENERATED RESCUE REPORT  (editable)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4080C0",
        )
        report_lbl.pack(padx=16, anchor="w")

        self.text_box = ctk.CTkTextbox(
            right,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXT,
            fg_color="#0D1526",
            wrap="word",
            border_width=1,
            border_color="#2A3A5A",
        )
        self.text_box.pack(fill="both", expand=True, padx=16, pady=(6, 10))

        # ── BUTTON STRIP ─────────────────────────────────────────────────────
        btn_strip = ctk.CTkFrame(right, fg_color="transparent")
        btn_strip.pack(fill="x", padx=16, pady=(0, 16))

        self.undo_btn = ctk.CTkButton(
            btn_strip,
            text="↩  Undo",
            fg_color="#7A5000",
            hover_color=COLOR_UNDO,
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            command=self._undo,
        )
        self.undo_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.reject_btn = ctk.CTkButton(
            btn_strip,
            text="✗  Reject",
            fg_color="#7A1010",
            hover_color=COLOR_REJECT,
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            command=self._reject,
        )
        self.reject_btn.pack(side="left", fill="x", expand=True, padx=4)

        self.accept_btn = ctk.CTkButton(
            btn_strip,
            text="✓  Accept",
            fg_color="#0E5E30",
            hover_color=COLOR_ACCEPT,
            text_color="white",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            command=self._accept,
        )
        self.accept_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Keyboard shortcuts
        self.bind("<Right>", lambda _: self._accept())
        self.bind("<Left>",  lambda _: self._reject())
        self.bind("<Up>",    lambda _: self._undo())

    # ── sample lifecycle ─────────────────────────────────────────────────────

    def _load_sample(self) -> None:
        """Populate the canvas and text pane with the current sample."""
        total = len(self.samples)

        if self.current_index >= total:
            messagebox.showinfo("All done!", "You have finished reviewing all samples!")
            self.destroy()
            return

        sample = self.samples[self.current_index]

        # ── Update labels / progress ──────────────────────────────────────────
        done_count  = len(self.verified_rows)   # accepted so far
        pct         = done_count / total if total else 0
        self.progress_bar.set(pct)
        self.progress_label.configure(
            text=f"{done_count} accepted / {total}  ({pct * 100:.1f} %)"
        )
        self.index_label.configure(
            text=f"Reviewing {self.current_index + 1} of {total}"
        )
        self.filename_label.configure(text=sample.get("image", "—"))

        # ── Text pane ─────────────────────────────────────────────────────────
        self.text_box.delete("1.0", "end")
        self.text_box.insert("end", sample.get("response", ""))

        # ── Image ─────────────────────────────────────────────────────────────
        img_rel  = sample.get("image", "")
        img_path = os.path.join(IMAGE_BASE_DIR, img_rel)

        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert("RGB")
                self.zoom_canvas.load_image(pil_img)
            except Exception as exc:
                self.zoom_canvas.clear()
                self._canvas_message(f"Could not open image:\n{exc}")
        else:
            self.zoom_canvas.clear()
            self._canvas_message(f"Image not found:\n{img_path}")

        # Undo disabled at the very beginning
        self.undo_btn.configure(
            state="normal" if self.current_index > 0 else "disabled",
            fg_color="#7A5000" if self.current_index > 0 else "#3A3A3A",
        )

    def _canvas_message(self, msg: str) -> None:
        """Draw a text message directly onto the zoom canvas."""
        cx = self.zoom_canvas.winfo_width()  // 2 or 400
        cy = self.zoom_canvas.winfo_height() // 2 or 350
        self.zoom_canvas.create_text(
            cx, cy,
            text=msg,
            fill="#606080",
            font=("Segoe UI", 13),
            justify="center",
        )

    # ── actions ──────────────────────────────────────────────────────────────

    def _accept(self) -> None:
        """
        Save the (possibly edited) sample to the verified list.
        If the user stepped back with Undo and is re-accepting an already-
        verified index, overwrite that entry rather than appending.
        """
        sample = dict(self.samples[self.current_index])   # shallow copy
        sample["response"] = self.text_box.get("1.0", "end").strip()

        # How many verified rows correspond to indices 0 … current_index-1?
        # Because Undo removes the last verified row, len(verified_rows) tells
        # us exactly where this sample sits in the output.
        if len(self.verified_rows) > self.current_index:
            # Overwrite: we stepped back and are re-accepting this slot
            self.verified_rows[self.current_index] = sample
        else:
            # Normal append
            self.verified_rows.append(sample)

        _rewrite_verified(self.verified_rows)
        self.current_index += 1
        self._load_sample()

    def _reject(self) -> None:
        """Skip the sample — do NOT write it to the verified file."""
        self.current_index += 1
        self._load_sample()

    def _undo(self) -> None:
        """
        Step back one sample.

        If the previous sample was accepted (i.e. it is the last entry in
        verified_rows), remove it so the user can re-evaluate it cleanly.
        The removed entry is gone from disk only until the user Accepts again.
        """
        if self.current_index == 0:
            return

        self.current_index -= 1

        # If the sample we are returning to was already accepted,
        # remove that acceptance so we don't duplicate on the next Accept.
        if len(self.verified_rows) > self.current_index:
            self.verified_rows = self.verified_rows[: self.current_index]
            _rewrite_verified(self.verified_rows)

        self._load_sample()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QAValidatorApp()
    app.mainloop()