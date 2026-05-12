"""
wsi_app.py
==========
PyQt5 GUI for Chameleon-WSI.

Matches the layout and workflow of Chameleon v1 with WSI-specific additions:
- Slide type selector (Biopsy / TMA)
- Tile size selector
- Thumbnail preview (before/after normalization)
- Per-slide progress tracking
- No pre-flight inspector (thumbnails serve as preview)
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QListWidget, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QFileDialog, QProgressBar, QFrame, QSplitter,
    QGroupBox, QScrollArea, QMessageBox, QSizePolicy, QVBoxLayout,
    QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from chameleon_wsi_core import (
    find_wsi_files, open_slide, get_thumbnail, get_slide_info,
    compute_slide_histogram_stats, compute_slide_reinhard_stats,
    compute_batch_average_wsi_cdf, compute_batch_average_wsi_reinhard,
    compute_reference_stain_macenko, compute_reference_stain_vahadane,
    run_wsi_histogram_batch, run_wsi_reinhard_batch,
    run_wsi_macenko_batch, run_wsi_vahadane_batch,
    run_wsi_tile_save_batch,
    write_wsi_log,
    TILE_SIZE
)
from normalizer_core import (
    compute_image_cdf, compute_reinhard_stats,
    apply_histogram_match, apply_reinhard,
    normalize_macenko, normalize_vahadane,
)

# ── Color palette (matches Chameleon v1) ──────────────────────────────────
BG      = '#141921'
PANEL   = '#1e2530'
PANEL2  = '#171e28'
ACCENT  = '#2f85cc'
ACCENT2 = '#38b38d'
TEXT    = '#ebecf2'
DIM     = '#8c96ad'
WARNING = '#f2a533'
SUCCESS = '#40c77a'
DANGER  = '#e65555'
BORDER  = '#2a3344'


def style_sheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: 'Segoe UI', sans-serif;
        font-size: 11px;
    }}
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 8px;
        font-size: 10px;
        font-weight: bold;
        color: {DIM};
        letter-spacing: 1px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    QLineEdit {{
        background-color: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        color: {TEXT};
    }}
    QPushButton {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 5px;
        padding: 5px 12px;
        color: {TEXT};
    }}
    QPushButton:hover {{ background-color: #2a3344; border-color: {ACCENT}; }}
    QPushButton:disabled {{ color: {DIM}; }}
    QPushButton#run_btn {{
        background-color: {ACCENT};
        border: none;
        font-size: 13px;
        font-weight: bold;
        color: white;
        padding: 10px;
        border-radius: 6px;
    }}
    QPushButton#cancel_btn {{
        background-color: {DANGER};
        border: none;
        font-size: 13px;
        color: white;
        padding: 10px;
        border-radius: 6px;
    }}
    QPushButton#quit_btn {{
        background-color: {PANEL2};
        border: 1px solid {BORDER};
        color: {DIM};
        font-size: 9px;
        border-radius: 4px;
    }}
    QListWidget {{
        background-color: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 4px;
        color: {TEXT};
        font-size: 10px;
    }}
    QListWidget::item:selected {{ background-color: {ACCENT}; color: white; }}
    QComboBox {{
        background-color: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 24px 4px 8px;
        color: {TEXT};
    }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {DIM};
    }}
    QComboBox QAbstractItemView {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        color: {TEXT};
        selection-background-color: {ACCENT};
    }}
    QRadioButton {{ color: {TEXT}; font-size: 11px; spacing: 8px; }}
    QRadioButton::indicator {{
        width: 14px; height: 14px; border-radius: 7px;
        border: 2px solid {DIM}; background-color: {PANEL2};
    }}
    QRadioButton::indicator:checked {{
        background-color: {ACCENT}; border-color: {ACCENT};
    }}
    QCheckBox {{ color: {TEXT}; font-size: 11px; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px; border-radius: 3px;
        border: 2px solid {DIM}; background-color: {PANEL2};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT}; border-color: {ACCENT};
    }}
    QProgressBar {{
        background-color: {PANEL2};
        border: 1px solid {BORDER};
        border-radius: 4px;
        height: 8px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}
    QScrollArea {{ border: none; background-color: {PANEL}; }}
    QLabel#title_lbl {{ font-size: 20px; font-weight: bold; color: {ACCENT}; }}
    QLabel#subtitle_lbl {{ font-size: 10px; color: {DIM}; }}
    QLabel#section_lbl {{ font-size: 9px; font-weight: bold; color: {DIM}; letter-spacing: 1px; }}
    QLabel#status_lbl {{ font-size: 10px; color: {DIM}; padding: 0 10px; }}
    QFrame#status_bar {{ background-color: #0d1117; max-height: 28px; min-height: 28px; }}
    """


# ── Image canvas ───────────────────────────────────────────────────────────

class ImageCanvas(FigureCanvas):
    def __init__(self, title='', title_color=DIM, parent=None):
        self.fig = Figure(facecolor=BG, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._title       = title
        self._title_color = title_color
        self._style()

    def _style(self):
        self.ax.set_facecolor(BG)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_edgecolor(BORDER)
        if self._title:
            self.ax.set_title(self._title, color=self._title_color,
                              fontsize=9, pad=4)
        self.fig.patch.set_facecolor(BG)

    def show_image(self, img, title=None, title_color=None):
        self.ax.clear(); self._style()
        if title:
            self.ax.set_title(title,
                              color=title_color or self._title_color,
                              fontsize=9, pad=4)
        if img is not None:
            self.ax.imshow(img)
        self.draw()

    def show_placeholder(self, msg, title=None):
        self.ax.clear(); self._style()
        if title:
            self.ax.set_title(title, color=DIM, fontsize=9, pad=4)
        self.ax.text(0.5, 0.5, msg, transform=self.ax.transAxes,
                     ha='center', va='center', color=DIM, fontsize=9,
                     wrap=True)
        self.draw()

    def clear_canvas(self):
        self.ax.clear(); self._style(); self.draw()


# ── Worker thread ──────────────────────────────────────────────────────────

class WorkerSignals(QObject):
    progress = pyqtSignal(int, int, str)
    log_line = pyqtSignal(str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)


class NormWorker(QThread):
    def __init__(self, mode, wsi_paths, output_dir,
                 slide_type, tile_size, save_log,
                 ref_path=None, n_workers=4, jpeg_quality=80,
                 compression='jpeg', output_format='wsi',
                 input_folder=None):
        super().__init__()
        self.mode          = mode
        self.wsi_paths     = wsi_paths
        self.output_dir    = output_dir
        self.slide_type    = slide_type
        self.tile_size     = tile_size
        self.save_log      = save_log
        self.ref_path      = ref_path
        self.n_workers     = n_workers
        self.jpeg_quality  = jpeg_quality
        self.compression   = compression
        self.output_format = output_format
        self.input_folder  = input_folder
        self._cancel       = False
        self.signals       = WorkerSignals()

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            n = len(self.wsi_paths)

            def prog(i, total, msg=''):
                self.signals.progress.emit(i, total, msg)

            def log_line(line):
                self.signals.log_line.emit(line)

            def cancel_flag():
                return self._cancel

            mode_names = {
                1: 'HistMatch-Reference',
                2: 'HistMatch-BatchAvg',
                3: 'Reinhard-Reference',
                4: 'Reinhard-BatchAvg',
                5: 'Macenko-Reference',
                6: 'Vahadane-Reference',
            }

            # ── Build target / fit reference ───────────────────────────
            if self.mode == 1:
                prog(0, 1, 'Loading reference slide…')
                ref    = open_slide(self.ref_path)
                stats  = compute_slide_histogram_stats(
                    ref, self.slide_type, self.tile_size, prog)
                target = stats['cdf']
                ref.close()

            elif self.mode == 2:
                prog(0, 1, 'Computing batch-average histogram…')
                target = compute_batch_average_wsi_cdf(
                    self.wsi_paths, self.slide_type,
                    self.tile_size, prog)

            elif self.mode == 3:
                prog(0, 1, 'Loading reference slide…')
                ref    = open_slide(self.ref_path)
                stats  = compute_slide_reinhard_stats(
                    ref, self.slide_type, self.tile_size, prog)
                target = {'mu': stats['mu'], 'sigma': stats['sigma']}
                ref.close()

            elif self.mode == 4:
                prog(0, 1, 'Computing batch-average Reinhard statistics…')
                target = compute_batch_average_wsi_reinhard(
                    self.wsi_paths, self.slide_type,
                    self.tile_size, prog)

            elif self.mode == 5:
                prog(0, 1, 'Fitting Macenko stain matrix from reference slide…')
                ref    = open_slide(self.ref_path)
                target = compute_reference_stain_macenko(
                    ref, self.slide_type, self.tile_size, prog)
                ref.close()
                sm = target['stain_matrix']
                log_line(f'Macenko stain matrix fitted from: '
                         f'{Path(self.ref_path).name}')
                log_line(f'  H: R={sm[0,0]:.4f}  G={sm[0,1]:.4f}  B={sm[0,2]:.4f}')
                log_line(f'  E: R={sm[1,0]:.4f}  G={sm[1,1]:.4f}  B={sm[1,2]:.4f}')

            else:  # mode 6
                prog(0, 1, 'Fitting Vahadane stain matrix from reference slide…')
                ref    = open_slide(self.ref_path)
                target = compute_reference_stain_vahadane(
                    ref, self.slide_type, self.tile_size, prog)
                ref.close()
                sm = target['stain_matrix']
                log_line(f'Vahadane stain matrix fitted from: '
                         f'{Path(self.ref_path).name}')
                log_line(f'  H: R={sm[0,0]:.4f}  G={sm[0,1]:.4f}  B={sm[0,2]:.4f}')
                log_line(f'  E: R={sm[1,0]:.4f}  G={sm[1,1]:.4f}  B={sm[1,2]:.4f}')

            # ── Build normalize_fn for tile-save mode ──────────────────
            if self.output_format == 'tiles':
                if self.mode in (1, 2):
                    normalize_fn = lambda img: apply_histogram_match(img, target)
                elif self.mode in (3, 4):
                    normalize_fn = lambda img: apply_reinhard(img, target)
                elif self.mode == 5:
                    normalize_fn = lambda img: normalize_macenko(img, target)
                else:
                    normalize_fn = lambda img: normalize_vahadane(img, target)

                log = run_wsi_tile_save_batch(
                    self.wsi_paths, normalize_fn, self.output_dir,
                    self.slide_type, self.tile_size,
                    prog, cancel_flag)

                for entry in log:
                    if 'error' in entry:
                        log_line(f'[ERROR]  {entry["filename"]}: '
                                 f'{entry["error"]}')
                    else:
                        log_line(
                            f'{entry["filename"]}  —  '
                            f'{entry["tiles_saved"]} tiles saved, '
                            f'{entry["tiles_skipped"]} background tiles skipped')

            else:
                # ── WSI output ─────────────────────────────────────────
                if self.mode in (1, 2):
                    log = run_wsi_histogram_batch(
                        self.wsi_paths, target, self.output_dir,
                        self.slide_type, self.tile_size,
                        self.n_workers, self.jpeg_quality,
                        self.compression, prog, cancel_flag)
                elif self.mode in (3, 4):
                    log = run_wsi_reinhard_batch(
                        self.wsi_paths, target, self.output_dir,
                        self.slide_type, self.tile_size,
                        self.n_workers, self.jpeg_quality,
                        self.compression, prog, cancel_flag)
                elif self.mode == 5:
                    log = run_wsi_macenko_batch(
                        self.wsi_paths, target, self.output_dir,
                        self.slide_type, self.tile_size,
                        self.n_workers, self.jpeg_quality,
                        self.compression, prog, cancel_flag)
                else:
                    log = run_wsi_vahadane_batch(
                        self.wsi_paths, target, self.output_dir,
                        self.slide_type, self.tile_size,
                        self.n_workers, self.jpeg_quality,
                        self.compression, prog, cancel_flag)

                for entry in log:
                    if 'error' in entry:
                        log_line(f'[ERROR]  {entry["filename"]}: '
                                 f'{entry["error"]}')
                    else:
                        log_line(
                            f'{entry["filename"]}  —  '
                            f'{entry.get("tiles_total","?")} tiles  '
                            f'({entry.get("elapsed_s","?")}s)')

            if self.save_log and log:
                write_wsi_log(
                    log,
                    self.output_dir,
                    mode_names[self.mode],
                    ref_path      = self.ref_path,
                    tile_size     = self.tile_size,
                    input_folder  = self.input_folder,
                    output_format = self.output_format,
                )
                log_line('Log files written to output folder.')

            if self._cancel:
                self.signals.finished.emit('Cancelled by user.')
            else:
                if self.output_format == 'tiles':
                    total_saved   = sum(e.get('tiles_saved',   0) for e in log
                                        if 'error' not in e)
                    total_skipped = sum(e.get('tiles_skipped', 0) for e in log
                                        if 'error' not in e)
                    self.signals.finished.emit(
                        f'{n} slide(s) processed  |  '
                        f'{total_saved} tiles saved, '
                        f'{total_skipped} background tiles skipped  '
                        f'→ {self.output_dir}')
                else:
                    self.signals.finished.emit(
                        f'{n} slide(s) normalized → {self.output_dir}')

        except Exception as e:
            self.signals.error.emit(str(e))



# ── Main window ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    MODE_DESCRIPTIONS = [
        'Match each slide\'s histogram distribution to a single chosen reference slide. Best when you have a high-quality reference with ideal staining.',
        'Build a theoretical mean histogram across all slides in the batch, then match every slide to that population average. No reference slide needed.',
        'Transfer LAB color statistics from a reference slide to each source slide. More conservative than histogram matching; lower artefact risk.',
        'Compute mean LAB statistics across the entire batch to build a bias-free synthetic reference, then apply Reinhard normalization. No reference slide needed.',
        'Estimate H&E stain vectors from a reference slide using SVD (Macenko method), then transfer those vectors to all slides. Recommended for H&E tissue.',
        'Estimate H&E stain vectors from a reference slide using sparse NMF (Vahadane method). Structure-preserving; highest quality but slower than Macenko.',
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Chameleon-WSI  v1.0')
        self.setMinimumSize(1300, 840)
        self.setStyleSheet(style_sheet())
        self._wsi_paths = []
        self._worker    = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_vl = QVBoxLayout(central)
        root_vl.setSpacing(0)
        root_vl.setContentsMargins(0, 0, 0, 0)

        root_vl.addWidget(self._make_header())

        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(2)
        body.setStyleSheet(
            f'QSplitter::handle {{ background-color: {BORDER}; }}')
        body.addWidget(self._make_left_panel())
        body.addWidget(self._make_right_panel())
        body.setSizes([440, 860])
        root_vl.addWidget(body)

        root_vl.addWidget(self._make_status_bar())

    def _make_header(self):
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(
            f'background-color: {PANEL}; border-bottom: 1px solid {BORDER};')
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)

        vl = QVBoxLayout()
        title = QLabel('Chameleon-WSI')
        title.setObjectName('title_lbl')
        sub = QLabel(
            'Whole-slide image batch stain normalization  |  '
            'H&E and IHC  |  SVS · NDPI · SCN · MRXS · Pyramidal TIFF')
        sub.setObjectName('subtitle_lbl')
        vl.addWidget(title)
        vl.addWidget(sub)
        hl.addLayout(vl)
        hl.addStretch()
        return header

    def _make_status_bar(self):
        bar = QFrame()
        bar.setObjectName('status_bar')
        bar.setStyleSheet(
            f'background-color: #0d1117; border-top: 1px solid {BORDER};')
        bl = QHBoxLayout(bar)
        self.status_lbl = QLabel(
            'Ready  –  select a mode, choose slide type, and load WSI files.')
        self.status_lbl.setObjectName('status_lbl')
        bl.addWidget(self.status_lbl)
        return bar

    def _make_left_panel(self):
        outer = QWidget()
        outer.setFixedWidth(440)
        outer.setStyleSheet(f'background-color: {PANEL};')
        outer_vl = QVBoxLayout(outer)
        outer_vl.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f'background-color: {PANEL}; border: none;')

        panel = QWidget()
        panel.setStyleSheet(f'background-color: {PANEL};')
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(10)

        # ── Mode selection ─────────────────────────────────────────────
        mode_group = QGroupBox('NORMALIZATION MODE')
        mg_vl = QVBoxLayout(mode_group)
        mg_vl.setSpacing(6)
        mg_vl.setContentsMargins(10, 14, 10, 10)

        self.mode_group = QButtonGroup()
        self.mode_radios = []
        mode_texts = [
            '1 – Histogram matching  →  reference slide',
            '2 – Histogram matching  →  batch-average CDF',
            '3 – Reinhard  →  reference slide',
            '4 – Reinhard  →  batch-average synthetic reference',
            '5 – Macenko  →  reference slide',
            '6 – Vahadane  →  reference slide',
        ]
        for i, text in enumerate(mode_texts):
            rb = QRadioButton(text)
            if i == 0:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i + 1)
            mg_vl.addWidget(rb)
            self.mode_radios.append(rb)
        self.mode_group.buttonToggled.connect(self._on_mode_changed)

        self.mode_desc = QLabel()
        self.mode_desc.setWordWrap(True)
        self.mode_desc.setStyleSheet(
            f'background-color: {PANEL2}; border: 1px solid {BORDER}; '
            f'border-radius: 4px; padding: 6px; color: {DIM}; font-size: 10px;')
        self.mode_desc.setMinimumHeight(52)
        mg_vl.addWidget(self.mode_desc)
        vl.addWidget(mode_group)

        # ── Slide type ─────────────────────────────────────────────────
        slide_group = QGroupBox('SLIDE TYPE')
        sg_vl = QVBoxLayout(slide_group)
        sg_vl.setContentsMargins(10, 14, 10, 10)
        sg_vl.setSpacing(6)

        self.slide_type_group = QButtonGroup()
        self.biopsy_radio = QRadioButton(
            'Biopsy / Resection  —  center-outward sampling, background exclusion')
        self.tma_radio    = QRadioButton(
            'Tissue Microarray (TMA)  —  full-grid sampling, tissue core selection')
        self.biopsy_radio.setChecked(True)
        self.slide_type_group.addButton(self.biopsy_radio, 0)
        self.slide_type_group.addButton(self.tma_radio,    1)
        sg_vl.addWidget(self.biopsy_radio)
        sg_vl.addWidget(self.tma_radio)

        # Slide type description
        self.slide_desc = QLabel(
            'Center-outward grid sampling. Low-variance tiles (background glass) '
            'are excluded via IQR outlier detection. Statistics computed from '
            'valid tissue tiles only.')
        self.slide_desc.setWordWrap(True)
        self.slide_desc.setStyleSheet(
            f'background-color: {PANEL2}; border: 1px solid {BORDER}; '
            f'border-radius: 4px; padding: 6px; color: {DIM}; font-size: 10px;')
        self.slide_desc.setFixedHeight(52)
        sg_vl.addWidget(self.slide_desc)
        self.slide_type_group.buttonToggled.connect(self._on_slide_type_changed)
        vl.addWidget(slide_group)

        # ── Input folder ───────────────────────────────────────────────
        input_group = QGroupBox('INPUT FOLDER')
        ig_vl = QVBoxLayout(input_group)
        ig_vl.setContentsMargins(10, 14, 10, 10)
        ig_hl = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText('Select WSI input folder…')
        browse_input = QPushButton('…')
        browse_input.setFixedWidth(36)
        browse_input.clicked.connect(self._browse_input)
        ig_hl.addWidget(self.input_field)
        ig_hl.addWidget(browse_input)
        ig_vl.addLayout(ig_hl)
        vl.addWidget(input_group)

        # ── Reference slide ────────────────────────────────────────────
        self.ref_group = QGroupBox('REFERENCE SLIDE  (modes 1, 3, 5 & 6 only)')
        rg_vl = QVBoxLayout(self.ref_group)
        rg_vl.setContentsMargins(10, 14, 10, 10)
        rg_hl = QHBoxLayout()
        self.ref_field = QLineEdit()
        self.ref_field.setPlaceholderText('Select a reference WSI…')
        browse_ref = QPushButton('…')
        browse_ref.setFixedWidth(36)
        browse_ref.clicked.connect(self._browse_ref)
        rg_hl.addWidget(self.ref_field)
        rg_hl.addWidget(browse_ref)
        rg_vl.addLayout(rg_hl)
        vl.addWidget(self.ref_group)

        # ── Output folder ──────────────────────────────────────────────
        output_group = QGroupBox('OUTPUT FOLDER')
        og_vl = QVBoxLayout(output_group)
        og_vl.setContentsMargins(10, 14, 10, 10)
        og_hl = QHBoxLayout()
        self.output_field = QLineEdit()
        self.output_field.setPlaceholderText('Select output folder…')
        browse_output = QPushButton('…')
        browse_output.setFixedWidth(36)
        browse_output.clicked.connect(self._browse_output)
        og_hl.addWidget(self.output_field)
        og_hl.addWidget(browse_output)
        og_vl.addLayout(og_hl)
        vl.addWidget(output_group)

        # ── Processing options ─────────────────────────────────────────
        opt_group = QGroupBox('PROCESSING OPTIONS')
        opt_vl = QVBoxLayout(opt_group)
        opt_vl.setContentsMargins(10, 14, 10, 10)
        opt_vl.setSpacing(8)

        # Tile size
        tile_box = QGroupBox('Tile Size')
        tile_box.setStyleSheet(
            f'QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; '
            f'margin-top: 8px; padding-top: 6px; font-size: 9px; color: {DIM}; }}'
            f'QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 4px; }}')
        tile_hl = QHBoxLayout(tile_box)
        tile_hl.setContentsMargins(8, 8, 8, 8)
        tile_lbl = QLabel('Tile size (px):')
        tile_lbl.setStyleSheet(f'color: {TEXT};')
        self.tile_combo = QComboBox()
        self.tile_combo.addItems(['256', '512', '1024'])
        self.tile_combo.setCurrentText('512')
        self.tile_combo.setFixedWidth(90)
        tile_hint = QLabel('512 recommended for most slides')
        tile_hint.setStyleSheet(f'color: {DIM}; font-size: 9px;')
        tile_hl.addWidget(tile_lbl)
        tile_hl.addWidget(self.tile_combo)
        tile_hl.addSpacing(8)
        tile_hl.addWidget(tile_hint)
        tile_hl.addStretch()
        opt_vl.addWidget(tile_box)

        # Workers
        import os as _os
        cpu_count = _os.cpu_count() or 4
        workers_box = QGroupBox('Parallel Workers')
        workers_box.setStyleSheet(
            f'QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; '
            f'margin-top: 8px; padding-top: 6px; font-size: 9px; color: {DIM}; }}'
            f'QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 4px; }}')
        workers_hl = QHBoxLayout(workers_box)
        workers_hl.setContentsMargins(8, 8, 8, 8)
        workers_lbl = QLabel('Workers:')
        workers_lbl.setStyleSheet(f'color: {TEXT};')
        self.workers_combo = QComboBox()
        for w in [1, 2, 4, 8]:
            if w <= cpu_count:
                self.workers_combo.addItem(str(w))
        default = max(2, cpu_count // 2)
        idx = self.workers_combo.findText(str(default))
        if idx >= 0:
            self.workers_combo.setCurrentIndex(idx)
        self.workers_combo.setFixedWidth(75)
        cpu_lbl = QLabel(f'CPU cores available: {cpu_count}')
        cpu_lbl.setStyleSheet(f'color: {DIM}; font-size: 9px;')
        workers_hl.addWidget(workers_lbl)
        workers_hl.addWidget(self.workers_combo)
        workers_hl.addSpacing(8)
        workers_hl.addWidget(cpu_lbl)
        workers_hl.addStretch()
        opt_vl.addWidget(workers_box)

        # Compression
        comp_box = QGroupBox('Output Compression')
        comp_box.setStyleSheet(
            f'QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; '
            f'margin-top: 8px; padding-top: 6px; font-size: 9px; color: {DIM}; }}'
            f'QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 4px; }}')
        comp_vl = QVBoxLayout(comp_box)
        comp_vl.setContentsMargins(8, 8, 8, 8)
        comp_vl.setSpacing(6)

        comp_hl = QHBoxLayout()
        comp_lbl = QLabel('Format:')
        comp_lbl.setStyleSheet(f'color: {TEXT};')
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(['JPEG (smaller files)', 'Deflate (lossless, no artefacts)'])
        self.compression_combo.setCurrentIndex(0)
        self.compression_combo.currentIndexChanged.connect(self._on_compression_changed)
        comp_hl.addWidget(comp_lbl)
        comp_hl.addWidget(self.compression_combo)
        comp_hl.addStretch()
        comp_vl.addLayout(comp_hl)

        self.jpeg_hl_widget = QWidget()
        jpeg_hl = QHBoxLayout(self.jpeg_hl_widget)
        jpeg_hl.setContentsMargins(0, 0, 0, 0)
        jpeg_lbl = QLabel('JPEG quality:')
        jpeg_lbl.setStyleSheet(f'color: {TEXT};')
        self.jpeg_combo = QComboBox()
        self.jpeg_combo.addItems(['60', '70', '80', '90', '95'])
        self.jpeg_combo.setCurrentText('80')
        self.jpeg_combo.setFixedWidth(75)
        jpeg_hint = QLabel('80 recommended')
        jpeg_hint.setStyleSheet(f'color: {DIM}; font-size: 9px;')
        jpeg_hl.addWidget(jpeg_lbl)
        jpeg_hl.addWidget(self.jpeg_combo)
        jpeg_hl.addSpacing(8)
        jpeg_hl.addWidget(jpeg_hint)
        jpeg_hl.addStretch()
        comp_vl.addWidget(self.jpeg_hl_widget)

        opt_vl.addWidget(comp_box)


        # Output format
        fmt_box = QGroupBox('Output Format')
        fmt_box.setStyleSheet(
            f'QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; '
            f'margin-top: 8px; padding-top: 6px; font-size: 9px; color: {DIM}; }}'
            f'QGroupBox::title {{ subcontrol-origin: margin; left: 6px; padding: 0 4px; }}')
        fmt_vl = QVBoxLayout(fmt_box)
        fmt_vl.setContentsMargins(8, 8, 8, 8)
        fmt_vl.setSpacing(6)

        self.fmt_group = QButtonGroup()
        self.fmt_wsi_radio   = QRadioButton('Normalized WSI  (pyramidal TIFF)')
        self.fmt_tiles_radio = QRadioButton('Save tiles  (PNG, tissue only)')
        self.fmt_wsi_radio.setChecked(True)
        self.fmt_group.addButton(self.fmt_wsi_radio,   0)
        self.fmt_group.addButton(self.fmt_tiles_radio, 1)
        fmt_vl.addWidget(self.fmt_wsi_radio)
        fmt_vl.addWidget(self.fmt_tiles_radio)

        self.tiles_info_lbl = QLabel(
            'Background tiles are excluded automatically. '
            'A count of saved and skipped tiles per slide will appear in '
            'the log panel and be recorded in the log files.')
        self.tiles_info_lbl.setWordWrap(True)
        self.tiles_info_lbl.setStyleSheet(
            f'color: {WARNING}; font-size: 9px;')
        self.tiles_info_lbl.setVisible(False)
        fmt_vl.addWidget(self.tiles_info_lbl)

        self.fmt_group.buttonToggled.connect(self._on_format_changed)
        opt_vl.addWidget(fmt_box)

        self.log_check = QCheckBox('Save processing log  (TXT + CSV)')
        self.log_check.setChecked(True)
        opt_vl.addWidget(self.log_check)
        vl.addWidget(opt_group)
        # ── WSI queue ──────────────────────────────────────────────────
        queue_group = QGroupBox('WSI QUEUE')
        qg_vl = QVBoxLayout(queue_group)
        qg_vl.setContentsMargins(10, 14, 10, 10)
        qg_vl.setSpacing(6)

        q_btns = QHBoxLayout()
        reload_btn = QPushButton('Reload')
        reload_btn.setFixedWidth(70)
        reload_btn.clicked.connect(self._reload_files)
        clear_btn = QPushButton('Clear')
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear_files)
        self.count_lbl = QLabel('0 slides loaded')
        self.count_lbl.setStyleSheet(f'color: {DIM}; font-size: 10px;')
        q_btns.addWidget(reload_btn)
        q_btns.addWidget(clear_btn)
        q_btns.addStretch()
        q_btns.addWidget(self.count_lbl)
        qg_vl.addLayout(q_btns)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(80)
        self.file_list.currentRowChanged.connect(
            self._on_slide_selected)
        qg_vl.addWidget(self.file_list)
        vl.addWidget(queue_group)

        vl.addStretch()

        # ── Progress ───────────────────────────────────────────────────
        self.progress_lbl = QLabel('Idle')
        self.progress_lbl.setStyleSheet(f'color: {DIM}; font-size: 10px;')
        vl.addWidget(self.progress_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        vl.addWidget(self.progress_bar)

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton('▶  Run Normalization')
        self.run_btn.setObjectName('run_btn')
        self.run_btn.setFixedHeight(42)
        self.run_btn.clicked.connect(self._run)

        self.cancel_btn = QPushButton('✕  Cancel')
        self.cancel_btn.setObjectName('cancel_btn')
        self.cancel_btn.setFixedHeight(42)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        vl.addLayout(btn_row)

        self.quit_btn = QPushButton('⏻  Quit Chameleon-WSI')
        self.quit_btn.setObjectName('quit_btn')
        self.quit_btn.setFixedHeight(26)
        self.quit_btn.clicked.connect(self.close)
        vl.addWidget(self.quit_btn)

        self._on_mode_changed()

        scroll.setWidget(panel)
        outer_vl.addWidget(scroll)
        return outer

    def _make_right_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f'background-color: {BG};')
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        preview_lbl = QLabel('SLIDE PREVIEW')
        preview_lbl.setObjectName('section_lbl')
        vl.addWidget(preview_lbl)

        # Slide info bar
        self.slide_info_lbl = QLabel('Select a slide from the queue to preview')
        self.slide_info_lbl.setStyleSheet(
            f'color: {DIM}; font-size: 10px; padding: 4px;')
        vl.addWidget(self.slide_info_lbl)

        # Thumbnail row: original + normalized
        thumb_row = QWidget()
        tr_hl = QHBoxLayout(thumb_row)
        tr_hl.setSpacing(8)

        orig_col = QWidget()
        oc_vl = QVBoxLayout(orig_col)
        oc_vl.setSpacing(2)
        orig_lbl = QLabel('ORIGINAL THUMBNAIL')
        orig_lbl.setAlignment(Qt.AlignCenter)
        orig_lbl.setStyleSheet(
            f'color: {DIM}; font-size: 9px; font-weight: bold;')
        self.canvas_orig = ImageCanvas(title_color=DIM)
        self.canvas_orig.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        oc_vl.addWidget(orig_lbl)
        oc_vl.addWidget(self.canvas_orig)

        norm_col = QWidget()
        nc_vl = QVBoxLayout(norm_col)
        nc_vl.setSpacing(2)
        norm_lbl = QLabel('NORMALIZED THUMBNAIL  (preview)')
        norm_lbl.setAlignment(Qt.AlignCenter)
        norm_lbl.setStyleSheet(
            f'color: {ACCENT}; font-size: 9px; font-weight: bold;')
        self.canvas_norm = ImageCanvas(title_color=ACCENT)
        self.canvas_norm.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        nc_vl.addWidget(norm_lbl)
        nc_vl.addWidget(self.canvas_norm)

        tr_hl.addWidget(orig_col)
        tr_hl.addWidget(norm_col)
        vl.addWidget(thumb_row, stretch=3)

        # Slide statistics panel
        stats_group = QGroupBox('SLIDE STATISTICS')
        sg_vl = QVBoxLayout(stats_group)
        sg_vl.setContentsMargins(10, 14, 10, 10)

        self.stats_lbl = QLabel('–')
        self.stats_lbl.setStyleSheet(
            f'color: {DIM}; font-size: 10px; font-family: monospace;')
        self.stats_lbl.setWordWrap(True)
        sg_vl.addWidget(self.stats_lbl)

        vl.addWidget(stats_group, stretch=1)

        # Processing log panel
        log_group = QGroupBox('PROCESSING LOG')
        lg_vl = QVBoxLayout(log_group)
        lg_vl.setContentsMargins(10, 14, 10, 10)

        self.log_text = QListWidget()
        self.log_text.setStyleSheet(
            f'background-color: {PANEL2}; border: 1px solid {BORDER}; '
            f'border-radius: 4px; color: {DIM}; font-size: 9px; '
            f'font-family: monospace;')
        self.log_text.setMinimumHeight(100)
        self.log_text.setMaximumHeight(160)
        lg_vl.addWidget(self.log_text)

        vl.addWidget(log_group, stretch=1)

        return panel

    # ── Mode / slide type callbacks ────────────────────────────────────

    def _on_compression_changed(self, idx):
        self.jpeg_hl_widget.setVisible(idx == 0)

    def _on_format_changed(self, *_):
        is_tiles = self.fmt_tiles_radio.isChecked()
        self.tiles_info_lbl.setVisible(is_tiles)
        # Compression options are irrelevant for PNG tile output
        self.jpeg_hl_widget.setVisible(
            not is_tiles and self.compression_combo.currentIndex() == 0)

    def _on_mode_changed(self, *_):
        mode = self.mode_group.checkedId()
        if mode < 1:
            return
        self.mode_desc.setText(self.MODE_DESCRIPTIONS[mode - 1])
        needs_ref = mode in (1, 3, 5, 6)
        self.ref_group.setEnabled(needs_ref)

    def _on_slide_type_changed(self, *_):
        if self.biopsy_radio.isChecked():
            self.slide_desc.setText(
                'Center-outward grid sampling. Low-variance tiles (background '
                'glass) are excluded via IQR outlier detection. Statistics '
                'computed from valid tissue tiles only.')
        else:
            self.slide_desc.setText(
                'Dense full-slide grid sampling. High-variance tiles (tissue '
                'cores) are selected via top-quartile standard deviation filter. '
                'Background glass is automatically discarded.')

    # ── File handling ──────────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select WSI Input Folder')
        if folder:
            self.input_field.setText(folder)
            self._load_folder(folder)

    def _browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Reference WSI', '',
            'WSI Files (*.svs *.ndpi *.scn *.mrxs *.tif *.tiff *.bif)')
        if path:
            self.ref_field.setText(path)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Output Folder')
        if folder:
            self.output_field.setText(folder)

    def _load_folder(self, folder):
        files = find_wsi_files(folder)
        if not files:
            QMessageBox.warning(
                self, 'No WSI Files',
                f'No supported WSI files found in:\n{folder}\n\n'
                'Supported: SVS, NDPI, SCN, MRXS, TIF/TIFF, BIF')
            return
        self._wsi_paths = files
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(Path(f).name)
        self.count_lbl.setText(f'{len(files)} slide(s) loaded')
        self._set_status(f'Loaded {len(files)} WSI files.', SUCCESS)

    def _reload_files(self):
        folder = self.input_field.text()
        if folder and os.path.isdir(folder):
            self._load_folder(folder)

    def _clear_files(self):
        self._wsi_paths = []
        self.file_list.clear()
        self.count_lbl.setText('0 slides loaded')
        self.canvas_orig.clear_canvas()
        self.canvas_norm.clear_canvas()
        self.stats_lbl.setText('–')
        self._set_status('File list cleared.', DIM)

    def _on_slide_selected(self, row):
        if row < 0 or row >= len(self._wsi_paths):
            return
        path = self._wsi_paths[row]
        try:
            slide = open_slide(path)
            info  = get_slide_info(slide)
            thumb = get_thumbnail(slide, max_size=1024)
            slide.close()

            self.canvas_orig.show_image(thumb, 'Original', DIM)

            # If a normalized output already exists for this slide show it.
            # Otherwise show the standard placeholder.
            output_dir = self.output_field.text()
            out_path   = (Path(output_dir) / f'{Path(path).stem}_norm.tiff'
                          if output_dir else None)
            if out_path and out_path.is_file():
                try:
                    from PIL import Image as _PIL
                    _PIL.MAX_IMAGE_PIXELS = None   # disable decompression bomb limit
                    nslide = open_slide(str(out_path))
                    nthumb = get_thumbnail(nslide, max_size=1024)
                    nslide.close()
                    self.canvas_norm.show_image(nthumb, 'Normalized', ACCENT)
                except Exception:
                    self.canvas_norm.show_placeholder(
                        'Run normalization to see preview\n'
                        '(thumbnail will update after processing)')
            else:
                self.canvas_norm.show_placeholder(
                    'Run normalization to see preview\n'
                    '(thumbnail will update after processing)')

            self.slide_info_lbl.setText(
                f'{Path(path).name}  |  '
                f'{info["width"]:,} × {info["height"]:,} px  |  '
                f'{info["levels"]} pyramid levels  |  '
                f'MPP: {info["mpp"]}')

            self.stats_lbl.setText(
                f'Dimensions:  {info["width"]:,} × {info["height"]:,} pixels\n'
                f'Pyramid levels:  {info["levels"]}\n'
                f'Microns per pixel:  {info["mpp"]}\n'
                f'Approx. tiles (512px):  '
                f'{int(np.ceil(info["width"]/512)) * int(np.ceil(info["height"]/512)):,}')

        except Exception as e:
            self.canvas_orig.show_placeholder(f'Could not read slide:\n{e}')
            self.slide_info_lbl.setText(f'Error reading {Path(path).name}')

    # ── Validate ───────────────────────────────────────────────────────

    def _validate(self) -> bool:
        if not self._wsi_paths:
            QMessageBox.warning(self, 'No Slides',
                                'Please load WSI files first.')
            return False
        mode = self.mode_group.checkedId()
        if mode in (1, 3, 5, 6):
            ref = self.ref_field.text()
            if not ref or not Path(ref).is_file():
                QMessageBox.warning(self, 'Missing Reference',
                                    'Please select a valid reference slide.')
                return False
        if not self.output_field.text():
            QMessageBox.warning(self, 'No Output Folder',
                                'Please specify an output folder.')
            return False
        os.makedirs(self.output_field.text(), exist_ok=True)
        return True

    # ── Run / Cancel ───────────────────────────────────────────────────

    def _run(self):
        if not self._validate():
            return

        self.log_text.clear()

        slide_type    = 'tma' if self.tma_radio.isChecked() else 'biopsy'
        mode          = self.mode_group.checkedId()
        output_format = 'tiles' if self.fmt_tiles_radio.isChecked() else 'wsi'
        compression   = 'deflate' if self.compression_combo.currentIndex() == 1 else 'jpeg'

        self._worker = NormWorker(
            mode          = mode,
            wsi_paths     = self._wsi_paths,
            output_dir    = self.output_field.text(),
            slide_type    = slide_type,
            tile_size     = int(self.tile_combo.currentText()),
            save_log      = self.log_check.isChecked(),
            ref_path      = self.ref_field.text() or None,
            n_workers     = int(self.workers_combo.currentText()),
            jpeg_quality  = int(self.jpeg_combo.currentText()),
            compression   = compression,
            output_format = output_format,
            input_folder  = self.input_field.text(),
        )
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.log_line.connect(self._on_log_line)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.error.connect(self._on_error)

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._set_status('Cancelling…', WARNING)

    def _on_progress(self, current, total, msg):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.progress_lbl.setText(msg)
        self._set_status(msg, ACCENT)

    def _on_log_line(self, line):
        self.log_text.addItem(line)
        self.log_text.scrollToBottom()

    def _on_finished(self, msg):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_lbl.setText('Complete')
        self._set_status(msg, SUCCESS)

        # Update the normalized thumbnail for the currently selected slide.
        # Only applies to WSI output mode — tile-save mode produces no single
        # output file to thumbnail.
        if self.fmt_wsi_radio.isChecked():
            self._update_norm_thumbnail()

        QMessageBox.information(self, 'Complete', msg)

    def _update_norm_thumbnail(self):
        """
        Load a thumbnail from the normalized output file for the currently
        selected slide and display it in canvas_norm.

        Called after processing completes.  Silently shows a placeholder if
        the output file cannot be found or read.
        """
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self._wsi_paths):
            return

        src_path   = self._wsi_paths[row]
        output_dir = self.output_field.text()
        stem       = Path(src_path).stem
        out_path   = Path(output_dir) / f'{stem}_norm.tiff'

        if not out_path.is_file():
            self.canvas_norm.show_placeholder(
                'Normalized output not found.\n'
                f'Expected: {out_path.name}')
            return

        try:
            from PIL import Image as _PIL
            _PIL.MAX_IMAGE_PIXELS = None   # disable decompression bomb limit
            slide = open_slide(str(out_path))
            thumb = get_thumbnail(slide, max_size=1024)
            slide.close()
            self.canvas_norm.show_image(thumb, 'Normalized', ACCENT)
        except Exception as e:
            self.canvas_norm.show_placeholder(
                f'Could not load normalized thumbnail:\n{e}')

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_status(f'Error: {msg}', DANGER)
        QMessageBox.critical(self, 'Error', msg)

    def _set_status(self, msg, color=DIM):
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(
            f'color: {color}; font-size: 10px; padding: 0 10px;')

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        event.accept()
        QApplication.quit()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Chameleon-WSI')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == '__main__':
    main()
