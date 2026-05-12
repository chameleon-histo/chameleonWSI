"""
wsi_core.py
===========
Core WSI handling for Chameleon-WSI.

Handles:
- OpenSlide-based reading of SVS, NDPI, SCN, MRXS, pyramidal TIFF
- Tile-based sampling with outlier exclusion
- Biopsy (center-outward) and TMA (full-grid positive selection) modes
- Slide-level statistics computation
- Two-pass tile normalization pipeline
- Pyramidal TIFF output via tifffile

All normalization algorithms are imported from normalizer_core.py (unchanged
from Chameleon v1). The WSI layer handles sampling and I/O only.
"""

import numpy as np
import os
import time
import csv
import datetime
import threading
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openslide
from normalizer_core import (
    apply_histogram_match,
    apply_reinhard,
    compute_image_cdf,
    compute_batch_average_cdf,
    compute_batch_average_reinhard_stats,
    compute_reinhard_stats,
    fast_rgb2lab,
    _wasserstein_dist,
    img_as_ubyte,
    fit_macenko,
    normalize_macenko,
    fit_vahadane,
    normalize_vahadane,
)

# ── Constants ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    '.svs', '.ndpi', '.scn', '.mrxs', '.vms', '.vmu',
    '.tif', '.tiff', '.bif'
}

TILE_SIZE          = 512      # default tile size in pixels
MIN_VALID_TILES    = 5        # minimum tiles needed for statistics
IQR_MULTIPLIER     = 1.5      # standard IQR fence multiplier
BIOPSY_GRID        = 7        # n×n grid for biopsy sampling
TMA_GRID           = 12       # n×n grid for TMA sampling (denser)
TMA_TOP_PERCENTILE = 75       # keep top N% by variance for TMA


# ── File discovery ─────────────────────────────────────────────────────────

def find_wsi_files(folder: str) -> list:
    """Return sorted list of supported WSI paths in folder."""
    folder = Path(folder)
    files  = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(f))
    return files


# ── OpenSlide helpers ──────────────────────────────────────────────────────

def open_slide(path: str) -> openslide.OpenSlide:
    """Open a WSI file and return an OpenSlide object."""
    return openslide.OpenSlide(path)


def get_slide_info(slide: openslide.OpenSlide) -> dict:
    """Return basic metadata about the slide."""
    dims   = slide.dimensions          # (width, height) at level 0
    levels = slide.level_count
    mpp    = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_X, 'unknown')
    return {
        'width':      dims[0],
        'height':     dims[1],
        'levels':     levels,
        'mpp':        mpp,
        'dimensions': dims,
    }


def get_thumbnail(slide: openslide.OpenSlide,
                  max_size: int = 1024) -> np.ndarray:
    """
    Return a thumbnail as uint8 RGB numpy array.
    Longest dimension is capped at max_size.
    """
    w, h  = slide.dimensions
    scale = max_size / max(w, h)
    thumb = slide.get_thumbnail((int(w * scale), int(h * scale)))
    arr   = np.array(thumb.convert('RGB'), dtype=np.uint8)
    return arr


def read_tile(slide: openslide.OpenSlide,
              x: int, y: int,
              tile_size: int = TILE_SIZE,
              level: int = 0) -> np.ndarray:
    """
    Read a single tile from the slide at level 0.
    Returns uint8 RGB numpy array of shape (tile_size, tile_size, 3).
    Handles edge tiles by padding with white.
    """
    region = slide.read_region((x, y), level, (tile_size, tile_size))
    arr    = np.array(region.convert('RGB'), dtype=np.uint8)
    return arr


# ── Tile scoring ───────────────────────────────────────────────────────────

def score_tile(tile: np.ndarray) -> float:
    """
    Compute a tissue score for a tile based on pixel standard deviation.
    High score = tissue. Low score = background glass.

    Background glass is nearly pure white with very low variance.
    Tissue has broader intensity distribution and higher variance.
    """
    return float(tile.std())


def is_mostly_white(tile: np.ndarray,
                    white_threshold: int = 230,
                    white_fraction: float = 0.90) -> bool:
    """
    Quick check: return True if >90% of pixels are near-white.
    Used as a fast pre-filter before IQR outlier detection.
    """
    mean_per_pixel = tile.mean(axis=2)   # (H, W)
    white_pixels   = (mean_per_pixel > white_threshold).sum()
    total_pixels   = mean_per_pixel.size
    return (white_pixels / total_pixels) >= white_fraction


# ── Grid generation ────────────────────────────────────────────────────────

def biopsy_grid_positions(slide_width: int, slide_height: int,
                           n: int = BIOPSY_GRID,
                           tile_size: int = TILE_SIZE) -> list:
    """
    Generate n×n grid positions ordered from center outward.
    Returns list of (x, y) top-left tile coordinates.
    """
    # Evenly spaced grid across the slide
    xs = np.linspace(tile_size, slide_width  - tile_size * 2, n, dtype=int)
    ys = np.linspace(tile_size, slide_height - tile_size * 2, n, dtype=int)

    # Create all grid positions
    positions = [(int(x), int(y)) for x in xs for y in ys]

    # Sort by distance from center
    cx = slide_width  / 2
    cy = slide_height / 2
    positions.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    return positions


def tma_grid_positions(slide_width: int, slide_height: int,
                        n: int = TMA_GRID,
                        tile_size: int = TILE_SIZE) -> list:
    """
    Generate dense n×n grid positions spread across the full slide.
    For TMA slides where tissue cores are distributed uniformly.
    Returns list of (x, y) top-left tile coordinates.
    """
    xs = np.linspace(tile_size, slide_width  - tile_size * 2, n, dtype=int)
    ys = np.linspace(tile_size, slide_height - tile_size * 2, n, dtype=int)
    return [(int(x), int(y)) for x in xs for y in ys]


# ── Outlier exclusion ──────────────────────────────────────────────────────

def filter_biopsy_tiles(tiles_and_scores: list) -> list:
    """
    For biopsy slides: exclude low-variance tiles (background glass).

    Uses IQR method — tiles below Q1 - 1.5×IQR are flagged as outliers.
    Also applies a fast white-pixel pre-filter.

    Returns list of (tile, score, x, y) for valid tiles only.
    """
    if not tiles_and_scores:
        return []

    scores = np.array([s for _, s, _, _ in tiles_and_scores])

    Q1, Q3 = np.percentile(scores, [25, 75])
    IQR    = Q3 - Q1
    lower  = Q1 - IQR_MULTIPLIER * IQR

    valid = [
        (tile, score, x, y)
        for tile, score, x, y in tiles_and_scores
        if score >= lower and not is_mostly_white(tile)
    ]

    return valid


def filter_tma_tiles(tiles_and_scores: list,
                     top_percentile: float = TMA_TOP_PERCENTILE) -> list:
    """
    For TMA slides: keep only high-variance tiles (tissue cores).

    Selects tiles in the top percentile by standard deviation score.
    These are almost certainly tissue cores rather than background glass.

    Returns list of (tile, score, x, y) for valid tiles only.
    """
    if not tiles_and_scores:
        return []

    scores    = np.array([s for _, s, _, _ in tiles_and_scores])
    threshold = np.percentile(scores, top_percentile)

    valid = [
        (tile, score, x, y)
        for tile, score, x, y in tiles_and_scores
        if score >= threshold
    ]

    return valid


# ── Statistics computation ─────────────────────────────────────────────────

def compute_slide_histogram_stats(slide: openslide.OpenSlide,
                                   slide_type: str = 'biopsy',
                                   tile_size: int = TILE_SIZE,
                                   progress_cb=None) -> dict:
    """
    Sample tiles from the slide, exclude outliers, and compute
    the mean histogram CDF from valid tissue tiles.

    Returns:
        {
            'cdf':          (256, 3) array,
            'valid_count':  int,
            'total_count':  int,
            'fallback':     bool,
        }
    """
    w, h = slide.dimensions

    # Generate sampling grid
    if slide_type == 'tma':
        positions = tma_grid_positions(w, h, TMA_GRID, tile_size)
    else:
        positions = biopsy_grid_positions(w, h, BIOPSY_GRID, tile_size)

    n_pos = len(positions)

    # Sample all tiles
    tiles_and_scores = []
    for i, (x, y) in enumerate(positions):
        try:
            tile  = read_tile(slide, x, y, tile_size)
            score = score_tile(tile)
            tiles_and_scores.append((tile, score, x, y))
        except Exception:
            pass
        if progress_cb:
            progress_cb(i + 1, n_pos, f'Sampling tiles {i+1}/{n_pos}…')

    # Filter based on slide type
    if slide_type == 'tma':
        valid = filter_tma_tiles(tiles_and_scores)
    else:
        valid = filter_biopsy_tiles(tiles_and_scores)

    # Fallback if too few valid tiles
    fallback = False
    if len(valid) < MIN_VALID_TILES:
        warnings.warn(
            f'Only {len(valid)} valid tiles found — using all sampled tiles.')
        valid    = tiles_and_scores
        fallback = True

    # Compute mean histogram
    sum_hist = np.zeros((256, 3), dtype=np.float64)
    for tile, _, _, _ in valid:
        for ch in range(3):
            h = np.bincount(tile[:, :, ch].ravel(), minlength=256)
            sum_hist[:, ch] += h

    avg_hist   = sum_hist / len(valid)
    target_cdf = np.zeros((256, 3), dtype=np.float64)
    for ch in range(3):
        cs = avg_hist[:, ch].cumsum()
        target_cdf[:, ch] = cs / cs[-1]

    return {
        'cdf':         target_cdf,
        'valid_count': len(valid),
        'total_count': len(tiles_and_scores),
        'fallback':    fallback,
    }


def compute_slide_reinhard_stats(slide: openslide.OpenSlide,
                                  slide_type: str = 'biopsy',
                                  tile_size: int = TILE_SIZE,
                                  progress_cb=None) -> dict:
    """
    Sample tiles, exclude outliers, and compute slide-level
    LAB mean and std from valid tissue tiles.

    Returns:
        {
            'mu':           array[3],
            'sigma':        array[3],
            'valid_count':  int,
            'total_count':  int,
            'fallback':     bool,
        }
    """
    w, h = slide.dimensions

    if slide_type == 'tma':
        positions = tma_grid_positions(w, h, TMA_GRID, tile_size)
    else:
        positions = biopsy_grid_positions(w, h, BIOPSY_GRID, tile_size)

    n_pos = len(positions)

    tiles_and_scores = []
    for i, (x, y) in enumerate(positions):
        try:
            tile  = read_tile(slide, x, y, tile_size)
            score = score_tile(tile)
            tiles_and_scores.append((tile, score, x, y))
        except Exception:
            pass
        if progress_cb:
            progress_cb(i + 1, n_pos, f'Sampling tiles {i+1}/{n_pos}…')

    if slide_type == 'tma':
        valid = filter_tma_tiles(tiles_and_scores)
    else:
        valid = filter_biopsy_tiles(tiles_and_scores)

    fallback = False
    if len(valid) < MIN_VALID_TILES:
        warnings.warn(
            f'Only {len(valid)} valid tiles found — using all sampled tiles.')
        valid    = tiles_and_scores
        fallback = True

    # Compute mean LAB statistics across valid tiles
    sum_mu    = np.zeros(3, dtype=np.float64)
    sum_sigma = np.zeros(3, dtype=np.float64)
    for tile, _, _, _ in valid:
        lab   = fast_rgb2lab(tile)
        flat  = lab.reshape(-1, 3)
        sum_mu    += flat.mean(axis=0)
        sum_sigma += flat.std(axis=0)

    return {
        'mu':          sum_mu    / len(valid),
        'sigma':       sum_sigma / len(valid),
        'valid_count': len(valid),
        'total_count': len(tiles_and_scores),
        'fallback':    fallback,
    }


# ── Batch average targets ──────────────────────────────────────────────────

def compute_batch_average_wsi_cdf(wsi_paths: list,
                                   slide_type: str = 'biopsy',
                                   tile_size: int = TILE_SIZE,
                                   progress_cb=None) -> np.ndarray:
    """
    Compute batch-average CDF across all WSI files.
    Samples each slide independently then averages.
    Returns (256, 3) array.
    """
    n        = len(wsi_paths)
    sum_cdf  = np.zeros((256, 3), dtype=np.float64)
    valid    = 0

    for i, path in enumerate(wsi_paths):
        try:
            slide  = open_slide(path)
            result = compute_slide_histogram_stats(
                slide, slide_type, tile_size)
            sum_cdf += result['cdf']
            valid   += 1
            slide.close()
        except Exception as e:
            warnings.warn(f'Could not process {path}: {e}')
        if progress_cb:
            progress_cb(i + 1, n, f'Analysing slide {i+1}/{n}…')

    if valid == 0:
        raise RuntimeError('No readable WSI files found.')

    return sum_cdf / valid


def compute_batch_average_wsi_reinhard(wsi_paths: list,
                                        slide_type: str = 'biopsy',
                                        tile_size: int = TILE_SIZE,
                                        progress_cb=None) -> dict:
    """
    Compute batch-average Reinhard statistics across all WSI files.
    Returns {'mu': array[3], 'sigma': array[3]}.
    """
    n         = len(wsi_paths)
    sum_mu    = np.zeros(3, dtype=np.float64)
    sum_sigma = np.zeros(3, dtype=np.float64)
    valid     = 0

    for i, path in enumerate(wsi_paths):
        try:
            slide  = open_slide(path)
            result = compute_slide_reinhard_stats(
                slide, slide_type, tile_size)
            sum_mu    += result['mu']
            sum_sigma += result['sigma']
            valid     += 1
            slide.close()
        except Exception as e:
            warnings.warn(f'Could not process {path}: {e}')
        if progress_cb:
            progress_cb(i + 1, n, f'Analysing slide {i+1}/{n}…')

    if valid == 0:
        raise RuntimeError('No readable WSI files found.')

    return {'mu': sum_mu / valid, 'sigma': sum_sigma / valid}


# ── Tile normalization pipeline ────────────────────────────────────────────




# ── Background-preserving normalization wrapper ────────────────────────────
#
# Background pixels (empty lumens, glass, slide margins) are identified
# using the LAB L* (lightness) channel rather than RGB mean.
#
# L* is perceptually weighted and separates pale-but-colored tissue from
# true white background more cleanly than RGB mean:
#   - Pale eosin-stained stroma typically has L* ≈ 75-88
#   - Near-white background / empty lumens typically have L* ≈ 92-100
#
# A pixel is classified as background when L* >= _BG_L_THRESHOLD.
# Background pixels are written as (255, 255, 255) in the output without
# passing through the normalization algorithm.
#
# _BG_L_THRESHOLD = 92 corresponds to approximately RGB mean 235-240,
# but because it operates in perceptual lightness space it is less likely
# to misclassify pale tissue as background compared to an RGB mean threshold.

_BG_L_THRESHOLD = 88.0   # LAB L* threshold (0-100 scale)


def _normalize_with_bg_mask(tile: np.ndarray,
                             normalize_fn) -> np.ndarray:
    """
    Apply normalize_fn to tile while preserving near-white background pixels.

    Background detection uses the LAB L* (lightness) channel computed via
    fast_rgb2lab().  Any pixel with L* >= _BG_L_THRESHOLD is classified as
    background and written as (255, 255, 255) in the output.
    """
    # Convert to LAB and extract L* channel — shape (H, W)
    lab     = fast_rgb2lab(tile)
    L       = lab[:, :, 0]
    bg_mask = L >= _BG_L_THRESHOLD

    # Entirely background tile — return white without calling the normalizer
    if bg_mask.all():
        return np.full_like(tile, 255)

    norm = normalize_fn(tile)
    norm = np.ascontiguousarray(norm, dtype=np.uint8)
    norm[bg_mask] = 255
    return norm



def _normalize_slide_pyvips(slide, normalize_fn, output_path,
                             tile_size, jpeg_quality, compression,
                             progress_cb=None, cancel_flag=None):
    """
    Normalize a full WSI and write a correctly assembled pyramidal TIFF
    compatible with OpenSlide, QuPath, and Bio-Formats.

    Output structure — sequential top-level IFDs with NewSubfileType tags:
        IFD 0  : full resolution,    NewSubfileType = 0
        IFD 1  : half resolution,    NewSubfileType = 1  (FILETYPE_REDUCEDIMAGE)
        IFD 2+ : quarter etc.,       NewSubfileType = 1

    OpenSlide's generic-tiff reader requires this exact layout.  SubIFDs
    (OME/tifffile default) are not followed by OpenSlide.

    Three-stage approach chosen for minimum peak RAM:

    Stage 1 — Normalize each tile and write to a temp directory as individual
               uncompressed TIFFs.  One tile in RAM at a time.

    Stage 2 — Stream level-0 tiles from disk into tifffile via an iterator,
               writing the full-resolution IFD with exact slide dimensions.
               File is closed after this step.

    Stage 3 — Open the written file with PIL (lazy, no pixel data loaded),
               call PIL.reduce(factor) to generate each sub-level via box
               sampling.  PIL.reduce() reads from disk without loading the
               full array into RAM — peak usage for this step is negligible
               regardless of slide size.  Sub-levels are appended to the file
               using tifffile's append mode.

    Peak RAM for a 37k×20k slide:
        Current (numpy canvas) approach : ~2.3 GB for sub-level assembly
        This approach                   : ~tile_size² × 3 bytes per tile
    """
    import tempfile
    import shutil
    import tifffile as _tifffile
    from PIL import Image as _PIL

    w, h      = slide.dimensions
    n_tiles_x = int(np.ceil(w / tile_size))
    n_tiles_y = int(np.ceil(h / tile_size))
    n_total   = n_tiles_x * n_tiles_y
    t0        = time.perf_counter()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: normalize tiles → temp directory ─────────────────────────
    tmp_dir   = Path(tempfile.mkdtemp(prefix='chameleon_wsi_tmp_'))
    completed = 0

    try:
        for ty in range(n_tiles_y):
            if cancel_flag and cancel_flag():
                break
            for tx in range(n_tiles_x):
                if cancel_flag and cancel_flag():
                    break

                x     = tx * tile_size
                y     = ty * tile_size
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)

                raw  = read_tile(slide, x, y, tile_size)
                crop = raw[:y_end - y, :x_end - x]
                norm = _normalize_with_bg_mask(crop, normalize_fn)
                norm = np.ascontiguousarray(norm, dtype=np.uint8)

                # Pad edge tiles to full tile_size — tifffile iterator
                # requires every yielded array to have the same shape
                if norm.shape[0] != tile_size or norm.shape[1] != tile_size:
                    padded = np.full(
                        (tile_size, tile_size, 3), 255, dtype=np.uint8)
                    padded[:norm.shape[0], :norm.shape[1]] = norm
                    norm = padded

                _PIL.fromarray(norm).save(
                    str(tmp_dir / f'tile_{ty:05d}_{tx:05d}.tif'),
                    format='TIFF', compression='raw')

                completed += 1
                if progress_cb:
                    progress_cb(completed, n_total,
                                f'Normalising tile {completed}/{n_total}...')

        if cancel_flag and cancel_flag():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {'filename': out_path.name,
                    'tiles_total': n_total, 'elapsed_s': 0}

        # ── Stage 2: write level-0 IFD via tile iterator ──────────────────
        if progress_cb:
            progress_cb(n_total, n_total, 'Writing full-resolution level...')

        # Resolution: MPP (µm/px) → pixels per cm  (1 cm = 10,000 µm)
        try:
            mpp   = float(slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X, 0.25))
            res   = 10_000.0 / mpp  # pixels per cm
            res_u = 3               # RESUNIT_CENTIMETER
        except Exception:
            res, res_u = 0.0, 1

        # Compression — JPEG requires imagecodecs; fall back to deflate
        if compression == 'deflate':
            comp      = 'deflate'
            comp_args = {'level': 6}
        else:
            try:
                import imagecodecs  # noqa: F401
                comp      = 'jpeg'
                comp_args = {'level': jpeg_quality}
            except ImportError:
                comp      = 'deflate'
                comp_args = {'level': 6}

        write_opts = dict(
            photometric    = 'rgb',
            planarconfig   = 'contig',
            dtype          = np.uint8,
            compression    = comp,
            compressionargs= comp_args,
            tile           = (tile_size, tile_size),
            resolution     = (res, res),
            resolutionunit = res_u,
            metadata       = None,
        )

        def _tile_iter():
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    yield np.array(_PIL.open(
                        str(tmp_dir / f'tile_{ty:05d}_{tx:05d}.tif')))

        with _tifffile.TiffWriter(str(out_path), bigtiff=True) as tif:
            tif.write(
                _tile_iter(),
                shape       = (h, w, 3),
                subfiletype = 0,
                **write_opts,
            )

        # ── Stage 3: append sub-levels using PIL.reduce() ─────────────────
        # PIL opens the file lazily — no pixel data is loaded until reduce()
        # is called.  reduce(factor) uses box sampling and reads from disk
        # without ever holding the full array in RAM.
        if progress_cb:
            progress_cb(n_total, n_total, 'Building pyramid sub-levels...')

        # Compute reduction factors: keep halving until < 2 × tile_size
        factors = []
        factor  = 2
        cw, ch  = w, h
        while cw > tile_size * 2 and ch > tile_size * 2:
            factors.append(factor)
            factor *= 2
            cw = w // factor
            ch = h // factor

        # Always at least one sub-level
        if not factors:
            factors = [2]

        _PIL.MAX_IMAGE_PIXELS = None   # disable decompression bomb limit for large WSI
        pil_l0 = _PIL.open(str(out_path))

        with _tifffile.TiffWriter(str(out_path), bigtiff=True,
                                   append=True) as tif:
            for factor in factors:
                lvl_img = pil_l0.reduce(factor)
                tif.write(
                    np.array(lvl_img),
                    subfiletype = 1,
                    **write_opts,
                )
                if progress_cb:
                    progress_cb(
                        n_total, n_total,
                        f'Writing sub-level {factor}x '
                        f'({lvl_img.width}×{lvl_img.height})...')

        pil_l0.close()
        elapsed = time.perf_counter() - t0
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            'filename':    out_path.name,
            'tiles_total': n_total,
            'elapsed_s':   round(elapsed, 2),
            'method':      'tifffile_sequential_ifds',
        }

    except Exception:
        # Preserve temp dir on failure for diagnosis
        raise


def normalize_slide_histogram(slide, target_cdf, output_path,
                               tile_size=TILE_SIZE, n_workers=4,
                               jpeg_quality=80, compression='jpeg',
                               progress_cb=None, cancel_flag=None):
    """Histogram normalization via pyvips streaming — no memmap."""
    return _normalize_slide_pyvips(
        slide,
        lambda img: apply_histogram_match(img, target_cdf),
        output_path, tile_size, jpeg_quality, compression,
        progress_cb, cancel_flag)


def normalize_slide_reinhard(slide, target_stats, slide_stats, output_path,
                              tile_size=TILE_SIZE, n_workers=4,
                              jpeg_quality=80, compression='jpeg',
                              progress_cb=None, cancel_flag=None):
    """Reinhard normalization via pyvips streaming — no memmap."""
    return _normalize_slide_pyvips(
        slide,
        lambda img: apply_reinhard(img, target_stats, src_stats=slide_stats),
        output_path, tile_size, jpeg_quality, compression,
        progress_cb, cancel_flag)

# ── Full batch runners ─────────────────────────────────────────────────────

def run_wsi_histogram_batch(wsi_paths: list,
                             target_cdf: np.ndarray,
                             output_dir: str,
                             slide_type: str = 'biopsy',
                             tile_size: int = TILE_SIZE,
                             n_workers: int = 4,
                             jpeg_quality: int = 80,
                             compression: str = 'jpeg',
                             progress_cb=None,
                             cancel_flag=None) -> list:
    """
    Normalize a batch of WSI files using histogram matching.
    Returns list of per-slide log dicts.
    """
    log = []
    n   = len(wsi_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(wsi_paths):
        if cancel_flag and cancel_flag():
            break
        if progress_cb:
            progress_cb(0, 1, f'Processing slide {i+1}/{n}: '
                        f'{Path(path).name}')
        try:
            slide    = open_slide(path)
            stem     = Path(path).stem
            out_path = os.path.join(output_dir, f'{stem}_norm.tiff')

            result = normalize_slide_histogram(
                slide, target_cdf, out_path,
                tile_size=tile_size,
                n_workers=n_workers,
                jpeg_quality=jpeg_quality,
                compression=compression,
                progress_cb=progress_cb,
                cancel_flag=cancel_flag,
            )
            slide.close()
            log.append(result)
        except Exception as e:
            import traceback
            print(f'\n--- ERROR processing {Path(path).name} ---')
            traceback.print_exc()
            print('---')
            log.append({'filename': Path(path).name, 'error': str(e)})

    return log


def run_wsi_reinhard_batch(wsi_paths: list,
                            target_stats: dict,
                            output_dir: str,
                            slide_type: str = 'biopsy',
                            tile_size: int = TILE_SIZE,
                            n_workers: int = 4,
                            jpeg_quality: int = 80,
                            compression: str = 'jpeg',
                            progress_cb=None,
                            cancel_flag=None) -> list:
    """
    Normalize a batch of WSI files using Reinhard color transfer.
    Computes slide-level source statistics per slide to prevent
    patchwork artefacts — the same transform is applied to every tile.
    Returns list of per-slide log dicts.
    """
    log = []
    n   = len(wsi_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(wsi_paths):
        if cancel_flag and cancel_flag():
            break
        if progress_cb:
            progress_cb(0, 1, f'Processing slide {i+1}/{n}: '
                        f'{Path(path).name}')
        try:
            slide = open_slide(path)
            stem  = Path(path).stem

            if progress_cb:
                progress_cb(0, 1, f'Computing slide statistics: '
                            f'{Path(path).name}')
            slide_stats = compute_slide_reinhard_stats(
                slide, slide_type, tile_size)

            out_path = os.path.join(output_dir, f'{stem}_norm.tiff')
            result   = normalize_slide_reinhard(
                slide, target_stats, slide_stats, out_path,
                tile_size=tile_size,
                n_workers=n_workers,
                jpeg_quality=jpeg_quality,
                compression=compression,
                progress_cb=progress_cb,
                cancel_flag=cancel_flag,
            )
            slide.close()

            result['valid_tiles']         = slide_stats['valid_count']
            result['total_tiles_sampled'] = slide_stats['total_count']
            result['fallback']            = slide_stats['fallback']
            log.append(result)

        except Exception as e:
            log.append({'filename': Path(path).name, 'error': str(e)})

    return log



# ── Stain method — reference slide fitting ─────────────────────────────────

def compute_reference_stain_macenko(slide: openslide.OpenSlide,
                                     slide_type: str = 'biopsy',
                                     tile_size: int = TILE_SIZE,
                                     progress_cb=None) -> dict:
    """
    Sample tissue tiles from the reference slide and fit the Macenko
    stain matrix. Returns the stain_params dict from fit_macenko().
    """
    w, h = slide.dimensions

    if slide_type == 'tma':
        positions = tma_grid_positions(w, h, TMA_GRID, tile_size)
    else:
        positions = biopsy_grid_positions(w, h, BIOPSY_GRID, tile_size)

    n_pos = len(positions)
    tiles_and_scores = []
    for i, (x, y) in enumerate(positions):
        try:
            tile  = read_tile(slide, x, y, tile_size)
            score = score_tile(tile)
            tiles_and_scores.append((tile, score, x, y))
        except Exception:
            pass
        if progress_cb:
            progress_cb(i + 1, n_pos, f'Sampling reference tiles {i+1}/{n_pos}…')

    if slide_type == 'tma':
        valid = filter_tma_tiles(tiles_and_scores)
    else:
        valid = filter_biopsy_tiles(tiles_and_scores)

    if len(valid) < MIN_VALID_TILES:
        valid = tiles_and_scores

    tissue_tiles = [tile for tile, _, _, _ in valid]
    return fit_macenko(tissue_tiles)


def compute_reference_stain_vahadane(slide: openslide.OpenSlide,
                                      slide_type: str = 'biopsy',
                                      tile_size: int = TILE_SIZE,
                                      progress_cb=None) -> dict:
    """
    Sample tissue tiles from the reference slide and fit the Vahadane
    stain matrix. Returns the stain_params dict from fit_vahadane().
    """
    w, h = slide.dimensions

    if slide_type == 'tma':
        positions = tma_grid_positions(w, h, TMA_GRID, tile_size)
    else:
        positions = biopsy_grid_positions(w, h, BIOPSY_GRID, tile_size)

    n_pos = len(positions)
    tiles_and_scores = []
    for i, (x, y) in enumerate(positions):
        try:
            tile  = read_tile(slide, x, y, tile_size)
            score = score_tile(tile)
            tiles_and_scores.append((tile, score, x, y))
        except Exception:
            pass
        if progress_cb:
            progress_cb(i + 1, n_pos, f'Sampling reference tiles {i+1}/{n_pos}…')

    if slide_type == 'tma':
        valid = filter_tma_tiles(tiles_and_scores)
    else:
        valid = filter_biopsy_tiles(tiles_and_scores)

    if len(valid) < MIN_VALID_TILES:
        valid = tiles_and_scores

    tissue_tiles = [tile for tile, _, _, _ in valid]
    return fit_vahadane(tissue_tiles)


# ── Stain normalization slide wrappers ─────────────────────────────────────

def normalize_slide_macenko(slide, stain_params, output_path,
                             tile_size=TILE_SIZE, n_workers=4,
                             jpeg_quality=80, compression='jpeg',
                             progress_cb=None, cancel_flag=None):
    """Macenko normalization via pyvips streaming."""
    return _normalize_slide_pyvips(
        slide,
        lambda img: normalize_macenko(img, stain_params),
        output_path, tile_size, jpeg_quality, compression,
        progress_cb, cancel_flag)


def normalize_slide_vahadane(slide, stain_params, output_path,
                              tile_size=TILE_SIZE, n_workers=4,
                              jpeg_quality=80, compression='jpeg',
                              progress_cb=None, cancel_flag=None):
    """Vahadane normalization via pyvips streaming."""
    return _normalize_slide_pyvips(
        slide,
        lambda img: normalize_vahadane(img, stain_params),
        output_path, tile_size, jpeg_quality, compression,
        progress_cb, cancel_flag)


# ── Stain method batch runners ─────────────────────────────────────────────

def run_wsi_macenko_batch(wsi_paths: list,
                           stain_params: dict,
                           output_dir: str,
                           slide_type: str = 'biopsy',
                           tile_size: int = TILE_SIZE,
                           n_workers: int = 4,
                           jpeg_quality: int = 80,
                           compression: str = 'jpeg',
                           progress_cb=None,
                           cancel_flag=None) -> list:
    """
    Normalize a batch of WSI files using Macenko stain normalization.
    stain_params is pre-computed from the reference slide.
    Returns list of per-slide log dicts.
    """
    log = []
    n   = len(wsi_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(wsi_paths):
        if cancel_flag and cancel_flag():
            break
        if progress_cb:
            progress_cb(0, 1, f'Processing slide {i+1}/{n}: {Path(path).name}')
        try:
            slide    = open_slide(path)
            stem     = Path(path).stem
            out_path = os.path.join(output_dir, f'{stem}_norm.tiff')

            result = normalize_slide_macenko(
                slide, stain_params, out_path,
                tile_size=tile_size, n_workers=n_workers,
                jpeg_quality=jpeg_quality, compression=compression,
                progress_cb=progress_cb, cancel_flag=cancel_flag,
            )
            slide.close()

            # Record stain matrix values for the log
            sm = stain_params['stain_matrix']
            result['stain_H_r'] = round(float(sm[0, 0]), 5)
            result['stain_H_g'] = round(float(sm[0, 1]), 5)
            result['stain_H_b'] = round(float(sm[0, 2]), 5)
            result['stain_E_r'] = round(float(sm[1, 0]), 5)
            result['stain_E_g'] = round(float(sm[1, 1]), 5)
            result['stain_E_b'] = round(float(sm[1, 2]), 5)
            log.append(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            log.append({'filename': Path(path).name, 'error': str(e)})

    return log


def run_wsi_vahadane_batch(wsi_paths: list,
                            stain_params: dict,
                            output_dir: str,
                            slide_type: str = 'biopsy',
                            tile_size: int = TILE_SIZE,
                            n_workers: int = 4,
                            jpeg_quality: int = 80,
                            compression: str = 'jpeg',
                            progress_cb=None,
                            cancel_flag=None) -> list:
    """
    Normalize a batch of WSI files using Vahadane stain normalization.
    stain_params is pre-computed from the reference slide.
    Returns list of per-slide log dicts.
    """
    log = []
    n   = len(wsi_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(wsi_paths):
        if cancel_flag and cancel_flag():
            break
        if progress_cb:
            progress_cb(0, 1, f'Processing slide {i+1}/{n}: {Path(path).name}')
        try:
            slide    = open_slide(path)
            stem     = Path(path).stem
            out_path = os.path.join(output_dir, f'{stem}_norm.tiff')

            result = normalize_slide_vahadane(
                slide, stain_params, out_path,
                tile_size=tile_size, n_workers=n_workers,
                jpeg_quality=jpeg_quality, compression=compression,
                progress_cb=progress_cb, cancel_flag=cancel_flag,
            )
            slide.close()

            sm = stain_params['stain_matrix']
            result['stain_H_r'] = round(float(sm[0, 0]), 5)
            result['stain_H_g'] = round(float(sm[0, 1]), 5)
            result['stain_H_b'] = round(float(sm[0, 2]), 5)
            result['stain_E_r'] = round(float(sm[1, 0]), 5)
            result['stain_E_g'] = round(float(sm[1, 1]), 5)
            result['stain_E_b'] = round(float(sm[1, 2]), 5)
            log.append(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            log.append({'filename': Path(path).name, 'error': str(e)})

    return log


# ── Tile-save pipeline ─────────────────────────────────────────────────────

def save_wsi_tiles(slide: openslide.OpenSlide,
                   output_dir: str,
                   slide_name: str,
                   normalize_fn,
                   slide_type: str = 'biopsy',
                   tile_size: int = TILE_SIZE,
                   progress_cb=None,
                   cancel_flag=None) -> dict:
    """
    Save normalized tissue tiles from a single WSI as individual PNG files.

    Background tiles are excluded using the same IQR outlier / white-pixel
    detection used by the normalization pipeline.  The user is informed of
    the counts via the returned dict and the progress callback.

    Tile filenames encode grid coordinates:
        {slide_name}_x{col:04d}_y{row:04d}.png

    Returns:
        {
            'filename':       str,
            'tiles_saved':    int,
            'tiles_skipped':  int,
            'tiles_total':    int,
            'elapsed_s':      float,
            'skipped_coords': list of (x, y) — top-left pixel of skipped tiles,
        }
    """
    from PIL import Image as _PIL_Image

    w, h = slide.dimensions
    n_tiles_x = int(np.ceil(w / tile_size))
    n_tiles_y = int(np.ceil(h / tile_size))
    n_total   = n_tiles_x * n_tiles_y

    slide_out_dir = Path(output_dir) / slide_name
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    saved    = 0
    skipped  = 0
    skipped_coords = []
    completed = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            if cancel_flag and cancel_flag():
                break

            x = tx * tile_size
            y = ty * tile_size

            try:
                tile = read_tile(slide, x, y, tile_size)

                # Background detection — same logic as normalization pipeline
                if is_mostly_white(tile) or score_tile(tile) < 5.0:
                    skipped += 1
                    skipped_coords.append((x, y))
                else:
                    norm = _normalize_with_bg_mask(tile, normalize_fn)
                    fname = f'{slide_name}_x{tx:04d}_y{ty:04d}.png'
                    _PIL_Image.fromarray(norm).save(
                        str(slide_out_dir / fname), 'PNG')
                    saved += 1

            except Exception:
                skipped += 1
                skipped_coords.append((x, y))

            completed += 1
            if progress_cb:
                progress_cb(
                    completed, n_total,
                    f'Tiles: {saved} saved, {skipped} background skipped '
                    f'({completed}/{n_total})')

        if cancel_flag and cancel_flag():
            break

    elapsed = time.perf_counter() - t0
    return {
        'filename':       slide_name,
        'tiles_saved':    saved,
        'tiles_skipped':  skipped,
        'tiles_total':    n_total,
        'elapsed_s':      round(elapsed, 2),
        'skipped_coords': skipped_coords,
    }


def run_wsi_tile_save_batch(wsi_paths: list,
                             normalize_fn,
                             output_dir: str,
                             slide_type: str = 'biopsy',
                             tile_size: int = TILE_SIZE,
                             progress_cb=None,
                             cancel_flag=None) -> list:
    """
    Save normalized tiles for all slides in the batch.
    normalize_fn is a callable(tile_array) → normalized_tile_array,
    already bound to the appropriate stain parameters.
    Returns list of per-slide result dicts.
    """
    log = []
    n   = len(wsi_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(wsi_paths):
        if cancel_flag and cancel_flag():
            break
        slide_name = Path(path).stem
        if progress_cb:
            progress_cb(0, 1, f'Saving tiles {i+1}/{n}: {Path(path).name}')
        try:
            slide  = open_slide(path)
            result = save_wsi_tiles(
                slide, output_dir, slide_name, normalize_fn,
                slide_type=slide_type, tile_size=tile_size,
                progress_cb=progress_cb, cancel_flag=cancel_flag,
            )
            slide.close()
            log.append(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            log.append({'filename': slide_name, 'error': str(e)})

    return log


# ── Dual log writer (TXT + CSV) ────────────────────────────────────────────

_MODE_DISPLAY_NAMES = {
    'HistMatch-Reference':  '1 – Histogram matching — reference image',
    'HistMatch-BatchAvg':   '2 – Histogram matching — batch average',
    'Reinhard-Reference':   '3 – Reinhard — reference image',
    'Reinhard-BatchAvg':    '4 – Reinhard — batch average',
    'Macenko-Reference':    '5 – Macenko — reference image',
    'Vahadane-Reference':   '6 – Vahadane — reference image',
}

_VERSION = '1.1'


def write_wsi_log(log: list,
                  output_dir: str,
                  mode_name: str,
                  ref_path: str = None,
                  tile_size: int = TILE_SIZE,
                  input_folder: str = None,
                  output_format: str = 'wsi'):
    """
    Write a human-readable TXT run log and a machine-readable CSV parameter
    log for the completed batch.

    Both files share a timestamp so they are clearly paired:
        chameleon_wsi_run_log_{mode}_{ts}.txt
        chameleon_wsi_normalization_params_{mode}_{ts}.csv

    The TXT contains the run header plus a one-line summary per slide.
    The CSV contains one row per slide with all numerical parameters.

    Background tile skip counts are recorded in both files when
    output_format == 'tiles'.
    """
    if not log:
        return

    ts           = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    display_mode = _MODE_DISPLAY_NAMES.get(mode_name, mode_name)
    out_dir      = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_mode = mode_name.replace(' ', '_').replace('–', '-')
    txt_path  = out_dir / f'chameleon_wsi_run_log_{safe_mode}_{ts}.txt'
    csv_path  = out_dir / f'chameleon_wsi_normalization_params_{safe_mode}_{ts}.csv'

    # ── TXT run log ───────────────────────────────────────────────────────
    now_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('=' * 72 + '\n')
        f.write('  CHAMELEON-WSI  —  Normalization Run Log\n')
        f.write(f'  Version  :  {_VERSION}\n')
        f.write('=' * 72 + '\n\n')

        f.write(f'  Date / Time    :  {now_str}\n')
        f.write(f'  Method         :  {display_mode}\n')
        f.write(f'  Output format  :  '
                f'{"Individual tiles (PNG)" if output_format == "tiles" else "Pyramidal TIFF"}\n')
        f.write(f'  Tile size      :  {tile_size} px\n')
        if input_folder:
            f.write(f'  Input folder   :  {input_folder}\n')
        if ref_path:
            f.write(f'  Reference slide:  {ref_path}\n')
        f.write(f'  Output folder  :  {output_dir}\n')
        f.write(f'  Slides in batch:  {len(log)}\n')

        # Stain matrix summary for Macenko/Vahadane (same for all slides)
        if log and 'stain_H_r' in log[0]:
            sm = log[0]
            f.write('\n  Stain matrix (from reference slide):\n')
            f.write(f'    Hematoxylin  R={sm["stain_H_r"]}  '
                    f'G={sm["stain_H_g"]}  B={sm["stain_H_b"]}\n')
            f.write(f'    Eosin        R={sm["stain_E_r"]}  '
                    f'G={sm["stain_E_g"]}  B={sm["stain_E_b"]}\n')

        if output_format == 'tiles':
            f.write('\n  NOTE: Background tiles were automatically excluded.\n')
            f.write('  Tile counts below reflect tissue tiles only.\n')

        f.write('\n' + '-' * 72 + '\n')
        f.write('  Per-slide summary\n')
        f.write('-' * 72 + '\n\n')

        errors = 0
        for entry in log:
            name = entry.get('filename', 'unknown')
            if 'error' in entry:
                f.write(f'  [ERROR]  {name}\n')
                f.write(f'           {entry["error"]}\n\n')
                errors += 1
            elif output_format == 'tiles':
                saved   = entry.get('tiles_saved',   '?')
                skipped = entry.get('tiles_skipped', '?')
                total   = entry.get('tiles_total',   '?')
                elapsed = entry.get('elapsed_s',     '?')
                f.write(f'  {name}\n')
                f.write(f'    Tiles saved    : {saved}\n')
                f.write(f'    Tiles skipped  : {skipped}  '
                        f'(background — excluded from output)\n')
                f.write(f'    Tiles evaluated: {total}\n')
                f.write(f'    Processing time: {elapsed}s\n\n')
            else:
                tiles   = entry.get('tiles_total', '?')
                elapsed = entry.get('elapsed_s',   '?')
                fallback = '  [fallback: all tiles used]' \
                           if entry.get('fallback') else ''
                f.write(f'  {name}\n')
                f.write(f'    Tiles processed: {tiles}{fallback}\n')
                f.write(f'    Processing time: {elapsed}s\n\n')

        f.write('-' * 72 + '\n')
        completed = len(log) - errors
        f.write(f'  Completed: {completed}/{len(log)} slides  |  '
                f'Errors: {errors}\n')
        f.write('=' * 72 + '\n')

    # ── CSV parameter log ─────────────────────────────────────────────────
    # Build unified fieldnames covering all possible entry keys
    all_keys = []
    seen     = set()
    # Fixed column order — common fields first, then method-specific
    priority = [
        'filename', 'tiles_total', 'tiles_saved', 'tiles_skipped',
        'valid_tiles', 'total_tiles_sampled', 'elapsed_s', 'fallback',
        'stain_H_r', 'stain_H_g', 'stain_H_b',
        'stain_E_r', 'stain_E_g', 'stain_E_b',
        'method', 'error',
    ]
    for k in priority:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for entry in log:
        for k in entry.keys():
            if k not in seen and k != 'skipped_coords':
                all_keys.append(k)
                seen.add(k)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(log)


