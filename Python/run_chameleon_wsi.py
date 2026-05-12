"""
run_wsi.py
==========
Launcher for Chameleon-WSI.

Usage:
    python run_wsi.py

Requirements:
    pip install PyQt5 matplotlib numpy scikit-image Pillow openslide-python tifffile zarr

Note on OpenSlide:
    Windows requires the OpenSlide binaries in addition to the Python package.
    Download from: https://openslide.org/download/
    Extract and add the 'bin' folder to your system PATH, or place the DLLs
    in the same folder as this script.

Packaging as standalone executable:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "ChameleonWSI" run_wsi.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # OpenSlide DLLs
    openslide_path = os.path.join(script_dir, 'openslide', 'bin')
    if os.path.exists(openslide_path):
        os.add_dll_directory(openslide_path)
        os.environ['PATH'] = openslide_path + os.pathsep + os.environ.get('PATH', '')

    # libvips DLLs — find any vips-dev-* folder
    for entry in os.listdir(script_dir):
        vips_bin = os.path.join(script_dir, entry, 'bin')
        if entry.startswith('vips') and os.path.isdir(vips_bin):
            os.add_dll_directory(vips_bin)
            os.environ['PATH'] = vips_bin + os.pathsep + os.environ.get('PATH', '')
            break

# Import pyvips here once — before any other module imports it —
# so it loads with the correct PATH already set.
# This prevents the cached import failure that occurs when pyvips
# is first imported inside a worker thread without DLL visibility.
try:
    import pyvips
    print(f'pyvips {pyvips.__version__} loaded OK')
except Exception as e:
    print(f'WARNING: pyvips failed to load: {e}')
    print('Output will fall back to uncompressed TIFF if pyvips is unavailable.')

from chameleon_wsi_app import main

if __name__ == '__main__':
    main()
