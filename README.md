# MOT Image Analysis GUI

A PyQt5-based desktop application for analysing absorption images from a **Magneto-Optical Trap (MOT)** experiment. Built to run both as a standalone tool and as an **ARTIQ dashboard applet** for live data acquisition.

## Features

- **Optical density (OD)** image computation from raw absorption imaging frames (`_withatoms`, `_withoutatoms`, `_background`)
- **2D Gaussian fit** with physical validation — automatically rejects fits that are too wide, sub-pixel, or off-centre
- **Interactive ROI** selection (drag or enter values) for isolating the atom cloud
- **Cloud width measurement** overlay with physical scale bar (configurable pixel calibration in mm/px)
- **Marginal distribution plots** (X and Y projections) with fit overlay
- **Sequence analysis window** — batch-processes a series of scans from an HDF5 file, extracts cloud width or integrated absorption vs. any experimental parameter, and plots results in real time
- **CSV export** of sequence analysis results including file metadata
- Supports `.h5` / `.hdf5` files produced by ARTIQ

## Project Structure

```
ARTIQ/
└── GUI/
    ├── main.py               # Entry point (standalone + ARTIQ applet)
    ├── model.py              # Data model: HDF5 loading, OD, fitting, sequence analysis
    ├── view.py               # Main image viewer widget
    ├── controller.py         # MVC controller wiring model ↔ view
    ├── sequence_view.py      # Sequence analysis sub-window
    ├── sequence_controller.py# Controller for the sequence window
    └── Analysis.py           # Physics: OD formula, 2D Gaussian, fit validation
```

## Installation

**Requirements**: Python 3.8+, conda recommended.

```bash
conda create -n mot_analysis python=3.10
conda activate mot_analysis
pip install pyqt5 pyqtgraph numpy scipy h5py
```

> If using as an ARTIQ applet, ARTIQ must be installed in the same environment.

## Running

### Standalone (no ARTIQ)

```bash
cd GUI
python main.py
```

### As ARTIQ applet

Register `GUI/main.py` as an applet in the ARTIQ dashboard and subscribe it to your camera image dataset.

## HDF5 File Format

The tool expects HDF5 files where each scan group contains:

| Dataset | Description |
|---------|-------------|
| `_withatoms` | 2D image with atoms |
| `_withoutatoms` | 2D reference image |
| `_background` | 2D dark frame |
| `Parameters/<name>` | Experimental parameters (used as x-axis in sequence analysis) |

## Fit Validation

The Gaussian fit is automatically rejected if:
- `sigma > 50%` of the image dimension (cloud too wide → likely background noise)
- `sigma < 1 px` (sub-pixel → noise)
- Amplitude `A ≤ 0` (fitting a dip, not a peak)
- Centre outside image bounds

Rejected fits in sequence analysis are **skipped and logged** without crashing the analysis.

## License

MIT
