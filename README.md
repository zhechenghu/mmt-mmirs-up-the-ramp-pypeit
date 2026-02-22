# Up-the-Ramp Fitting Tool for MMIRS

Converts raw MMT/MMIRS H2RG detector ramp data into calibrated count-rate images for downstream spectroscopic reduction with [PypeIt](https://github.com/pypeit/PypeIt).

## Motivation

The official MMIRS pipeline ([Chilingarian et al. 2015](http://adsabs.harvard.edu/abs/2015PASP..127..406C)) requires IDL, which I am not familiar to. PypeIt supports MMIRS but uses a simple correlated double sampling (first minus last read), which leaves cosmic rays. L.A.Cosmic with the PypeIt default parameters often clipping real source flux when flagging CR. This tool performs proper up-the-ramp fitting with jump detection, producing cleaner images that PypeIt can reduce with default settings, or without CR flagging for bright sources.

## What It Does

MMIRS uses a Teledyne H2RG detector that performs non-destructive reads up the ramp. Each raw FITS file contains N reads stored as separate image extensions. This tool provides:

1. **Reference pixel correction** — follows `mmirs_read_amp` in PypeIt's [MMIRS module](https://github.com/pypeit/PypeIt/blob/release/pypeit/spectrographs/mmt_mmirs.py), which is the python implementation for [Chilingarian et al. 2015](http://adsabs.harvard.edu/abs/2015PASP..127..406C).
2. **Up-the-ramp fitting** — uses `fitramp.py`, copied from [t-brandt/fitramp](https://github.com/t-brandt/fitramp) ([Brandt 2024a](http://arxiv.org/abs/2404.01326), [Brandt 2024b](http://arxiv.org/abs/2309.08753)).
3. **Custom PypeIt spectrograph** — `mmt_mmirs_ramp.py`, a subclass of `MMTMMIRSSpectrograph` for handling ramp-fitted data. Install it manually into your PypeIt installation. More details are attached in the end of the README file.

### Auto-Sigma Calibration

The ramp fitter requires a single-read noise (sigma). But the FITS header `RDNOISE` (3.14 e-) is far too low, using it causes ~85% false-positive jump detections in a 300 s exposure. By default, the tool calibrates sigma from a dark frame:

1. Selects the dark with the most groups (requires >= 10 groups, `IMAGETYP = 'dark'`).
2. Ramp-fits 200 evenly-spaced rows from the central 80% of the detector without jump detection.
3. Rescales an initial guess so that median chi-squared matches expected degrees of freedom.

To skip auto-calibration, provide `--sig` explicitly or use `--no-auto-sig`.

## Requirements

- Python 3.8+
- numpy, astropy, scipy, pypeit

## Usage

```bash
python fit_mmirs_ramps.py PATH [PATH ...] [options]
```

`PATH` can be a FITS file or a directory. Directories are scanned for `*.fits` files; only multi-read ramp files (>= 2 image extensions) are processed.

Example — process an entire epoch:

```bash
python fit_mmirs_ramps.py data/ -o output
```

## Example Data Download

Large raw FITS example files are distributed as GitHub Release assets (not tracked in git), because GitHub blocks files larger than 100 MB in normal repository history.

Download the latest release assets into `example_data/`:

```bash
mkdir -p example_data
cd example_data

BASE_URL="https://github.com/zhechenghu/mmt-mmirs-up-the-ramp-pypeit/releases/latest/download"
curl -L -O "$BASE_URL/PAAlignJ134208_longslit.5475.fits"
curl -L -O "$BASE_URL/PAAlignJ134208_longslit.5476.fits"
curl -L -O "$BASE_URL/flat.5494.fits"
curl -L -O "$BASE_URL/flat.5495.fits"
```

If you need a specific version, replace `releases/latest` with `releases/download/<tag>`.

### Options

| Flag                     | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `-o`, `--output-dir DIR` | Output directory (default: `output`)                                            |
| `--sig FLOAT`            | Single-read noise in electrons. Providing this skips auto-calibration.          |
| `--no-auto-sig`          | Disable auto-sigma calibration; use `--sig` value or built-in default (9.5 e-). |
| `--gain FLOAT`           | Override GAIN from FITS header (e-/ADU).                                        |
| `--no-jumps`             | Disable cosmic ray / jump detection.                                            |
| `--no-refpix`            | Disable reference pixel subtraction.                                            |
| `--overwrite`            | Overwrite existing output files.                                                |

## Output Format

Each output file (`*_ramp.fits`) contains:

| Extension | Name    | Type        | Description                             |
| --------- | ------- | ----------- | --------------------------------------- |
| 0         | PRIMARY | Header only | Original metadata + ramp-fit parameters |
| 1         | SCI     | float32     | Count rate (e-/s)                       |
| 2         | ERR     | float32     | Uncertainty (e-/s)                      |
| 3         | CHI2    | float32     | Chi-squared per pixel                   |
| 4         | NJUMP   | int16       | Number of masked resultant differences  |

Processing parameters are recorded in the header as `RAMP_SIG`, `RAMPGAIN`, `RAMPJUMP`, `RAMPREFP`, and `RAMPSOFT`. PypeIt-required keywords are propagated to the SCI extension header.

## Processing Notes

- Data is trimmed to DATASEC `[5:2044, 5:2044]` (2040 x 2040 pixels).
- FITS extensions are sorted by EXTVER, not file order.
- Rows are processed one at a time to keep memory usage manageable.
- Files with only 1 image extension are automatically skipped.
- This tool performs ramp fitting only. Dark subtraction, flat fielding, wavelength calibration, etc. are handled by PypeIt.

# Custom PypeIt spectrograph

## How to install

Append the code in `mmt_mmirs_ramp.py` to the existing PypeIt module at `pypeit/spectrographs/mmt_mmirs.py`.

This works because PypeIt discovers spectrographs via `all_subclasses()` on already-imported modules — no changes to `__init__.py` needed.

## A note about trace fitting

If I use the default PypeIt parameters for MMT/MMIRS longslit reduction. It shows a ~5 px object trace offset, and real object light is always masked.

## Root Cause

MMIRS uses `bound_detector = True`, but the real slit edge is wider than the detector, so slit edges are synthetic straight lines at constant spatial pixel. The object trace is initialized from these straight edges, but the real trace is inclined. The `fit_trace()` function then anchor the fit to this bad initial guess. The loop in this function keep reject the source pixels and use noise to replace them. High-order polynomials (`trace_npoly=5`) make this worse: they are able to overfit noise and thus being pulled straight in the wings by the replacement values.

## Fix

| Parameter        | Default | Fix    | Why                                             |
| ---------------- | ------- | ------ | ----------------------------------------------- |
| `trace_npoly`    | 5       | **1**  | Too rigid to chase noise; captures linear trend |
| `trace_maxdev`   | 2.0     | **50** | Keeps all data points in the fit                |
| `trace_maxshift` | 1.0     | **20** | Lets centroids reach the true source position   |

All three are needed: maxshift lets centroids measure the real source, maxdev keeps them in the fit, and npoly=1 ensures a robust global fit. `npoly=2` was tested and still fails — the quadratic has enough freedom to be pulled off by the anchoring mechanisms.

## A note about noise

PypeIt treats the ramp-fitted image as a single read, so the read noise it applies is the single-read noise. To account for the noise reduction from ramp fitting, we convert the auto-calibrated single-read noise $\sigma$ into an effective read noise:

```
eff_ronoise = ramp_sig * np.sqrt(12.0 / ngroups)
```

**Derivation.** For $N$ uniformly spaced reads with interval $\Delta t$ and single-read noise $\sigma$, the read-noise contribution to the variance of the fitted slope (in counts) is

$$
\begin{aligned}
\mathrm{Var}_{\rm RN}(\text{counts}) &= \frac{12\,\sigma^2}{N(N^2-1)\,\Delta t^2} \times \bigl((N-1)\,\Delta t\bigr)^2 \\
&= \frac{12\,\sigma^2\,(N-1)}{N\,(N+1)}
\end{aligned}
$$

so the effective read noise is

$$\sigma_{\rm eff} = \sigma\,\sqrt{\frac{12\,(N-1)}{N\,(N+1)}} \;\approx\; \sigma\,\sqrt{\frac{12}{N}} \quad (N \gg 1)$$

This has been verified against the count variance of a synthetic dark frame with uniformly spaced reads and Gaussian read noise.

## Acknowledgments

Code was largely written with assistance from Claude Opus 4.6 with Claude Code. The ramp-fitting algorithm is from [Brandt (2024)](http://arxiv.org/abs/2404.01326). The overscan correction is from [Chilingarian et al. (2015)](http://adsabs.harvard.edu/abs/2015PASP..127..406C) and the python implementation is from [PypeIt](https://github.com/pypeit/PypeIt).
