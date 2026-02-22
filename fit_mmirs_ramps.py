#!/usr/bin/env python
"""
Up-the-ramp fitting for MMIRS multi-extension FITS files.

Reads raw MMIRS ramp data (ADU), converts to electrons using the GAIN
keyword, performs optimal ramp fitting with cosmic ray rejection using
the fitramp module, and writes output FITS files with count rate (e-/s),
uncertainty, chi-squared, and jump count per pixel.

Usage:
    python fit_mmirs_ramps.py PATH [PATH ...] [-o OUTPUT_DIR] [--sig 9.5]
                              [--gain 0.95] [--no-jumps] [--no-refpix]
                              [--no-auto-sig] [--overwrite]

PATH can be a FITS file or a directory.  When a directory is given, all
*.fits files in that directory (non-recursive) are scanned and only
multi-read ramp files (>= 2 image extensions) are selected.
"""

import sys
import os
import glob
import argparse
import time
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from astropy.stats import sigma_clipped_stats

# Add the reference code directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_provided_ref")
)
import fitramp


# --- Constants ---
DATASEC_SLICE = (
    slice(4, 2044),
    slice(4, 2044),
)  # Python 0-indexed [5:2044] FITS -> [4:2044) Python
DEFAULT_SIG = 9.5  # electrons, single-read noise after refpix correction (EP1; use --sig 19.0 for EP2)
DEFAULT_OUTPUT_DIR = "output"
REFPIX_NAMPS = 32  # Number of readout amplifiers
REFPIX_SAVGOL_WINDOW = 11
REFPIX_SAVGOL_POLYORDER = 5
REFPIX_SIGMA_CLIP = 3.0


def find_dark_frame(files, min_groups=10):
    """
    Find the best dark frame for sigma calibration from a list of FITS files.

    Selects the dark with the most groups (reads), requiring at least
    min_groups for sufficient statistics.

    Parameters
    ----------
    files : list of str
        List of FITS file paths to search.
    min_groups : int
        Minimum number of groups required (default 10).

    Returns
    -------
    best_path : str or None
        Path to the best dark frame, or None if none found.
    best_ngroups : int or None
        Number of groups in the best dark, or None if none found.
    """
    best_path = None
    best_ngroups = None
    for fpath in files:
        try:
            with fits.open(fpath) as hdul:
                # Count image extensions
                img_hdus = [
                    hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2
                ]
                ngroups = len(img_hdus)
                if ngroups < min_groups:
                    continue
                # Check IMAGETYP from last extension (highest EXTVER has metadata)
                last_hdu = max(img_hdus, key=lambda h: h.header.get("EXTVER", 0))
                imagetyp = last_hdu.header.get("IMAGETYP", "")
                if imagetyp.strip().lower() != "dark":
                    continue
                if best_ngroups is None or ngroups > best_ngroups:
                    best_path = fpath
                    best_ngroups = ngroups
        except Exception:
            continue
    return best_path, best_ngroups


def find_ramp_files(directory):
    """
    Scan a directory for FITS files that are multi-read ramps.

    Checks each *.fits file (non-recursive) by counting image extensions.
    Files with >= 2 image extensions are considered valid ramps.

    Parameters
    ----------
    directory : str
        Path to directory to scan.

    Returns
    -------
    ramp_files : list of str
        Sorted list of paths to valid ramp FITS files.
    skipped : list of (str, str)
        List of (filename, reason) for skipped files.
    """
    pattern = os.path.join(directory, "*.fits")
    candidates = sorted(glob.glob(pattern))
    ramp_files = []
    skipped = []
    for fpath in candidates:
        try:
            with fits.open(fpath) as hdul:
                nimages = sum(
                    1 for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2
                )
            if nimages >= 2:
                ramp_files.append(fpath)
            else:
                skipped.append(
                    (os.path.basename(fpath), f"{nimages} image ext (need >= 2)")
                )
        except Exception as e:
            skipped.append((os.path.basename(fpath), str(e)))
    return ramp_files, skipped


def refpix_correct(img, namps=REFPIX_NAMPS):
    """
    Reference pixel correction for H2RG detector with multi-amplifier readout.

    Adapted from PypeIt mmirs_read_amp() / MMIRS IDL pipeline refpix.pro.

    Stage 1: Subtract row-by-row bias drift using reference columns (cols 0-3
    and 2044-2047). The per-row mean of 8 reference pixels is Savitzky-Golay
    smoothed to avoid injecting noise.

    Stage 2: Subtract per-amplifier DC offset using reference rows (rows 1-3
    and 2044-2047; row 0 is skipped due to anomalous offset). For each amp,
    even and odd columns are corrected separately (they can have different
    offsets), then averaged.

    Parameters
    ----------
    img : ndarray, shape (nrows, ncols), float64
        Single detector read in ADU. Modified in place.
    namps : int
        Number of readout amplifiers (default 32).

    Returns
    -------
    img : ndarray
        Corrected image (same array, modified in place).
    """
    nrows, ncols = img.shape
    ampsize = ncols // namps

    # --- Stage 1: Row-by-row correction from reference columns ---
    ref_cols = np.concatenate([np.arange(4), np.arange(ncols - 4, ncols)])
    refvec = np.mean(img[:, ref_cols], axis=1)
    svec = savgol_filter(
        refvec, REFPIX_SAVGOL_WINDOW, polyorder=REFPIX_SAVGOL_POLYORDER
    )
    img -= svec[:, np.newaxis]

    # --- Stage 2: Per-amplifier DC offset from reference rows ---
    # Skip row 0 (anomalous offset); use rows 1-3 and 2044-2047
    ref_rows = np.concatenate([np.arange(1, 4), np.arange(nrows - 4, nrows)])
    img_ref = img[ref_rows, :]

    for amp in range(namps):
        col_start = amp * ampsize
        even_cols = col_start + 2 * np.arange(ampsize // 2)
        odd_cols = even_cols + 1
        ref_even, _, _ = sigma_clipped_stats(
            img_ref[:, even_cols], sigma=REFPIX_SIGMA_CLIP
        )
        ref_odd, _, _ = sigma_clipped_stats(
            img_ref[:, odd_cols], sigma=REFPIX_SIGMA_CLIP
        )
        offset = (ref_even + ref_odd) / 2.0
        img[:, col_start : col_start + ampsize] -= offset

    return img


def load_ramp(filepath, refpix=True):
    """
    Load a MMIRS FITS ramp file.

    Opens the FITS file, sorts image extensions by EXTVER to get correct
    read order, optionally applies reference pixel correction, and extracts
    the DATASEC region as float64.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    refpix : bool
        Apply reference pixel subtraction before trimming to DATASEC.

    Returns
    -------
    reads : ndarray, shape (ngroups, 2040, 2040), float64
        The non-destructive reads in time order.
    header : fits.Header
        Primary header of the FITS file.
    grptime : float
        Time between groups in seconds (GRPTIME keyword).
    ngroups : int
        Number of groups (reads) in the ramp.
    gain : float
        Detector gain in e-/ADU (GAIN keyword).
    """
    with fits.open(filepath) as hdul:
        # Collect image extensions and sort by EXTVER for correct read order
        img_hdus = [
            (hdu.header["EXTVER"], hdu)
            for hdu in hdul
            if hdu.data is not None and hdu.data.ndim == 2
        ]
        img_hdus.sort(key=lambda x: x[0])

        # Grab header from last extension (highest EXTVER) — this contains
        # all observational metadata (WCS, target, detector, timing, etc.).
        # The primary HDU header is nearly empty, and earlier reads have
        # incomplete headers (the controller fills metadata on the final read).
        header = img_hdus[-1][1].header.copy()

        ngroups = len(img_hdus)
        # Extract DATASEC region
        ny = DATASEC_SLICE[0].stop - DATASEC_SLICE[0].start
        nx = DATASEC_SLICE[1].stop - DATASEC_SLICE[1].start
        reads = np.empty((ngroups, ny, nx), dtype=np.float64)

        for i, (_, hdu) in enumerate(img_hdus):
            if refpix:
                # Load full frame, apply refpix in ADU space, then trim
                frame = hdu.data.astype(np.float64)
                refpix_correct(frame)
                reads[i] = frame[DATASEC_SLICE]
            else:
                reads[i] = hdu.data[DATASEC_SLICE].astype(np.float64)

        grptime = header.get("GRPTIME")
        if grptime is None:
            raise ValueError(f"GRPTIME keyword not found in {filepath}")

        gain = header.get("GAIN")
        if gain is None:
            raise ValueError(f"GAIN keyword not found in {filepath}")

    return reads, header, grptime, ngroups, gain


def compute_diffs(reads, covar):
    """
    Compute scaled resultant differences from reads.

    Parameters
    ----------
    reads : ndarray, shape (ngroups, ny, nx)
        Non-destructive reads in time order.
    covar : fitramp.Covar
        Covariance object (used for delta_t).

    Returns
    -------
    diffs : ndarray, shape (ngroups-1, ny, nx), float64
        Scaled resultant differences: (reads[i+1] - reads[i]) / delta_t[i].
    """
    ngroups, ny, nx = reads.shape
    diffs = np.empty((ngroups - 1, ny, nx), dtype=np.float64)
    for i in range(ngroups - 1):
        diffs[i] = (reads[i + 1] - reads[i]) / covar.delta_t[i]
    return diffs


def calibrate_sigma(
    filepath, sig_guess=15.0, refpix=True, gain_override=None, nrows_cal=200
):
    """
    Auto-calibrate single-read noise from a dark frame.

    Ramp-fits a subset of rows without jump detection, measures median
    chi-squared, and rescales sig_guess so that chi² matches the expected
    degrees of freedom.

    Parameters
    ----------
    filepath : str
        Path to a dark FITS file.
    sig_guess : float
        Initial guess for single-read noise in electrons (default 15.0).
    refpix : bool
        Apply reference pixel correction.
    gain_override : float or None
        Override GAIN from FITS header.
    nrows_cal : int
        Number of rows to sample for calibration (default 200).

    Returns
    -------
    sig_cal : float
        Calibrated single-read noise in electrons.
    median_chisq : float
        Measured median chi-squared.
    expected_chisq : float
        Expected chi-squared (ngroups - 2).
    """
    # Load the dark frame
    reads, header, grptime, ngroups, gain_hdr = load_ramp(filepath, refpix=refpix)
    gain = gain_override if gain_override is not None else gain_hdr
    reads *= gain  # ADU -> electrons

    # Build covariance and compute diffs
    readtimes = [grptime * (i + 1) for i in range(ngroups)]
    covar = fitramp.Covar(readtimes)
    diffs = compute_diffs(reads, covar)
    del reads

    ny, nx = diffs.shape[1], diffs.shape[2]

    # Select nrows_cal evenly-spaced rows from the central 80%
    margin = int(ny * 0.10)
    row_candidates = np.arange(margin, ny - margin)
    nrows_cal = min(nrows_cal, len(row_candidates))
    indices = np.linspace(0, len(row_candidates) - 1, nrows_cal, dtype=int)
    cal_rows = row_candidates[indices]

    # Ramp-fit selected rows without jump detection (two-pass debias)
    sig_row = np.full(nx, sig_guess, dtype=np.float64)
    all_chisq = []
    for row in cal_rows:
        result = fitramp.fit_ramps(diffs[:, row, :], covar, sig_row)
        countrateguess = result.countrate * (result.countrate > 0)
        result = fitramp.fit_ramps(
            diffs[:, row, :], covar, sig_row, countrateguess=countrateguess
        )
        all_chisq.append(result.chisq)

    all_chisq = np.concatenate(all_chisq)
    median_chisq = float(np.median(all_chisq))

    # Expected chi² = ngroups - 2 (N-1 diffs minus 1 fitted slope)
    expected_chisq = float(ngroups - 2)
    sig_cal = sig_guess * np.sqrt(median_chisq / expected_chisq)

    return float(sig_cal), median_chisq, expected_chisq


def save_output(filepath, countrate, uncert, chisq, njumps, header, params):
    """
    Write ramp-fit results to a multi-extension FITS file.

    Extensions:
        Primary HDU: input header with processing keywords
        SCI: count rate (float32, electron/s)
        ERR: uncertainty (float32, electron/s)
        CHI2: chi-squared (float32)
        NJUMP: number of masked diffs per pixel (int16)

    Parameters
    ----------
    filepath : str
        Output file path.
    countrate : ndarray (2040, 2040)
    uncert : ndarray (2040, 2040)
    chisq : ndarray (2040, 2040)
    njumps : ndarray (2040, 2040)
    header : fits.Header
        Original primary header.
    params : dict
        Processing parameters to record in header.
    """
    # Build primary header from extension header, stripping extension-only keys
    phdr = header.copy()
    for kw in ["XTENSION", "EXTNAME", "EXTVER", "PCOUNT", "GCOUNT", "INHERIT"]:
        phdr.pop(kw, None)
    phdr["RAMPFIT"] = (True, "Up-the-ramp fitting applied")
    phdr["RAMP_SIG"] = (params["sig"], "Read noise used (electrons)")
    phdr["RAMPGAIN"] = (params["gain"], "Gain used (e-/ADU)")
    phdr["RAMPJUMP"] = (params["detect_jumps"], "Jump detection enabled")
    phdr["RAMPREFP"] = (params["refpix"], "Reference pixel correction applied")
    phdr["RAMPSOFT"] = ("fit_mmirs_ramps.py", "Ramp fitting software")
    # Adjust CRPIX for DATASEC trim (shifted by 4 pixels)
    for kw in ["CRPIX1", "CRPIX2"]:
        if kw in phdr:
            phdr[kw] = phdr[kw] - 4

    primary = fits.PrimaryHDU(header=phdr)
    sci = fits.ImageHDU(countrate.astype(np.float32), name="SCI")
    sci.header["BUNIT"] = ("electron/s", "Count rate units")

    # Propagate metadata to SCI header so PypeIt can find them at ext=1
    _PYPEIT_KEYWORDS = [
        "RA",
        "DEC",
        "OBJECT",
        "APERTURE",
        "FILTER",
        "EXPTIME",
        "AIRMASS",
        "DISPERSE",
        "IMAGETYP",
        "INSTRUME",
        "DATE-OBS",
        "GAIN",
        "RAMPFIT",
        "RAMP_SIG",
        "RAMPGAIN",
        "RAMPJUMP",
        "RAMPREFP",
        "HXRGGRUP",
    ]
    for kw in _PYPEIT_KEYWORDS:
        if kw in phdr:
            sci.header[kw] = (phdr[kw], phdr.comments[kw])
    sci.header["CCDSUM"] = ("1 1", "CCD pixel binning (spatial spectral)")
    sci.header["DATASEC"] = ("[1:2040,1:2040]", "Data section (trimmed)")

    err = fits.ImageHDU(uncert.astype(np.float32), name="ERR")
    err.header["BUNIT"] = ("electron/s", "Uncertainty units")
    chi2 = fits.ImageHDU(chisq.astype(np.float32), name="CHI2")
    chi2.header["BUNIT"] = ("", "Chi-squared (dimensionless)")
    njmp = fits.ImageHDU(njumps.astype(np.int16), name="NJUMP")
    njmp.header["BUNIT"] = ("", "Number of masked resultant differences")

    hdul = fits.HDUList([primary, sci, err, chi2, njmp])
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    hdul.writeto(filepath, overwrite=params.get("overwrite", False))


def fit_single_file(
    filepath,
    output_dir,
    sig,
    detect_jumps=True,
    overwrite=False,
    gain_override=None,
    refpix=True,
):
    """
    Perform up-the-ramp fitting on a single MMIRS FITS file.

    Parameters
    ----------
    filepath : str
        Input FITS file path.
    output_dir : str
        Directory for output file.
    sig : float
        Single-read noise in electrons.
    detect_jumps : bool
        Whether to run cosmic ray detection.
    overwrite : bool
        Whether to overwrite existing output.
    gain_override : float or None
        If set, override the GAIN keyword from the FITS header.
    refpix : bool
        Apply reference pixel subtraction (default True).

    Returns
    -------
    outpath : str
        Path to the output file.
    """
    basename = os.path.basename(filepath)
    name, ext = os.path.splitext(basename)
    outname = f"{name}_ramp{ext}"
    outpath = os.path.join(output_dir, outname)

    if os.path.exists(outpath) and not overwrite:
        print(f"  Output exists, skipping: {outpath}", flush=True)
        return outpath

    t0 = time.time()
    print(f"  Loading {basename}...", flush=True)

    # 1. Load reads
    reads, header, grptime, ngroups, gain_hdr = load_ramp(filepath, refpix=refpix)
    gain = gain_override if gain_override is not None else gain_hdr
    ny, nx = reads.shape[1], reads.shape[2]
    print(
        f"  {ngroups} groups, {ny}x{nx} pixels, GRPTIME={grptime:.3f}s, "
        f"GAIN={gain:.2f} e-/ADU",
        flush=True,
    )

    # Convert reads from ADU to electrons
    reads *= gain

    # 2. Build Covar from read times
    readtimes = [grptime * (i + 1) for i in range(ngroups)]
    covar = fitramp.Covar(readtimes)

    # 3. Compute scaled diffs, free reads
    diffs = compute_diffs(reads, covar)
    del reads

    # 4. Allocate output arrays
    countrate = np.empty((ny, nx), dtype=np.float64)
    uncert = np.empty((ny, nx), dtype=np.float64)
    chisq = np.empty((ny, nx), dtype=np.float64)
    njumps = np.zeros((ny, nx), dtype=np.int32)

    # sig as 1D array for each row
    sig_row = np.full(nx, sig, dtype=np.float64)

    # 5. Row-by-row fitting
    ndiffs = diffs.shape[0]
    for row in range(ny):
        if detect_jumps:
            diffs2use, countrates = fitramp.mask_jumps(diffs[:, row, :], covar, sig_row)
            result = fitramp.fit_ramps(
                diffs[:, row, :],
                covar,
                sig_row,
                diffs2use=diffs2use,
                countrateguess=countrates * (countrates > 0),
            )
            njumps[row] = ndiffs - np.sum(diffs2use, axis=0)
        else:
            result = fitramp.fit_ramps(diffs[:, row, :], covar, sig_row)
            # Debias: refit with clamped countrate guess
            countrateguess = result.countrate * (result.countrate > 0)
            result = fitramp.fit_ramps(
                diffs[:, row, :], covar, sig_row, countrateguess=countrateguess
            )

        countrate[row] = result.countrate
        uncert[row] = result.uncert
        chisq[row] = result.chisq

        if (row + 1) % 200 == 0 or row == ny - 1:
            elapsed = time.time() - t0
            print(f"  Row {row + 1}/{ny}  ({elapsed:.1f}s)", flush=True)

    # 6. Save output
    params = {
        "sig": sig,
        "gain": gain,
        "detect_jumps": detect_jumps,
        "refpix": refpix,
        "overwrite": overwrite,
    }
    save_output(outpath, countrate, uncert, chisq, njumps, header, params)
    elapsed = time.time() - t0
    print(f"  Saved: {outpath}  ({elapsed:.1f}s total)", flush=True)
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Up-the-ramp fitting for MMIRS FITS files."
    )
    parser.add_argument("paths", nargs="+", help="Input FITS file(s) or directory(ies)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sig",
        type=float,
        default=None,
        help="Single-read noise in electrons. If not set, auto-calibrated from "
        "the longest dark frame in the input files.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Override GAIN from FITS header (e-/ADU)",
    )
    parser.add_argument(
        "--no-jumps", action="store_true", help="Disable cosmic ray / jump detection"
    )
    parser.add_argument(
        "--no-refpix", action="store_true", help="Disable reference pixel subtraction"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--no-auto-sig",
        action="store_true",
        help=f"Disable auto-sigma calibration; use --sig or default ({DEFAULT_SIG} e-).",
    )
    args = parser.parse_args()

    detect_jumps = not args.no_jumps
    refpix = not args.no_refpix

    # Expand directories into individual ramp FITS files
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            print(f"Scanning directory: {path}")
            ramp_files, skipped = find_ramp_files(path)
            print(f"  Found {len(ramp_files)} ramp files", end="")
            if skipped:
                print(f", skipped {len(skipped)}:", end="")
                for name, reason in skipped:
                    print(f"\n    {name}: {reason}", end="")
            print()
            files.extend(ramp_files)
        else:
            files.append(path)

    if not files:
        print("No files to process.")
        sys.exit(0)

    # Determine sigma: auto-calibrate from dark (default) or use explicit value
    if args.sig is not None:
        # Explicit --sig provided: use it directly
        if not args.no_auto_sig:
            print(f"Note: --sig={args.sig} provided, skipping auto-calibration.")
    elif args.no_auto_sig:
        # No --sig and --no-auto-sig: fall back to default
        args.sig = DEFAULT_SIG
    else:
        # Default: auto-calibrate from dark frame
        print("Auto-sigma calibration: searching for dark frames...", flush=True)
        dark_path, dark_ngroups = find_dark_frame(files)
        if dark_path is None:
            print(
                "ERROR: No dark frame with >= 10 groups found in the input files.\n"
                "Provide --sig manually or include dark frames in the input."
            )
            sys.exit(1)
        print(
            f"  Selected dark: {os.path.basename(dark_path)} "
            f"({dark_ngroups} groups)",
            flush=True,
        )
        t_cal = time.time()
        sig_cal, median_chisq, expected_chisq = calibrate_sigma(
            dark_path,
            sig_guess=15.0,
            refpix=refpix,
            gain_override=args.gain,
        )
        dt_cal = time.time() - t_cal
        print(
            f"  Calibrated sigma: {sig_cal:.2f} e-  "
            f"(median chi²={median_chisq:.2f}, expected={expected_chisq:.1f}, "
            f"{dt_cal:.1f}s)",
            flush=True,
        )
        args.sig = sig_cal

    gain_str = f"{args.gain:.2f} e-/ADU (override)" if args.gain else "from header"
    print(
        f"\nRamp fitting: sig={args.sig} e-, gain={gain_str}, "
        f"jump_detection={'ON' if detect_jumps else 'OFF'}, "
        f"refpix={'ON' if refpix else 'OFF'}"
    )
    print(f"Output dir: {args.output_dir}")
    print(f"Files to process: {len(files)}")
    print()

    n_success = 0
    n_fail = 0
    for filepath in files:
        print(f"[{n_success + n_fail + 1}/{len(files)}] {filepath}")
        try:
            fit_single_file(
                filepath,
                args.output_dir,
                args.sig,
                detect_jumps=detect_jumps,
                overwrite=args.overwrite,
                gain_override=args.gain,
                refpix=refpix,
            )
            n_success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            n_fail += 1
        print()

    print(f"Done: {n_success} succeeded, {n_fail} failed.")


if __name__ == "__main__":
    main()
