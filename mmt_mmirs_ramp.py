"""
Custom PypeIt spectrograph module for ramp-fitted MMIRS data.

Subclasses the standard MMTMMIRSSpectrograph to handle files produced by
fit_mmirs_ramps.py. Ramp-fitted files are detected via the RAMPFIT keyword
in the ext 1 header. For these files:

  - get_rawimage() reads the SCI extension directly (already processed),
    converts from electron/s to total electrons via EXPTIME, and skips
    CDS and reference pixel correction.

  - get_detector_par() returns gain=1.0 (data already in electrons),
    effective ramp-fit read noise, and darkcurr=0.0.

Raw (non-ramp-fitted) files are handled identically to the original module.

Installation:
    1. Copy the module to /Users/zchu/MyPrograms/miniconda3/envs/pypeit/lib/python3.11/site-packages/pypeit/spectrographs/mmt_mmirs_ramp.py.
    2. Register it by adding from pypeit.spectrographs import mmt_mmirs_ramp to spectrographs/__init__.py.
"""

import numpy as np

from astropy.io import fits

from pypeit import log
from pypeit import utils
from pypeit import io
from pypeit.core import parse
from pypeit.images import detector_container
from pypeit.spectrographs.mmt_mmirs import MMTMMIRSSpectrograph, mmirs_read_amp


class MMTMMIRSRampSpectrograph(MMTMMIRSSpectrograph):
    """
    MMT/MMIRS spectrograph with support for ramp-fitted data.

    Inherits everything from MMTMMIRSSpectrograph. Only get_rawimage()
    and get_detector_par() are overridden to handle ramp-fitted files.
    """

    ndet = 1
    name = "mmt_mmirs_ramp"
    supported = True

    def get_detector_par(self, det, hdu=None):
        """
        Return detector metadata, adapted for ramp-fitted data.

        For ramp-fitted files (RAMPFIT=True in ext 1):
          - gain = 1.0 (data is in electrons after EXPTIME multiplication)
          - ronoise = RAMP_SIG * sqrt(12 / HXRGGRUP) (effective ramp noise)
          - darkcurr = 0.0 (dark current embedded in count rate)

        For raw files: returns the original parameters.
        """
        # Check if this is a ramp-fitted file
        is_ramp = False
        if hdu is not None:
            try:
                is_ramp = bool(hdu[1].header.get("RAMPFIT", False))
            except (IndexError, KeyError):
                pass

        if is_ramp:
            h1 = hdu[1].header
            ramp_sig = h1.get("RAMP_SIG", 9.5)
            ngroups = h1.get("HXRGGRUP", 69)
            eff_ronoise = ramp_sig * np.sqrt(12.0 / ngroups)

            detector_dict = dict(
                binning="1,1",
                det=1,
                dataext=1,
                specaxis=0,
                specflip=False,
                spatflip=False,
                platescale=0.2012,
                darkcurr=0.0,
                saturation=700000.0,
                nonlinear=1.0,
                mincounts=-1e10,
                numamplifiers=1,
                gain=np.atleast_1d(1.0),
                ronoise=np.atleast_1d(eff_ronoise),
                datasec=np.atleast_1d("[:,:]"),
                oscansec=None,
            )
            return detector_container.DetectorContainer(**detector_dict)
        else:
            return super().get_detector_par(det, hdu=hdu)

    def get_rawimage(self, raw_file, det):
        """
        Read a raw or ramp-fitted MMIRS image.

        For ramp-fitted files (RAMPFIT=True in ext 1):
          - Reads SCI (ext 1) directly as float64
          - Multiplies by EXPTIME to convert electron/s -> total electrons
          - Applies flipud for orientation consistency
          - Skips CDS and reference pixel correction (already done)

        For raw files: delegates to the original implementation.
        """
        fil = utils.find_single_file(f"{raw_file}*", required=True)

        log.info(f"Reading MMIRS file: {fil}")
        hdu = io.fits_open(fil)
        head1 = hdu[1].header

        # Check if ramp-fitted
        is_ramp = bool(head1.get("RAMPFIT", False))

        if is_ramp:
            detector_par = self.get_detector_par(det if det is not None else 1, hdu=hdu)

            # Read the SCI extension directly
            data = hdu[1].data.astype(np.float64)

            # Convert electron/s to total electrons
            exptime = head1["EXPTIME"]
            data *= exptime
            log.info(
                f"Ramp-fitted file: multiplied by EXPTIME={exptime:.3f}s "
                f"to get total electrons"
            )

            # Apply the zJ+HK cut if applicable
            binning = head1["CCDSUM"]
            xbin, ybin = [int(ibin) for ibin in binning.split(" ")]
            if (head1.get("FILTER") == "zJ") and (head1.get("DISPERSE") == "HK"):
                data = data[: int(998 / ybin), :]

            rawdatasec_img = np.ones_like(data, dtype="int")
            oscansec_img = np.zeros_like(data, dtype="int")

            return (
                detector_par,
                np.flipud(data),
                hdu,
                exptime,
                np.flipud(rawdatasec_img),
                np.flipud(oscansec_img),
            )

        else:
            # Fall back to original CDS + refpix logic
            return super().get_rawimage(raw_file, det)
