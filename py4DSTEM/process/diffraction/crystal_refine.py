import numpy as np
from scipy import linalg
from typing import Union, Optional, List
from time import time
from tqdm import tqdm
from ..utils import tqdmnd
import matplotlib.pyplot as plt

from ...io.datastructure import PointList, PointListArray, DataCube


def estimate_thickness(
    self,
    bragg_peaks: PointList,
    orientation: np.ndarray,
    bloch_beams: PointList,
    thickness: np.ndarray,
    min_peaks: int = 4,
    ax=None,
) -> float:
    """
    Estimate thickness of diffraction pattern encoded in ``bragg_peaks`` by computing
    a thickness series of dynamical patters using ``bloch_beams`` and comparing
    relative intensities.

    Args:
        bragg_peaks (PointList):        experimental measured disk intensities, with hkl indices
        bloch_beams (PointList):        beams to include in the Bloch wave dynamical diffraction
                                        calculation
        thickness (ndarray):            Array of thickness values to compare against
    """

    if np.atleast_1d(bragg_peaks.data).shape[0] < min_peaks:
        return 0.0

    ZA = orientation[
        :, 2
    ]  # this should be the ZA component of the orientation matrix ??

    bloch = self.generate_dynamical_diffraction_pattern(
        bloch_beams,
        thickness=thickness,
        zone_axis=ZA,
        always_return_list=True,
    )

    # normalize each Bloch wave pattern to the direct beam intensity
    zerobeam = [0, 0, 0]
    pld = bloch[0].data
    idx = np.argwhere(
        np.atleast_1d(
            np.logical_and(
                np.logical_and(pld["h"] == zerobeam[0], pld["k"] == zerobeam[1]),
                pld["l"] == zerobeam[2],
            )
        )
    )[0][0]
    for b in bloch:
        b.data["intensity"] /= b.data["intensity"][idx]

    # get indices that match beams in bragg_peaks to beams in bloch_beams
    hkl_bragg = np.vstack(
        (bragg_peaks.data["h"], bragg_peaks.data["k"], bragg_peaks.data["l"])
    ).T
    hkl_bloch = np.vstack(
        (bloch_beams.data["h"], bloch_beams.data["k"], bloch_beams.data["l"])
    ).T
    a, b = np.mgrid[0 : hkl_bragg.shape[0], 0 : hkl_bloch.shape[0]]
    # matches contains two arrays, one with the incdices into bragg_beams and one with indices into bloch_beams,
    # which correspond to slices that pair up the common beams correctly
    matches = np.nonzero(
        np.all(
            (hkl_bragg[a.ravel(), :] == hkl_bloch[b.ravel(), :]).reshape(
                a.shape + (3,)
            ),
            axis=2,
        )
    )

    # To penalize disks missing in the experiment, copy the Bloch beam list and, zero the intensities,
    # and copy back the experimental ones where they were found
    bragg_peaks_all = bloch_beams.copy()
    bragg_peaks_all.data["intensity"] = 0.0
    bragg_peaks_all.data["intensity"][matches[1]] = bragg_peaks.data["intensity"][
        matches[0]
    ]

    # cost function: takes two structures arrays with (qx,qy,I,h,k,l) and returns a floating point number for the score
    def cost_function(bps, bbs):
        # return np.sum( (np.sqrt(bps['intensity']) - np.sqrt(bbs['intensity'])) )
        # return np.sum( bps['intensity'] * bbs['intensity'] )
        return np.sum(np.abs(bps["intensity"] - bbs["intensity"]))
        # return (np.sum( bps['intensity'] * bbs['intensity'] ) - 1) / (np.sum(bbs['intensity']) - 1)
        # return np.sum( bps['intensity'] * np.sqrt(bbs['intensity'] )) / (np.sum(np.sqrt(bbs['intensity'])))

    # scores = np.array([cost_function(bragg_peaks.data[matches[0]], bbs.data[matches[1]]) for bbs in bloch])
    scores = np.array([cost_function(bragg_peaks_all.data, bbs.data) for bbs in bloch])

    if ax is not None:
        ax[0].plot(thickness, scores)
        bt = np.vstack([bb.data[matches[1]]["intensity"] for bb in bloch]).T
        for b in bt:
            ax[1].plot(thickness, np.sqrt(b))
        for b, c in zip(
            bragg_peaks.data[matches[0]]["intensity"], plt.rcParams["axes.prop_cycle"]
        ):
            ax[1].axhline(np.sqrt(b), c=c["color"])

    return thickness[np.nanargmin(scores)]


def estimate_thickness_tilt(
    self,
    bragg_peaks: PointList,
    orientation: np.ndarray,
    bloch_beams: PointList,
    thickness: np.ndarray,
    min_peaks: int = 4,
    tilt_refine_range=18.0,
    tilt_refine_step_size_inv_A=0.25,
    normalize_to_direct_beam=True,
    ax=None,
) -> float:
    """
    Estimate thickness of diffraction pattern encoded in ``bragg_peaks`` by computing
    a thickness series of dynamical patters using ``bloch_beams`` and comparing
    relative intensities.

    Args:
        bragg_peaks (PointList):        experimental measured disk intensities, with hkl indices
        bloch_beams (PointList):        beams to include in the Bloch wave dynamical diffraction
                                        calculation
        thickness (ndarray):            Array of thickness values to compare against
    """

    # get indices that match beams in bragg_peaks to beams in bloch_beams
    hkl_bragg = np.vstack(
        (bragg_peaks.data["h"], bragg_peaks.data["k"], bragg_peaks.data["l"])
    ).T
    hkl_bloch = np.vstack(
        (bloch_beams.data["h"], bloch_beams.data["k"], bloch_beams.data["l"])
    ).T
    a, b = np.mgrid[0 : hkl_bragg.shape[0], 0 : hkl_bloch.shape[0]]
    # matches contains two arrays, one with the incdices into bragg_beams and one with indices into bloch_beams,
    # which correspond to slices that pair up the common beams correctly
    matches = np.nonzero(
        np.all(
            (hkl_bragg[a.ravel(), :] == hkl_bloch[b.ravel(), :]).reshape(
                a.shape + (3,)
            ),
            axis=2,
        )
    )

    if matches[0].shape[0] < min_peaks:
        return 0.0, [0, 0, 0]

    ZA = orientation[
        :, 2
    ]  # this should be the ZA component of the orientation matrix ??

    bloch, CBED_ZA = self.generate_CBED(
        beams=bloch_beams,
        thickness=thickness,
        alpha_mrad=tilt_refine_range,
        pixel_size_inv_A=tilt_refine_step_size_inv_A,
        zone_axis=ZA,
        LACBED=True,
        verbose=False,
        progress_bar=False,
        return_ZA=True,
    )

    # normalize each Bloch wave pattern to the direct beam intensity
    if normalize_to_direct_beam:
        for ts in bloch:  # loop over thicknesses
            for b in ts.values():  # loop over LACBED disks
                b /= ts[(0, 0, 0)]

    # To penalize disks missing in the experiment, copy the Bloch beam list and, zero the intensities,
    # and copy back the experimental ones where they were found
    bragg_peaks_all = bloch_beams.copy()
    bragg_peaks_all.data["intensity"] = 0.0
    bragg_peaks_all.data["intensity"][matches[1]] = bragg_peaks.data["intensity"][
        matches[0]
    ]

    # Generate an array like the flattened LACBED dict that has the experimental intensities in it
    bragg_peaks_lacbed = np.zeros(bloch[0][(0, 0, 0)].shape + (len(bloch[0].keys()),))
    bragg_peaks_lacbed[:, :, matches[1]] = bragg_peaks.data[matches[0]]["intensity"]

    def cost_function(bps, bbs):
        # return np.sum( (np.sqrt(bps['intensity']) - np.sqrt(bbs['intensity'])) )
        # return np.sum( bps['intensity'] * bbs['intensity'] )
        return np.sum(np.abs(bps - bbs), axis=2)
        # return (np.sum( bps['intensity'] * bbs['intensity'] ) - 1) / (np.sum(bbs['intensity']) - 1)
        # return np.sum( bps['intensity'] * np.sqrt(bbs['intensity'] )) / (np.sum(np.sqrt(bbs['intensity'])))

    scores = np.array(
        [
            cost_function(bragg_peaks_lacbed, np.dstack([arr for arr in bbs.values()]))
            for bbs in bloch
        ]
    )
    # print(scores.shape)

    if ax is not None:
        ax.plot(thickness, np.nanmin(scores, axis=(1, 2)))

    # plotting

    idx = np.unravel_index(np.nanargmin(scores), scores.shape)
    tZA = CBED_ZA[idx[1:3]]
    # print(idx, scores.shape)

    return thickness[idx[0]], tZA


def index_Bragg_peaks_from_orientation(
    self,
    bragg_peaks: PointList,
    orientation: np.ndarray,
    tol_distance: float = 0.05,
    sigma_excitation_error: float = 0.02,
    tol_excitation_error_mult: float = 3,
    tol_intensity: float = 0.1,
    k_max: float = None,
) -> PointList:
    """
    Given a set of experimental Bragg disk locations and the orientation matrix
    computed in ``match_single_pattern``, select the experimental peaks that are
    within ``tol_distance`` of the kinematically predicted ones and return a
    new PointList containing (qx,qy,Intensity,h,k,l) for the matching peaks.

    Args:
        bragg_peaks:        (PointList) peaks to index, with (qx,qy,intensity) fields
        orientation:        (tuple/array) orientation to generate comparison peaks from.
                                Can be 3-element for a zone axis or [3x3] matrix to include
                                in-plane rotation
        tol_distance        (float) distance threshold from ideal peaks to index an experimental peak
        The remaining args are passed on to Crystal.generate_diffraction_pattern:
            sigma_excitation_error
            tol_excitation_error_mult
            tol_intensity
            k_max

    Note that to index kinematically forbidden peaks present in a pattern, you will
    likely need to set tol_intensity=0

    """
    match_dtype = np.dtype(
        [
            ("qx", np.float64),
            ("qy", np.float64),
            ("intensity", np.float64),
            ("h", np.int64),
            ("k", np.int64),
            ("l", np.int64),
        ]
    )

    sim_peaks = self.generate_diffraction_pattern(
        zone_axis=orientation,
        sigma_excitation_error=sigma_excitation_error,
        tol_excitation_error_mult=tol_excitation_error_mult,
        tol_intensity=tol_intensity,
        k_max=k_max,
    )

    if sim_peaks.length == 0:
        print("Warning! No kinematic peaks found!")
        return PointList(match_dtype)

    # Accumulate matches as a list of len-1 arrays, then concatenate later
    # TODO: do this a smarter way
    matches = []
    # loop over all experimental peaks
    for i in range(bragg_peaks.length):
        # get current peak
        qx, qy = bragg_peaks.data["qx"][i], bragg_peaks.data["qy"][i]

        # get find closest peak, and check if it's within tol_distance
        dq = np.hypot(sim_peaks.data["qx"] - qx, sim_peaks.data["qy"] - qy)
        if np.min(dq) < tol_distance:
            idx = np.argmin(dq)
            matches.append(
                np.array(
                    [
                        (
                            sim_peaks.data["qx"][idx],
                            sim_peaks.data["qy"][idx],
                            bragg_peaks.data["intensity"][i],
                            sim_peaks.data["h"][idx],
                            sim_peaks.data["k"][idx],
                            sim_peaks.data["l"][idx],
                        )
                    ],
                    dtype=match_dtype,
                )
            )

    if len(matches) > 0:
        return PointList(match_dtype, np.squeeze(np.array(matches))), sim_peaks
    else:
        return PointList(match_dtype), sim_peaks


def measure_disk_intensities(
    datacube: DataCube,
    braggpeaks: PointListArray,
    disk_radius: float,
    background_window_size: int,
    background_type: str = "mean",
):
    """
    Update the "intensity" field of the PointListArray ``braggpeaks`` by
    fitting a background function to the region around each identified
    Bragg disk and subtracting it from the integrated intensity.

    Background interpolation methods:
        'mean': Mean value inside the background region
        'median': Median value inside background region
        'plane': Plane fit to background region
    """

    assert background_type in (
        "mean",
        "median",
        "plane",
    ), f"Unrecognized background mode {background_type}"

    # precompute some masks
    qx, qy = np.mgrid[
        -background_window_size : background_window_size + 1,
        -background_window_size : background_window_size + 1,
    ]
    r = np.hypot(qx, qy)
    disk_mask = r < disk_radius
    disk_area = np.sum(disk_mask * 1.0)

    window_mask = ~disk_mask
    bg_coords = np.vstack(
        (
            qx[window_mask].ravel(),
            qy[window_mask].ravel(),
            np.ones_like(qx)[window_mask].ravel(),
        )
    ).T

    window_coords = np.dstack((qx, qy, np.ones_like(qx)))

    for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny):
        pla = braggpeaks.get_pointlist(rx, ry)

        for disk in pla.data:
            # Get the image of the disk
            centx, centy = int(disk["qx"]), int(disk["qy"])

            # check the edge boundary
            if (
                (centx - background_window_size < 0)
                or (centx + background_window_size + 1 > datacube.Q_Nx)
                or (centy - background_window_size < 0)
                or (centy + background_window_size + 1 > datacube.Q_Ny)
            ):
                disk["intensity"] = 0
                continue

            diskimg = datacube.data[
                rx,
                ry,
                centx - background_window_size : centx + background_window_size + 1,
                centy - background_window_size : centy + background_window_size + 1,
            ]

            if background_type == "mean":
                background = np.mean(diskimg[window_mask])
                disk["intensity"] = np.sum(diskimg[disk_mask] - background)

            elif background_type == "median":
                background = np.median(diskimg[window_mask])
                disk["intensity"] = np.sum(diskimg[disk_mask] - background)

            elif background_type == "plane":
                bg_int = diskimg[window_mask].ravel()
                plane_coeffs = np.linalg.lstsq(bg_coords, bg_int, rcond=None)[0]
                bg_fit = window_coords @ plane_coeffs
                disk["intensity"] = np.sum((diskimg - bg_fit)[disk_mask])

            # if (rx,ry) == (60,60):
            #     from pdb import set_trace
            #     set_trace()

    return braggpeaks
