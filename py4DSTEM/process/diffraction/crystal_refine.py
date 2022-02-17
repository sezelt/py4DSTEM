import numpy as np
from scipy import linalg
from typing import Union, Optional
from time import time
from tqdm import tqdm

from ...io.datastructure import PointList


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
        return 0.
    
    ZA = orientation[:,2] # this should be the ZA component of the orientation matrix ??
    
    bloch = self.generate_dynamical_diffraction_pattern(bloch_beams,
                                                        thickness=thickness,
                                                        zone_axis=ZA,
                                                        always_return_list=True,
                                                       )
    
    # normalize each Bloch wave pattern to the direct beam intensity
    zerobeam = [0, 0, 0]
    pld = bloch[0].data
    idx = np.argwhere(
        np.atleast_1d(np.logical_and(
            np.logical_and(pld["h"] == zerobeam[0], pld["k"] == zerobeam[1]),
            pld["l"] == zerobeam[2],
        ))
    )[0][0]
    for b in bloch:
        b.data['intensity'] /= b.data['intensity'][idx]
    
    # get indices that match beams in bragg_peaks to beams in bloch_beams
    hkl_bragg = np.vstack((bragg_peaks.data['h'],bragg_peaks.data['k'],bragg_peaks.data['l'])).T
    hkl_bloch = np.vstack((bloch_beams.data['h'],bloch_beams.data['k'],bloch_beams.data['l'])).T
    a,b = np.mgrid[0:hkl_bragg.shape[0],0:hkl_bloch.shape[0]]
    # matches contains two arrays, one with the incdices into bragg_beams and one with indices into bloch_beams,
    # which correspond to slices that pair up the common beams correctly
    matches = np.nonzero(np.all((hkl_bragg[a.ravel(),:] == hkl_bloch[b.ravel(),:]).reshape(a.shape + (3,)),axis=2))
    
    # cost function: takes two structures arrays with (qx,qy,I,h,k,l) and returns a floating point number for the score
    def cost_function(bps, bbs):
        return np.sum(  np.sqrt((bps['intensity'] - np.sqrt(bbs['intensity']))**2) * np.sqrt(bps['intensity']))
        # return (np.sum( bps['intensity'] * bbs['intensity'] ) - 1) / (np.sum(bbs['intensity']) - 1)
        # return np.sum( bps['intensity'] * np.sqrt(bbs['intensity'] )) / (np.sum(np.sqrt(bbs['intensity'])))
    
    scores = np.array([cost_function(bragg_peaks.data[matches[0]], bbs.data[matches[1]]) for bbs in bloch])
    
    # if ax is not None:
    #     ax[0].plot(thickness,scores)
    #     bt = np.vstack([bb.data[matches[1]]['intensity'] for bb in bloch]).T
    #     for b in bt:
    #         ax[1].plot(thickness,np.sqrt(b))
    #     for b,c in zip(bragg_peaks.data[matches[0]]['intensity'],plt.rcParams['axes.prop_cycle']):
    #         ax[1].axhline(b,c=c['color'])
    
    
    return thickness[np.nanargmin(scores)]


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

    return PointList(match_dtype, np.squeeze(np.array(matches))), sim_peaks
