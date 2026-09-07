"""VDA (Visually-Dependent Accuracy) -- the harmonic mean of the intact
accuracy (Acc(C0)) and the accuracy drop caused by removing the image
(VDD against C1).

It is proposed to answer, with a single number, the observation that "being
correct matters, but being correct through genuine dependence on the image
matters even more in medical imaging". It follows the same spirit as F1,
combining two dimensions that only pay off together: a shortcut-prone model
(high accuracy, low VDD) is penalised, and so is a poor but honest model
(high VDD, low accuracy). Because it is a harmonic mean, a model must score
well on both dimensions -- a near-zero value on either one drags the whole
score down.

VDA is computed once over the aggregated means, never as an average of
per-seed VDA values."""
from __future__ import annotations


def compute_vda(acc_c0: float, ldv_c1: float) -> float | None:
    """Returns None when the harmonic mean is undefined
    (acc_c0 + ldv_c1 <= 0) -- not expected for the data in this study (VDD
    against C1 was positive and significant for all 11 models, without
    exception), but guarded defensively in case a model with a negative
    VDD(C1) appears in the future. Negative VDD values have already been
    observed in the summary matrix for secondary C4 conditions (mild
    occlusion/noise levels where accuracy can drift slightly upward by
    chance), even though never for C1.

    The argument is named ldv_c1 because "ldv" is the historical JSON key
    that holds the VDD value; it is VDD(C1) = Acc(C0) - Acc(C1)."""
    denom = acc_c0 + ldv_c1
    if denom <= 0:
        return None
    return 2 * acc_c0 * ldv_c1 / denom
