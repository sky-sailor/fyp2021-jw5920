"""
Signal Quality Index (SQI) Calculation Functions
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import numpy as np
import vital_sqi.sqi as sq
import pandas as pd
from scipy.stats import skew, kurtosis


def snr(x, axis=0, ddof=0):
    """Signal-to-noise ratio"""
    a = np.asanyarray(x)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def zcr(x):
    """Zero crossing rate"""
    return 0.5 * np.mean(np.abs(np.diff(np.sign(x))))


def mcr(x):
    """Mean crossing rate"""
    return zcr(x - np.mean(x))


def perfusion(x, y):
    """Perfusion
    x: raw signal
    y: filtered signal
    """
    return (np.max(y) - np.min(y)) / np.abs(np.mean(x))  # * 100 changed here # many inf results


def correlogram(x):
    """Correlogram"""
    return sq.rpeaks_sqi.correlogram_sqi(x)


def msq(x):
    """Mean signal quality"""
    # Library
    from vital_sqi.common.rpeak_detection import PeakDetector
    # Detection of peaks
    detector = PeakDetector()
    peak_list, trough_list = detector.ppg_detector(x, 7)
    # Return
    return sq.standard_sqi.msq_sqi(x, peaks_1=peak_list, peak_detect2=6)


# def dtw(x):
#     """Dynamic time warping
#     . note: It is very slow!!
#     Returns [mean, std]
#     """
#     # Library
#     from vital_sqi.common.rpeak_detection import PeakDetector
#     # Detection of peaks
#     detector = PeakDetector()
#     peak_list, trough_list = detector.ppg_detector(x, 7)
#     # Per beats
#     dtw_list = sq.standard_sqi.per_beat_sqi( \
#         sqi_func=sq.dtw_sqi, troughs=trough_list,
#         signal=x, taper=True, template_type=1
#     )
#     # Return mean
#     return [np.mean(dtw_list),
#             np.std(dtw_list)]


def sqi_all(x):
    """Compute all SQIs"""
    # Information
    dinfo = {
        'first': x.idx.iloc[0],
        'last': x.idx.iloc[-1],
        'skew': skew(x['PLETH']),
        'kurtosis': kurtosis(x['PLETH']),
        'snr': snr(x['PLETH']),
        'mcr': mcr(x['PLETH']),
        'zcr': zcr(x['PLETH_BPF']),
        'msq': msq(x['PLETH_BPF']),
        'perfusion': perfusion(x['PLETH'], x['PLETH_BPF']),
        'correlogram': correlogram(x['PLETH_BPF']),
        # 'dtw': dtw(x['0_bbpf'])
    }

    # Return
    return pd.Series(dinfo)
