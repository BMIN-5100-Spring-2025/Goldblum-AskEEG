import re
import scipy as sc
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt

# UTILITY FUNCTIONS
## FEATURE EXTRACTION
def num_wins(xLen, fs, winLen, winDisp):
    return int(
        ((xLen / fs - winLen + winDisp) - ((xLen / fs - winLen + winDisp) % winDisp))
        / winDisp
    )


def MovingWinClips(x, fs, winLen, winDisp):
    # calculate number of windows and initialize receiver
    nWins = num_wins(len(x), fs, winLen, winDisp)
    samples = np.empty((nWins, int(winLen * fs)))
    # create window indices - these windows are left aligned
    idxs = np.array(
        [(winDisp * fs * i, (winLen + winDisp * i) * fs) for i in range(nWins)]
    ).astype(int)
    # apply feature function to each channel
    for i in range(idxs.shape[0]):
        samples[i, :] = x[idxs[i, 0] : idxs[i, 1]]

    return samples


## ————————————————CHANNEL PREPROCESSING————————————————


def clean_labels(channel_li) -> np.ndarray:
    """
    Clean and standardize channel labels.

    Parameters:
    - channel_li (Union[Iterable[str], str]): Either a single channel label or an iterable of channel labels.

    Returns:
    - np.ndarray: An array of cleaned and standardized channel labels.

    Example:
    >>> clean_labels('LA 01')
    array(['LA1'])
    """
    if isinstance(channel_li, str):
        channel_li = [channel_li]
    new_channels = []
    for i in range(len(channel_li)):
        # standardizes channel names
        label_num_search = re.search(r"\d", channel_li[i])
        if label_num_search is not None:
            label_num_idx = label_num_search.start()
            label_non_num = channel_li[i][:label_num_idx]
            label_num = channel_li[i][label_num_idx:]
            label_num = label_num.lstrip("0")
            label = label_non_num + label_num
        else:
            label = channel_li[i]
        label = label.replace("EEG", "")
        label = label.replace("Ref", "")
        label = label.replace(" ", "")
        label = label.replace("-", "")
        label = label.replace("CAR", "")
        label = label.replace("HIPP", "DH")
        label = label.replace("AMY", "DA")
        label = label.replace("FP", "Fp")
        label = label.replace("FZ", "Fz")
        label = label.replace("PZ", "Pz")
        label = label.replace("PZ", "Pz")
        label = label.replace("FPz", "Fpz")
        label = label.replace("FPZ", "Fpz")
        label = "T3" if label == "T7" else label
        label = "T4" if label == "T8" else label
        label = "T5" if label == "P7" else label
        label = "T6" if label == "P8" else label
        new_channels.append(label)
    return np.array(new_channels)


def check_channel_type(channel_li) -> np.ndarray:
    """
    Find non-iEEG channel labels.

    Parameters:
    - channel_li (Union[Iterable[str], str]): Either an iterable of channel labels or a single channel label.

    Returns:
    - np.ndarray: Boolean array indicating whether each channel is non-iEEG.
    """
    scalp = [
        "O",
        "C",
        "CZ",
        "F",
        "FP",
        "FZ",
        "T",
        "P",
        "PZ",
        "FPZ",
    ]
    ekg = ["EKG", "ECG"]
    emg = ["EMG"]
    eog = ["LOC", "ROC"]
    other = ["RATE"]

    if isinstance(channel_li, str):
        channel_li = [channel_li]
    ch_df = []
    for i in channel_li:
        regex_match = re.search(r"\d", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0})
            continue
        label_num_idx = regex_match.start()
        label_non_num = i[:label_num_idx]
        label_num = i[label_num_idx:]
        ch_df.append({"name": i, "lead": label_non_num, "contact": label_num})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead.upper() in ekg:
            ch_df.loc[group.index, "type"] = "ekg"
            continue
        if lead.upper() in scalp:
            ch_df.loc[group.index, "type"] = "eeg"
            if i == "O1" or i == "O2":
                if (
                    channel_li.count("O3") == 1 or channel_li.count("O4") == 1
                ):  # if intracranial, should have these too
                    ch_df.loc[group.index, "type"] = "ieeg"
            continue
        if lead.upper() in emg:
            ch_df.loc[group.index, "type"] = "emg"
            continue
        if lead.upper() in eog:
            ch_df.loc[group.index, "type"] = "eog"
            continue
        if lead.upper() in other:
            ch_df.loc[group.index, "type"] = "misc"
            continue
        if len(group) > 16:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df["type"].to_numpy()


def detect_bad_channels_eeg(data, fs):
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 400
    abs_thresh2 = 500

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs), 1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):
        ich = which_chs[i]
        eeg = values[:, ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)

        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if sum(np.diff(eeg, 1) == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            flat_ch.append(ich)
            continue

        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) + sum(abs(eeg) > abs_thresh2) > 0.1 * len(
            eeg
        ):
            bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            bad.append(ich)
            high_var_ch.append(ich)
            continue

        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg - np.nanmean(eeg))

        # Get power
        P = abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]

        # Take first half
        P = P[: np.ceil(len(P) / 2).astype(int)]
        freqs = freqs[: np.ceil(len(freqs) / 2).astype(int)]

        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)]) / sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std

    channel_mask = np.ones((values.shape[1],), dtype=bool)
    channel_mask[bad] = False
    details["noisy"] = noisy_ch
    details["nans"] = nan_ch
    details["zeros"] = zero_ch
    details["flat"] = flat_ch
    details["var"] = high_var_ch
    details["higher_std"] = bad_std
    details["high_voltage"] = high_ch

    return channel_mask, details


def detect_bad_channels(data, fs, lf_stim=False):
    """
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    """
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs), 1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):
        ich = which_chs[i]
        eeg = values[:, ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)

        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        ## Remove channels with zeros in more than half
        if sum(eeg < 1) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if sum(np.diff(eeg, 1) <= 1e-3) > (0.5 * len(eeg)):
            bad.append(ich)
            flat_ch.append(ich)

        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue

        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg - np.nanmean(eeg))

        # Get power
        P = abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]

        # Take first half
        P = P[: np.ceil(len(P) / 2).astype(int)]
        freqs = freqs[: np.ceil(len(freqs) / 2).astype(int)]

        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)]) / sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std
    # for ch in bad_std:
    #     if ch not in bad:
    #         if ~lf_stim:
    #             bad.append(ch)
    channel_mask = np.ones((values.shape[1],), dtype=bool)
    channel_mask[bad] = False
    details["noisy"] = noisy_ch
    details["nans"] = nan_ch
    details["zeros"] = zero_ch
    details["flat"] = flat_ch
    details["var"] = high_var_ch
    details["higher_std"] = bad_std
    details["high_voltage"] = high_ch

    return channel_mask, details


def bipolar_montage_ieeg(ch_list) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_
        ch_types (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    bipolar_labels = []
    bipolar_idx = []
    for i, ch in enumerate(ch_list):
        ch1Ind = i
        label_num_search = re.search(r"\d", ch)
        if label_num_search is not None:
            label_num_idx = label_num_search.start()
            label_non_num = ch[:label_num_idx]
            label_num = int(ch[label_num_idx:])
            ch2_num = label_num + 1
            ch2 = label_non_num + f"{ch2_num}"
            ch2exists = np.where(ch_list == ch2)[0]
            if len(ch2exists) > 0:
                ch2Ind = ch2exists[0]
            else:
                ch2Ind = np.nan
            ch3_num = label_num + 2
            ch3 = label_non_num + f"{ch3_num}"
            ch3exists = np.where(ch_list == ch3)[0]
            if len(ch3exists) > 0:
                ch3Ind = ch3exists[0]
            else:
                ch3Ind = np.nan
            if np.isnan(ch2Ind):
                if not np.isnan(ch3Ind):
                    bipolar_idx.append([ch1Ind, ch3Ind, np.nan])
                    bipolar_labels.append(ch + "-" + ch3)
            else:
                bipolar_idx.append([ch1Ind, ch2Ind, ch3Ind])
                bipolar_labels.append(ch + "-" + ch2)
    return np.array(bipolar_labels), np.array(bipolar_idx)


def bipolar_montage_eeg(ch_list):

    ch_dict = {ch: i for i, ch in enumerate(ch_list)}
    ch1 = [
        "Fp1",
        "F7",
        "T3",
        "T5",
        "Fp2",
        "F8",
        "T4",
        "T6",
        "Fp1",
        "F3",
        "C3",
        "P3",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "Fz",
    ]
    ch2 = [
        "F7",
        "T3",
        "T5",
        "O1",
        "F8",
        "T4",
        "T6",
        "O2",
        "F3",
        "C3",
        "P3",
        "O1",
        "F4",
        "C4",
        "P4",
        "O1",
        "Cz",
    ]
    ch1_index = [ch_dict.get(ch, np.nan) for ch in ch1]
    ch2_index = [ch_dict.get(ch, np.nan) for ch in ch2]
    bipolar_index = np.array([ch1_index, ch2_index]).T
    bipolar_labels = [f"{ch1[i]}-{ch2[i]}" for i in range(len(ch1))]
    nan_mask = np.any(np.isnan(bipolar_index), axis=1)
    return np.array(bipolar_labels)[~nan_mask], bipolar_index[~nan_mask]


def car(data):
    """
    Perform Common Average Reference (CAR) on the input iEEG data.
    """
    out_data = data - np.nanmean(data, 1)[:, np.newaxis]
    return out_data


def bipolar(data, bipolar_index):
    out_data = data[:, bipolar_index[:, 0]] - data[:, bipolar_index[:, 1]]
    return out_data


## ————————————SIGNAL PREPROCESSING——————————————


def downsample(data, fs, target):
    signal_len = int(data.shape[0] / fs * target)
    data_bpd = sc.signal.resample(data, signal_len, axis=0)
    return data_bpd


# def interp_resample(time,data,target):
#     signal_len = int(data.shape[0]/fs*target)
#     data_bpd = sc.signal.resample(data,signal_len,axis=0)
#     return data_bpd,target


def notch_filter(data: np.ndarray, fs: float) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    b, a = butter(4, (58, 62), "bandstop", fs=fs)
    d, c = butter(4, (118, 122), "bandstop", fs=fs)

    data_filt = filtfilt(b, a, data, axis=0)
    data_filt_filt = filtfilt(d, c, data_filt, axis=0)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt_filt


def bandpass_filter(data: np.ndarray, fs: float, order=3, lo=1, hi=150) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        order (int, optional): _description_. Defaults to 3.
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.

    Returns:
        np.array: _description_
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    data_filt = sosfiltfilt(sos, data, axis=0)
    return data_filt


def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp - 1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp - 1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i] * w[0] + w[1])
    return data_white


from sklearn.linear_model import LinearRegression


def pre_whiten(data: np.ndarray) -> np.ndarray:
    """Pre-whiten the input data using linear regression.

    Args:
        data (np.ndarray): Matrix representing data. Each column is a channel, and each row is a time point.

    Returns:
        np.ndarray: Pre-whitened data matrix.
    """
    prewhite_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        vals = data[:, i].reshape(-1, 1)
        if np.sum(~np.isnan(vals)) == 0:
            continue
        model = LinearRegression().fit(vals[:-1, :], vals[1:, :])
        E = model.predict(vals[:-1, :]) - vals[1:, :]
        if len(E) < len(vals):
            E = np.concatenate([E, E[-1] * np.zeros([len(vals) - len(E), 1])])
        prewhite_data[:, i] = E.reshape(-1)

    return prewhite_data


# ——————————————————————somewhat feature related——————————————————————————————-
from scipy.signal import welch
from scipy.integrate import simpson


def bandpower(
    data,
    fs,
    band,
    win_size=None,
    relative=False,
) -> np.ndarray:
    """Adapted from https://raphaelvallat.com/bandpower.html
    Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array or 2d-array
        Input signal in the time-domain. (time by channels)
    fs : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    win_size : float
        Length of each window in seconds.
        If None, win_size = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : np.ndarray
        Absolute or relative band power. channels by bands
    """
    band = np.asarray(band)
    assert len(band) == 2, "CNTtools:invalidBandRange"
    assert band[0] < band[1], "CNTtools:invalidBandRange"
    if np.ndim(data) == 1:
        data = data[:, np.newaxis]
    nchan = data.shape[1]
    bp = np.nan * np.zeros(nchan)
    low, high = band

    # Define window length
    # if win_size is not None:
    #     nperseg = int(win_size * fs)
    # else:
    #     nperseg = int((2 / low) * fs)

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data.T, fs)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    if psd.ndim == 2:
        bp = simpson(psd[:, idx_band], dx=freq_res)
    elif psd.ndim == 1:
        bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)

    return bp
