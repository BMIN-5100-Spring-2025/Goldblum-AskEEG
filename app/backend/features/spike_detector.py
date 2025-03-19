import numpy as np
import concurrent.futures
from scipy import signal as sig


def eeg_filter(signal, fc, filttype, fs):
    """
    Filters an EEG signal using a Butterworth filter.

    Parameters:
    - signal (array-like): The EEG signal to be filtered. This should be a one-dimensional array of values.
    - fc (float): The cutoff frequency for the filter in Hz. Must be less than half the sampling rate (fs/2).
    - filttype (str): The type of the filter. Acceptable values are 'low', 'high', 'bandpass', and 'bandstop'.
    - fs (float): The sampling rate of the EEG signal in Hz.

    Returns:
    - array-like: The filtered EEG signal.

    Raises:
    - AssertionError: If the cutoff frequency is greater than or equal to half the sampling rate.

    Note:
    - This function uses the Butterworth filter from the `scipy.signal` module.
    """
    assert fc < fs / 2, "Cutoff frequency must be < one half the sampling rate"
    # Initialize constants for filter
    order = 6
    # Create and apply filter
    B, A = sig.butter(order, fc, filttype, fs=fs)
    return sig.filtfilt(B, A, signal)


def find_peaks(signal):
    """
    Finds the peaks and troughs in a given signal.

    This function identifies the peaks and troughs of a signal by computing its first derivative
    and then determining where the derivative changes sign.

    Parameters:
    - signal (array-like): A one-dimensional array of numeric values representing the signal.

    Returns:
    - tuple: A tuple containing two arrays:
        1. An array of indices indicating the locations of the troughs.
        2. An array of indices indicating the locations of the peaks.

    Notes:
    - Peaks are regions where the signal increases and then decreases, while troughs are regions where the signal decreases and then increases.
    - This function may not capture all peaks or troughs if the signal has very small fluctuations or noise.

    Examples:
    >>> signal = [1, 3, 7, 6, 4, 5, 8, 6]
    >>> find_peaks(signal)
    (array([2]), array([6]))
    """
    ds = np.diff(signal, axis=0)
    ds = np.insert(ds, 0, ds[0])  # pad diff
    mask = np.argwhere(np.abs(ds[1:]) <= 1e-3).squeeze()  # got rid of +1
    ds[mask] = ds[mask - 1]
    ds = np.sign(ds)
    ds = np.diff(ds)
    ds = np.insert(ds, 0, ds[0])
    t = np.argwhere(ds > 0)
    p = np.argwhere(ds < 0)
    return p, t


def multi_channel_requirement(gdf, nchs, fs):
    # Need to change so that it returns total spike counter to help remove duplicates
    min_chs = 2
    if nchs < 16:
        max_chs = np.inf
    else:
        max_chs = np.ceil(nchs / 2)
    min_time = 100 * 1e-3 * fs

    # Check if there is even more than 1 spiking channel. Will throw error
    try:
        if len(np.unique(gdf[:, 1])) < min_chs:
            return np.array([])
    except IndexError:
        return np.array([])

    final_spikes = []

    s = 0  # start at time negative one for 0 based indexing
    curr_seq = [s]
    last_time = gdf[s, 0]
    spike_count = 0
    while s < (gdf.shape[0] - 1):  # check to see if we are at last spike
        # move to next spike time
        new_time = gdf[s + 1, 0]  # calculate the next spike time

        # if it's within the time diff
        if (
            new_time - last_time
        ) < min_time:  # check that the spikes are within the window of time
            curr_seq.append(s + 1)  # append it to the current sequence

            if s == (
                gdf.shape[0] - 2
            ):  # see if you just added the last spike, if so, done with sequence
                # done with sequence, check if the number of involved chs is
                # appropriate
                l = len(np.unique(gdf[curr_seq, 1]))
                if l >= min_chs and l <= max_chs:
                    final_spikes.append(
                        np.hstack(
                            (
                                gdf[curr_seq, :],
                                np.ones((len(curr_seq), 1)) * spike_count,
                            )
                        )
                    )
        else:
            # done with sequence, check if the length of sequence is
            # appropriate
            l = len(np.unique(gdf[curr_seq, 1]))
            if (l >= min_chs) & (l <= max_chs):
                final_spikes.append(
                    np.hstack(
                        (gdf[curr_seq, :], np.ones((len(curr_seq), 1)) * spike_count)
                    )
                )
                spike_count += 1
            # reset sequence
            curr_seq = [s + 1]
        # increase the last time
        last_time = gdf[s + 1, 0]

        # increase the current spike
        s += 1

    if len(final_spikes) != 0:
        return np.vstack(
            final_spikes
        )  # return all final spikes with a spike sequence counter
    else:
        return np.array([])


def multi_channel_requirement_with_label(gdf, nchs, fs):
    # Need to change so that it returns total spike counter to help remove duplicates
    min_chs = 2
    if nchs < 16:
        max_chs = np.inf
    else:
        max_chs = np.ceil(nchs / 2)
    min_time = 100 * 1e-3 * fs

    # Check if there is even more than 1 spiking channel. Will throw error
    try:
        if len(np.unique(gdf[:, 1])) < min_chs:  # check unique channels
            return np.array([])
    except IndexError:
        return np.array([])

    final_spikes = []

    s = 0  # start at time negative one for 0 based indexing
    curr_seq = [s]
    last_time = gdf[s, 0]
    spike_count = 0
    while s < (gdf.shape[0] - 1):  # check to see if we are at last spike
        # move to next spike time
        new_time = gdf[s + 1, 0]  # calculate the next spike time

        # if it's within the time diff
        if (
            new_time - last_time
        ) < min_time:  # check that the spikes are within the window of time
            curr_seq.append(s + 1)  # append it to the current sequence

            if s == (
                gdf.shape[0] - 2
            ):  # see if you just added the last spike, if so, done with sequence
                # done with sequence, check if the number of involved chs is
                # appropriate
                l = len(np.unique(gdf[curr_seq, 1]))
                if l >= min_chs and l <= max_chs:
                    final_spikes.append(
                        np.hstack(
                            (
                                gdf[curr_seq, :2],
                                np.ones((len(curr_seq), 1)) * spike_count,
                                gdf[curr_seq, 2:],
                            )
                        )
                    )
        else:
            # done with sequence, check if the length of sequence is
            # appropriate
            l = len(np.unique(gdf[curr_seq, 1]))
            if (l >= min_chs) & (l <= max_chs):
                final_spikes.append(
                    np.hstack(
                        (
                            gdf[curr_seq, :2],
                            np.ones((len(curr_seq), 1)) * spike_count,
                            gdf[curr_seq, 2:],
                        )
                    )
                )
                spike_count += 1
            # reset sequence
            curr_seq = [s + 1]
        # increase the last time
        last_time = gdf[s + 1, 0]

        # increase the current spike
        s += 1

    if len(final_spikes) != 0:
        return np.vstack(
            final_spikes
        )  # return all final spikes with a spike sequence counter
    else:
        return np.array([])


def process_channel(
    signal, fs, tmul, absthresh, sur_time, too_high_abs, spkdur, lpf1, hpf
):
    out = []  # initialize preliminary spike receiver

    if np.any(np.isnan(signal)):
        return None  # if there are any nans in the signal skip the channel (worth investigating)

    # Re-adjust the mean of the signal to be zero
    signal = signal - np.mean(signal)

    # receiver initialization
    spike_times = []
    spike_durs = []
    spike_amps = []

    # low pass filter to remove artifact
    lpsignal = eeg_filter(signal, lpf1, "lowpass", fs)

    # high pass filter for the 'spike' component
    hpsignal = eeg_filter(lpsignal, hpf, "highpass", fs)

    # defining thresholds
    lthresh = np.median(np.abs(hpsignal))
    thresh = lthresh * tmul  # this is the final threshold we want to impose

    signals = [hpsignal, -hpsignal]

    for ksignal in signals:
        # apply custom peak finder
        spp, spv = find_peaks(ksignal)  # calculate peaks and troughs
        spp, spv = spp.squeeze(), spv.squeeze()  # reformat

        # find the durations less than or equal to that of a spike
        idx = np.argwhere(np.diff(spp) <= spkdur[1]).squeeze()
        startdx = spp[idx]  # indices for each spike that has a long enough duration
        startdx1 = spp[idx + 1]  # indices for each "next" spike

        # Loop over peaks
        for i in range(len(startdx)):
            spkmintic = spv[(spv > startdx[i]) & (spv < startdx1[i])]
            # find the valley that is between the two peaks
            if not any(spkmintic):
                continue
            max_height = max(
                np.abs(ksignal[startdx1[i]] - ksignal[spkmintic]),
                np.abs(ksignal[startdx[i]] - ksignal[spkmintic]),
            )[0]
            if max_height > thresh:  # see if the peaks are big enough
                spike_times.append(int(spkmintic))  # add index to the spike list
                spike_durs.append(
                    (startdx1[i] - startdx[i]) * 1000 / fs
                )  # add spike duration to list
                spike_amps.append(max_height)  # add spike amplitude to list

    # Generate spikes matrix
    spikes = np.vstack([spike_times, spike_durs, spike_amps]).T

    # initialize exclusion receivers
    toosmall = []
    toosharp = []
    toobig = []

    # now have all the info we need to decide if this thing is a spike or not.
    for i in range(spikes.shape[0]):  # for each spike
        # re-define baseline to be 2 seconds surrounding
        surround = sur_time
        istart = int(
            max(0, np.around(spikes[i, 0] - surround * fs))
        )  # find -2s index, ensuring not to exceed idx bounds
        iend = int(
            min(len(hpsignal), np.around(spikes[i, 0] + surround * fs + 1))
        )  # find +2s index, ensuring not to exceed idx bounds

        alt_thresh = (
            np.median(np.abs(hpsignal[istart:iend])) * tmul
        )  # identify threshold within this window

        if (spikes[i, 2] > alt_thresh) & (
            spikes[i, 2] > absthresh
        ):  # both parts together are bigger than thresh: so have some flexibility in relative sizes
            if (
                spikes[i, 1] > spkdur[0]
            ):  # spike wave cannot be too sharp: then it is either too small or noise
                if spikes[i, 2] < too_high_abs:
                    out.append(spikes[i, 0])  # add timestamp of spike to output list
                else:
                    toobig.append(spikes[i, 0])  # spike is above too_high_abs
            else:
                toosharp.append(spikes[i, 0])  # spike duration is too short
        else:
            toosmall.append(spikes[i, 0])  # window-relative spike height is too short

    # Spike Realignment
    if out:
        timeToPeak = np.array([-0.15, 0.15])
        fullSurround = np.array([-sur_time, sur_time]) * fs
        idxToPeak = timeToPeak * fs

        hpsignal_length = len(hpsignal)

        for i in range(len(out)):
            currIdx = out[i]
            surround_idx = np.arange(
                max(0, round(currIdx + fullSurround[0])),
                min(round(currIdx + fullSurround[1]), hpsignal_length),
            )
            idxToLook = np.arange(
                max(0, round(currIdx + idxToPeak[0])),
                min(round(currIdx + idxToPeak[1]), hpsignal_length),
            )
            snapshot = hpsignal[idxToLook] - np.median(hpsignal[surround_idx])
            I = np.argmax(np.abs(snapshot))
            out[i] = idxToLook[0] + I

    return np.array(out)


def spike_detector(data, fs, **kwargs):
    """
    Parameters
    data:           np.NDArray - iEEG recordings (m samples x n channels)
    fs:             int - sampling frequency

    kwargs
    tmul:           float - 19 - threshold multiplier
    absthresh:      float - 100 - absolute threshold for spikes
    sur_time:       float - 0.5 - time surrounding a spike to analyze in seconds
    close_to_edge:  float - 0.05 - beginning and ending buffer in seconds
    too_high_abs:   float - threshold for artifact rejection
    spkdur:         Iterable - min and max spike duration thresholds in ms (min, max)
    lpf1:           float - low pass filter cutoff frequency
    hpf:            float - high pass filter cutoff frequency

    Returns
    gdf:            np.NDArray - spike locations (m spikes x (peak index, channel))
    """

    ### Assigning KWARGS using a more efficient method ###############
    tmul = kwargs.get("tmul", 19)  # 25
    absthresh = kwargs.get("absthresh", 100)
    sur_time = kwargs.get("sur_time", 0.5)
    close_to_edge = kwargs.get("close_to_edge", 0.05)
    too_high_abs = kwargs.get("too_high_abs", 1e3)
    # spike duration must be less than this in ms. It gets converted to samples here
    spkdur = kwargs.get("spkdur", np.array([15, 260]))
    lpf1 = kwargs.get("lpf1", 30)  # low pass filter for spikey component
    hpf = kwargs.get("hpf", 7)  # high pass filter for spikey component
    ###################################

    # Assertions and assignments
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()
    if not isinstance(spkdur, np.ndarray):
        spkdur = np.array(spkdur)

    # Receiver and constant variable initialization
    # all_spikes = np.ndarray((1,2),dtype=float)
    all_spikes = []
    nchs = data.shape[1]
    spkdur = (spkdur / 1000) * fs  # From milliseconds to samples

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_to_channel = {
            executor.submit(
                process_channel,
                data[:, j],  # collect channel
                fs,
                tmul,
                absthresh,
                sur_time,
                too_high_abs,
                spkdur,
                lpf1,
                hpf,
            ): j
            for j in range(nchs)
        }

        for future in concurrent.futures.as_completed(futures_to_channel):
            j = futures_to_channel[future]
            try:
                channel_out = future.result()
                # Concatenate the list of spikes to the global spike receiver
                if channel_out is not None and channel_out.any():
                    temp = (
                        np.array(
                            [
                                np.expand_dims(channel_out, 1),
                                np.tile([j], (len(channel_out), 1)),
                            ],
                            dtype=float,
                        )
                        .squeeze()
                        .T
                    )
                    # all_spikes = np.vstack((all_spikes,temp))
                    all_spikes.append(temp)
                    # change all spikes to a list, append and then vstack all at end
            except Exception as exc:
                print(f"Channel {j} generated an exception: {exc}")

    # Final Post-Processing - sort spikes by time not np.isnan(all_spikes).all():
    if len(all_spikes) == 0:
        return np.array([])
    else:
        all_spikes = np.vstack(all_spikes)
        all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))
        idx = np.argsort(all_spikes[:, 0], axis=0)
        gdf = all_spikes[idx, :]

        # if there are no spikes, just give up
        # print('No spikes detected')

    # if all_spikes.any():
    #     # remove exact copies of spikes
    #     all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))

    #     # sort spikes
    #     idx = np.argsort(all_spikes[:,0],axis=0)
    #     gdf = all_spikes[idx,:]

    ## Remove those too close to beginning and end
    if gdf.shape[0]:
        close_idx = close_to_edge * fs
        gdf = gdf[gdf[:, 0] > close_idx, :]
        gdf = gdf[gdf[:, 0] < data.shape[0] - close_idx, :]

    # removing duplicate spikes with closeness threshold
    if gdf.any():
        # distance between spike times
        gdf_diff = np.diff(gdf, axis=0)
        # time difference is below threshold - thresh not necessary because of spike realignment
        mask1 = np.abs(gdf_diff[:, 0]) < 100e-3 * fs
        # ensuring they're on different channels
        mask2 = gdf_diff[:, 1] == 0
        too_close = np.argwhere(mask1 & mask2) + 1

        # coerce shape of mask to <=2 dimensions
        too_close = too_close.squeeze()
        close_mask = np.ones((gdf.shape[0],), dtype=bool)
        close_mask[too_close] = False
        gdf = gdf[close_mask, :]

    # Check that spike occurs in multiple channels
    if gdf.any() & (nchs > 1):
        gdf = multi_channel_requirement(gdf, nchs, fs)

    return gdf
