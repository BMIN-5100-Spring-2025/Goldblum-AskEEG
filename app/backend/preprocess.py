import json
import pandas as pd
import numpy as np
from utils import bandpass_filter, notch_filter, detect_bad_channels, detect_bad_channels_eeg, bipolar_montage_ieeg, bipolar_montage_eeg, clean_labels, check_channel_type, pre_whiten

class Preprocessor():
    """
    Do basic preprocessing for a data batch
    1. Bandpass filter 0.5-100 Hz
    2. Notch filter 60 Hz
    3. Detect bad channels, if more than 20% of channels are bad, mark as a potential artifact
    4. Downsample to 256 Hz (optional)
    5. Re-reference (CAR and bipolar), for CAR, bad channels are excluded in calculating mean, bad channel indices are provided
    6. Pre-whiten (if True)
    7. Return the processed data in data frame, with other metadata saved in a dictionary

    User can retrieve either the re-refed data, or other metadata fields through get_last_packet method.
    
    Note for interface with Azure, can insert database writing after filter, two re-referencing, and pre-whiten, and also for bad masks
    May need a "patient" column to specify which patient the data is from
    """
    def __init__(self, lowcut = 0.5, highcut = 100, artifact_perc = 0.2, batch_size = 1):
        self.chs = None
        self.n_chs = None
        self.lowcut = lowcut
        self.highcut = highcut
        # removed downsample temporarily, I think the SDK and the headbox type available naturally makes all EEG data 256 Hz
        # and iEEG data 512 Hz
        # self.down_fs = down_fs 
        self.artifact_perc = artifact_perc
        self.batch_size = batch_size
        self.bad_channels = None
        self.fitted = False
        self.last_packet = None

    def _filter_data(self, data):
        # bandpass filter
        data = bandpass_filter(data, self.fs, lo = self.lowcut, hi = self.highcut)
        data = notch_filter(data, self.fs)
        return data

    def fit(self, info):
        self.fs = info['sampling_freq']
        self.fs_raw = info['sampling_freq_raw']
        
        self.batch_sample_raw = int(self.fs_raw*self.batch_size) 
        self.batch_sample = int(self.fs*self.batch_size)
        self.sample_step = self.batch_sample_raw//self.batch_sample
        
        self.chs = clean_labels(info['channel_names'])
        self.nchs = len(self.chs)
        self.type = info['study_type']
        self.ch_type = check_channel_type(self.chs)
        self.ieeg_idx = [True if i == 'ieeg' else False for i in self.ch_type ]
        self.eeg_idx = [True if i == 'eeg' else False for i in self.ch_type ]
        self.ekg_idx = [True if i == 'ekg' else False for i in self.ch_type ]
        self.eog_idx = [True if i == 'eog' else False for i in self.ch_type ]
        if self.type.lower() == 'ieeg':# not sure what's the code for intracranial
            self.nchs_eeg = np.sum(self.ieeg_idx)
            self.bipolar_labels, self.bipolar_idx = bipolar_montage_ieeg(self.chs[self.ieeg_idx])
            self.nchs_bipolar = len(self.bipolar_labels)
            self.down_fs = 512
        elif self.type.lower() == 'eeg':
            self.nchs_eeg = np.sum(self.eeg_idx)
            self.bipolar_labels, self.bipolar_idx = bipolar_montage_eeg(self.chs[self.eeg_idx])
            self.nchs_bipolar = len(self.bipolar_labels)
            self.down_fs = 256
        self.fitted = True

    def get_batches(self, data, last_batch_ind):
        """
        Add an index with title batch to the data dataframe
        """
        # need to check this, if the raw sampling freq is 1024 and the transmission sampling freq is 512,
        # the stamp samples may not be consecutive
        # the batch_starts should be from the original sample freq
        batch_starts = np.arange(data.index[0][1],data.index[-1][1],self.batch_sample_raw) # use raw batch sample number to set stamp sample range
        stamps = data.index.get_level_values('stamp').values
        inds = np.digitize(stamps, batch_starts, right = False)
        batch_indices = np.squeeze(last_batch_ind+inds)
        unique_batch = sorted(np.unique(batch_indices))
        data_copy = data.copy()
        data_copy['batch'] = batch_indices
        data_copy = data_copy.set_index('batch',append=True)
        # use batch sample number to check if the last batch is full, if not, leave the last batch for next round
        # in this way, at the end of processing, the last batch, whether full or not, need to be processed
        batches = [data_copy.iloc[batch_indices == batch,:] for batch in unique_batch]
        batches = [batch for batch in batches if batch.shape[0] == self.batch_sample] # can change to maximum allowed size for filters
        return batches
    
    def preprocess(self, data, last_batch_ind):
        """
        Do actual preprocess. 

        Args:
            data (pd.Dataframe): A pd.dataframe with columns being channel names, and two index, the first being seconds from start, the second being stamps
            last_batch_ind (int): An index for last available batch to make indexing consecutive
            ref (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert self.fitted == True, "Processor not fitted yet"
        assert data.shape[1] == self.nchs, "Data has different number of channels than fitted processor"
        batches = self.get_batches(data, last_batch_ind)
        artifact = False
        artifact_perc = 0
        packets = []
        for batch in batches:
            batch_ind = batch.index[0][2]
            data = batch.values
            index = batch.index

            # removed downsample temporarily
            # if self.fs > self.down_fs:
            #     data = downsample(data, self.fs, self.down_fs)
            #     timestamps = downsample(timestamps, self.fs, self.down_fs)

            processed = self._filter_data(data)
            raw_filtered = pd.DataFrame(processed, columns=self.chs, index=index)

            if self.type == 'ieeg':
                eeg_data = processed[:,self.ieeg_idx]
                eeg_chs = self.chs[self.ieeg_idx]
                bad_mask, details  = detect_bad_channels(eeg_data, self.fs)
            elif self.type == 'eeg':
                eeg_data = processed[:,self.eeg_idx]
                eeg_chs = self.chs[self.eeg_idx]
                bad_mask, details  = detect_bad_channels_eeg(eeg_data, self.fs)
            artifact_perc = np.sum(bad_mask == False)/self.nchs
            if artifact_perc > self.artifact_perc:
                artifact = True

            # re-reference
            # CAR
            car_data = eeg_data - np.mean(eeg_data[:,bad_mask], axis = 1)[:,np.newaxis]
            
            # BIPOLAR
            bad_ind = np.where(~bad_mask)[0]
            if self.type == 'ieeg':
                tmp_bipolar_index = self.bipolar_idx.copy()
                for i, ch in enumerate(tmp_bipolar_index):
                    if np.isin(ch[1],bad_ind) and not np.isin(ch[2],bad_ind):
                        tmp_bipolar_index[i,1] = tmp_bipolar_index[i,2]
                bipolar_data = eeg_data[:,tmp_bipolar_index[:,0]] - eeg_data[:,tmp_bipolar_index[:,1]]
                bad_mask_bipolar = np.any(np.isin(tmp_bipolar_index[:,:1],bad_ind),axis=1)
            else:
                bipolar_data = eeg_data[:,self.bipolar_idx[:,0]] - eeg_data[:,self.bipolar_idx[:,1]]
                bad_mask_bipolar = ~np.any(np.isin(self.bipolar_idx,bad_ind),axis=1)

            car_data_df = pd.DataFrame(car_data, columns = eeg_chs, index=index)
            bipolar_data_df = pd.DataFrame(bipolar_data, columns = self.bipolar_labels, index=index)

            car_data_prewhite = pd.DataFrame(pre_whiten(car_data), columns = eeg_chs, index=index)
            bipolar_data_prewhite = pd.DataFrame(pre_whiten(bipolar_data), columns = self.bipolar_labels, index=index)

            ekg_data = pd.DataFrame(processed[:,self.ekg_idx], columns = self.chs[self.ekg_idx], index=index)
            eog_data = pd.DataFrame(processed[:,self.eog_idx], columns = self.chs[self.eog_idx], index=index)

            car_bad = pd.DataFrame(bad_mask.reshape(1,-1), columns=eeg_chs, index=pd.Index([batch_ind],name='batch'))
            bipolar_bad = pd.DataFrame(bad_mask_bipolar.reshape(1,-1), columns=self.bipolar_labels, index=pd.Index([batch_ind],name='batch'))
            artifact = pd.DataFrame(artifact, index = pd.Index([batch_ind], name='batch'), columns = ['artifact'])
            artifact_perc = pd.DataFrame(artifact_perc, index = pd.Index([batch_ind], name='batch'), columns = ['artifact_perc'])

            packet = {'raw': data, 'raw_fs': self.fs, 
                                'filtered':raw_filtered, 'fs': self.down_fs,
                                'bad_mask': bad_mask, 'bad_details': details, 
                                'artifact': artifact, 'artifact_perc': artifact_perc,
                                'CAR': car_data_df, 'BIPOLAR': bipolar_data_df, 
                                'CAR_prewhite': car_data_prewhite, 'BIPOLAR_prewhite': bipolar_data_prewhite,
                                'CAR_bad': car_bad, 'BIPOLAR_bad': bipolar_bad,
                                'EKG':ekg_data, 'EOG': eog_data}
            
            packets.append(packet)
            
        return packets

preprocess_settings = json.load(open('preprocess_config.json'))