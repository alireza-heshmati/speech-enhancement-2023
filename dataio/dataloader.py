import os
import librosa

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from speechbrain.processing.signal_processing import reverberate

from dataio.utils import  collate_fn

    
class SpeechAudioDataset(Dataset):
    """This class reads data and a label.

    Arguments
    ---------
    noisy_files : str
        List of names of train, validation and test noisy audio files.
    clean_files : str
        List of names of train, validation and test clean audio files as labels.
    reverb_files : str
        List of names of reverb data for the augmentation.
    base_path : str
        Path of dataset file (./dataset).
    target_rate : int
        Sampling rate of audio.
    max_length : int
        Maximum length for audio.
    t_reverb : float
        Threshold to include reverb data (min:0, max:1)

    Returns
    -------
    x_clean : float (Tensor)
        Readed clean audio. 

    x_noisy : float (Tensor)
        Readed noisy audio.

    """
    def __init__(self,
                 noisy_files,
                 clean_files,
                 reverb_files,
                 base_path,
                 target_rate,
                 max_length,
                 t_reverb 
                ):
        
        # list of files
        self.base_path = base_path
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.reverb_files = reverb_files
        self.len_reverb = len(reverb_files)
        # fixed len
        self.target_rate = target_rate
        self.max_length = max_length
        self.t_reverb = t_reverb

    def __len__(self):
        """This measures the number of total dataset."""
        return len(self.noisy_files)
        

    def __getitem__(self, index):
        """This method read each data, label and framed label according to super class Dataset. """
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        x_clean, x_noisy = self.same_length(x_clean, x_noisy)
        
        x_clean = torch.from_numpy(self.crop_audio(x_clean))
        x_noisy = torch.from_numpy(self.crop_audio(x_noisy))

        # For ratio of reverbed data
        if torch.rand(1) < self.t_reverb:
            x_noisy = self.create_reverb(x_noisy, self.reverb_files[index % self.len_reverb])

        return x_noisy, x_clean

    
    def create_reverb(self, sig, reverb_filename):
        """Create reverbed signal.

        Arguments
        ---------
        sig : float (Tensor)
            The signal that needs to be added with reverberations.
        reverb_filename : str
            name of reverberation file.

        Returns
        -------
        reverb_sig: float (Tensor)
            The reverbed signal.

        """
        reverb_ = torch.from_numpy(self.load_sample(reverb_filename))
        reverb_sig = reverberate(sig.unsqueeze(dim = 0), reverb_, rescale_amp= 'peak')

        return reverb_sig.squeeze()
    
    def load_sample(self, filename):
        """Load audio file.

        Arguments
        ---------
        filename : str
            name of audio file.

        Returns
        -------
       waveform : float
            Loaded waveform.

        """
        filename = os.path.join(self.base_path, filename)
        waveform, _ = librosa.load(filename, sr=self.target_rate)
        return waveform

    
    def same_length(self, x_clean, x_noisy):
        """Align the clean data with noisy data.

        Arguments
        ---------
        x_clean : float (Tensor)
            Clean data.
        x_noisy : float (Tensor)
            Noisy data.

        Returns
        -------
        x_clean : float (Tensor)
            Aligned clean data.
        x_noisy : float (Tensor)
            Aligned noisy data.

        """
        
        clean_length = len(x_clean)
        noisy_length = len(x_noisy)

        if clean_length > noisy_length:
            x_clean = x_clean[:noisy_length]
        else:
            x_noisy = x_noisy[:clean_length]
        
        return x_clean, x_noisy
    
    def crop_audio(self, x):
        """Crop the audio length more than some threshold.

        Arguments
        ---------
        x : float (Tensor)
            Waveform.

        Returns
        -------
        x : float (Tensor)
            Cropped waveform.

        """
        if len(x) > self.max_length:
            x = x[:self.max_length]
        
        return x
        
        
        
        

    
    
# for reading and preparing dataset
def audio_data_loader(noisy_filenames,
                      clean_filenames,
                      reverb_files,
                      data_base_path,
                      target_rate,
                      max_length, 
                      t_reverb,
                      batch_size, 
                      num_workers, 
                      pin_memory,
                      training):
    """This class prepare dataset for train, validation, test.

    Arguments
    ---------
    noisy_filenames : str
        List of names of train, validation and test noisy audio files.
    clean_filenames : str
        List of names of train, validation and test clean audio files as labels.
    reverb_files : str
        List of names of reverb data for the augmentation.
    data_base_path : str
        Path of dataset file (./dataset).
    target_rate : int
        Sampling rate of audio.
    max_length : int
        Maximum length for audio.
    t_reverb : float
        Threshold to include reverb data (min:0, max:1)
    batch_size : int
        Size of batch for training and evaluating.
    num_workers : int
        Number of cpu used.
    pin_memory : bool
        This is good for using GPU to collect data.
    training : bool
        define training mode or evaluating.

    Returns
    -------
    loader: class
        loader of train, validation or test set

    """
    
    dataset = SpeechAudioDataset(noisy_filenames,
                                 clean_filenames,
                                 reverb_files,
                                 data_base_path,
                                 target_rate,
                                 max_length,
                                 t_reverb 
                                 )
    
    if training:
        Sampler = RandomSampler(dataset, replacement=False, num_samples=300_000)
    else:
        Sampler = None
    
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=Sampler,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return loader
