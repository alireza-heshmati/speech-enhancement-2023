import os
from tqdm import tqdm
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from einops import rearrange


def create_best_test(test_noisy_path, test_clean_path, min_SNR = 10):
    """Devide evaluating set into two groups.

    Arguments
    ---------
    test_noisy_path : str
        Pathes of the noisy audio files.
    test_clean_path : str
        Pathes of the clean audio files.
    min_SNR : int
        Threshold to divide.

    Returns
    -------
    noisy : str
        First divided Pathes of the noisy audio files (upper than min_SNR).
    clean : str
        First divided Pathes of the audio files (upper than than min_SNR).
    nois_2 : str
        Second divided Pathes of the noisy audio files (under than min_SNR).
    clean_2 : str
        Second divided Pathes of the audio files (under than min_SNR).
    """

    i = 0
    noisy = []
    clean = []
    noisy_2 = []
    clean_2 = []

    for txt_name in test_noisy_path:
        if txt_name.split("/")[1][:-2] == "Inf":
            noisy.append(txt_name)
            clean.append(test_clean_path[i])

        elif int(txt_name.split("/")[1][:-2]) >= min_SNR:
            noisy.append(txt_name)
            clean.append(test_clean_path[i])
        else :
            noisy_2.append(txt_name)
            clean_2.append(test_clean_path[i])
        
        i += 1

    return noisy, clean, noisy_2, clean_2


def create_specific_filenames(noisy_path, clean_path, SNR=10):
    """Devide evaluating set into a specific group.

    Arguments
    ---------
    noisy_path : str
        Pathes of the noisy audio files.
    clean_path : str
        Pathes of the clean audio files.
    SNR : int
        Defined SNR.

    Returns
    -------
    noisy : str
        Divided Pathes of the noisy audio files.
    clean : str
        Divided Pathes of the audio files.
    """

    noisy = []
    clean = []

    for noisy_filename, clean_filename in zip(noisy_path, clean_path):
        if noisy_filename.split("/")[-2] == f"{SNR}dB":
            noisy.append(noisy_filename)
            clean.append(clean_filename)

    return noisy, clean



def create_pair_filenames(filenames):
    """Create paired noisy and clean data files name.

    Arguments
    ---------
    filenames : str
        Dataset files name.

    Returns
    -------
    noisy_filenames : str
        Paired files name of the noisy audio files with clean ones.
    clean_filenames : str
        Paired files name of the clean audio files with noisy ones.
    """

    clean_filenames = []
    noisy_filenames = []

    for chunk_filenames in tqdm(filenames):
        path = chunk_filenames[0].split(os.path.sep)
        filename = os.path.join(path[0],'InfdB',path[2].split('SPLIT')[0]+'SPLIT-SPLITInfdB.mp3')
        clean_filenames += [filename] * len(chunk_filenames)
        noisy_filenames += chunk_filenames.tolist()

    return noisy_filenames, clean_filenames      
    
def read_reverb_filenames(base_path = "./dataset"):
    """Read reveb data files name.

    Arguments
    ---------
    base_path : str
        Base path of dataset file.

    Returns
    -------
    noise_path : str
        Pahtes of the reverb audio files.
    """

    noise_1_filenames = pd.read_csv(f"{base_path}/simulated_rirs/rir_list_largeroom.csv").to_numpy().squeeze()
    noise_2_filenames = pd.read_csv(f"{base_path}/simulated_rirs/rir_list_mediumroom.csv").to_numpy().squeeze()
    noise_3_filenames = pd.read_csv(f"{base_path}/simulated_rirs/rir_list_smallroom.csv").to_numpy().squeeze()
    
    noise_filenames = list(noise_1_filenames) + list(noise_2_filenames) + list(noise_3_filenames)
    noise_path = [list_.split("RIRS_NOISES/")[1] for list_ in tqdm(noise_filenames) ]
    random.seed(12)
    random.shuffle(noise_path)

    return noise_path


def read_data_filenames(base_path = "./dataset"):
    """Read train, vslidation and test data files name.

    Arguments
    ---------
    base_path : str
        Base path of dataset file.

    Returns
    -------
    train_noisy_filenames : str
        Files name of the noisy train files.
    train_clean_filenames : str
        Files name of the clean train files.
    valid_noisy_filenames : str
        Files name of the noisy validation files.
    valid_clean_filenames : str
        Files name of the clean validation files.
    test_noisy_filenames : str
        Files name of the noisy test files.
    test_clean_filenames : str
        Files name of the clean test files.
    """

    train_filenames = pd.read_csv(f"{base_path}/train_filenames_v2.csv")
    valid_filenames = pd.read_csv(f"{base_path}/valid_filenames.csv")
    test_filenames = pd.read_csv(f"{base_path}/test_filenames.csv")

    train_filenames = train_filenames.to_numpy().squeeze()
    train_filenames = rearrange(train_filenames, "(n c) -> n c", c=1)

    valid_filenames = valid_filenames.to_numpy().squeeze()
    valid_filenames = rearrange(valid_filenames, "(n c) -> n c", c=22)

    test_filenames = test_filenames.to_numpy().squeeze()
    test_filenames = rearrange(test_filenames, "(n c) -> n c", c=22)

    train_noisy_filenames, train_clean_filenames = create_pair_filenames(train_filenames)
    valid_noisy_filenames, valid_clean_filenames = create_pair_filenames(valid_filenames)
    test_noisy_filenames, test_clean_filenames = create_pair_filenames(test_filenames)

    valid_noisy_filenames, valid_clean_filenames , _, _ = create_best_test(valid_noisy_filenames, valid_clean_filenames, 10)
    valid_noisy_filenames, valid_clean_filenames = valid_noisy_filenames[:100_000], valid_clean_filenames[:100_000]

    return train_noisy_filenames, train_clean_filenames,valid_noisy_filenames,\
          valid_clean_filenames, test_noisy_filenames, test_clean_filenames
    
def collate_fn(batch):
    """help torch.loader give data and labels. indeed preparing
     readed dataset for network with padding.

    Arguments
    ---------
    batch : tuple, Tensor
        It includes the output of Dataset class

    Returns
    -------
    inputs : float (Tensor)
        Padded noisy audios
    targets : float (Tensor)
        Padded clean audios as targets
    length_ratio : float (Tensor)
        Real length of each audio

    """
    inputs, targets, length_ratio = [], [], []
    for noisy_input, clean_target in batch:
        inputs.append(noisy_input)
        targets.append(clean_target)
        length_ratio.append(len(noisy_input))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    length_ratio = torch.tensor(length_ratio, dtype=torch.long) / inputs.shape[1]

    return inputs, targets, length_ratio
    
    
    