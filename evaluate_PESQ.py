import argparse
import torch
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
from pesq import pesq_batch
from recipes.utils import load_model_config
from models.utils import create_enhancement_model_and_configs
from dataio.utils import read_reverb_filenames,read_data_filenames, create_specific_filenames
from dataio.dataloader import audio_data_loader
from models.utils import STFT, ISTFT, EnhancementW2W


# use cuda if cuda available 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_name", "--name_of_enhancement_model",
                required=True,
                type=str,
                help="name of GAGNet models, such as gagnet, gagnet-v2 and gagnet-v4" )

ap.add_argument("-data_path", "--path_of_data_files", 
                required=True,
                type=str,
                help="path of main file of data, such as ./dataset")

ap.add_argument("-data_cfg", "--data_config_path", 
                required=True,
                type=str,
                help="config of preparing data, such as ./recipes/dataio.json")

args = vars(ap.parse_args())


model_name = args["name_of_enhancement_model"]
BASEPATH = args["path_of_data_files"]
data_cfg = args["data_config_path"]

reverb_path = read_reverb_filenames(base_path = BASEPATH)

_, _,_, _, test_noisy_filenames, test_clean_filenames = read_data_filenames(base_path = BASEPATH)


stft_layer = STFT().to(DEVICE)
istft_layer = ISTFT().to(DEVICE)

enhacement, model_configs = create_enhancement_model_and_configs(model_name = model_name,
                                                                  DEVICE = DEVICE)

save_model_path = model_configs["save_path_rev"]

enhacement.load_state_dict(torch.load(save_model_path))

enh_model = EnhancementW2W(enhacement, stft_layer, istft_layer)

data_configs = load_model_config(data_cfg)

PESQ = {
    "PESQ_WB_CR" : {},
    "PESQ_NB_CR" : {},
}

enh_model.eval()

for SNR in range(-2, 31, 2):
    print(f"---------------- {SNR} dB -----------------")

    test_noisy_filenames_spec, test_clean_filenames_spec = create_specific_filenames(
        test_noisy_filenames, test_clean_filenames, SNR=SNR)

    test_dataset = audio_data_loader(test_noisy_filenames_spec,
                                    test_clean_filenames_spec,
                                    reverb_path,
                                    BASEPATH,
                                    data_configs["test"]["SAMPLE_RATE"],
                                    data_configs["test"]["MAX_LENGTH"],
                                    data_configs["test"]["T_REVERB"],
                                    data_configs["test"]["BATCH_SIZE"], 
                                    data_configs["test"]["NUM_WORKER"],
                                    data_configs["test"]["PIN_MEMORY"],
                                    data_configs["test"]["TRAINING"])

    PESQ_WB_CN, PESQ_WB_CR, PESQ_NB_CN, PESQ_NB_CR = [], [], [], []
    torch.cuda.empty_cache()
    for x_noisy, x_clean, _ in tqdm(test_dataset):

        with torch.no_grad():
            x_clean = x_clean.to(DEVICE)
            x_noisy = x_noisy.to(DEVICE)

            x_clean = enh_model.normalizer(x_clean.unsqueeze(dim=1), 16000)

            x_noisy_rec = enh_model(x_noisy)

        
        x_clean = x_clean.cpu().numpy()
        x_noisy_rec = x_noisy_rec.cpu().numpy()

        if x_clean.shape[1] > x_noisy_rec.shape[1]:
            x_clean = x_clean[:,:x_noisy_rec.shape[1]]
        else:
            x_noisy_rec = x_noisy_rec[:,:x_clean.shape[1]]

        PESQ_WB_CR += pesq_batch(16000, x_clean, x_noisy_rec, mode='wb')
        PESQ_NB_CR += pesq_batch(16000, x_clean, x_noisy_rec, mode='nb')

    PESQ["PESQ_WB_CR"][SNR] = np.mean(PESQ_WB_CR)
    PESQ["PESQ_NB_CR"][SNR] = np.mean(PESQ_NB_CR)


X = list(PESQ["PESQ_WB_CR"].keys())
Y = list(PESQ["PESQ_WB_CR"].values())
plt.plot(X, Y, "-o")

X = list(PESQ["PESQ_NB_CR"].keys())
Y = list(PESQ["PESQ_NB_CR"].values())
plt.plot(X, Y, "--P")

plt.title("PESQ")
plt.legend(["PESQ_WB", "PESQ_NB"])
plt.xlabel("SNR(dB)")
plt.ylabel("PESQ")
plt.grid()
plt.show()





















