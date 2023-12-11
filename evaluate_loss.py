import torch
import argparse
import time
from recipes.utils import load_model_config
from models.utils import create_enhancement_model_and_configs
from tools.train_utils import pip_loss_fn, enh_loss_fn, evaluate_epoch
from dataio.utils import read_reverb_filenames,read_data_filenames, create_best_test
from dataio.dataloader import audio_data_loader
from models.utils import STFT, EnhancemetPipline_train, EnhancemetSolo_train, load_asr_encoder

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

ap.add_argument("-ecoder_ckp", "--Path_of_Hamrah_encoder", 
                required=True,
                type=str,
                help="path for loading checkpoints of Hamrah encoder,  such as ./checkpoints/ASR-Model/asr-encoder.ckpt" )

ap.add_argument("-pip_loss", "--IS_PIPELINE_loss", 
                required=True,
                type=bool,
                help="Whether evaluate pipeline loss or enhancement loss, True or False" )

args = vars(ap.parse_args())

model_name = args["name_of_enhancement_model"]
BASEPATH = args["path_of_data_files"]
data_cfg = args["data_config_path"]
ecoder_ckp = args["Path_of_Hamrah_encoder"]
pip_loss = args["IS_PIPELINE_loss"]

reverb_path = read_reverb_filenames(base_path = BASEPATH)

_, _,_, _, test_noisy_filenames, test_clean_filenames = read_data_filenames(base_path = BASEPATH)

test_noisy_filenames, test_clean_filenames, test_noisy_filenames_2, \
      test_clean_filenames_2 = create_best_test(test_noisy_filenames, test_clean_filenames, 10)

data_configs = load_model_config(data_cfg)

test_dataset = audio_data_loader(test_noisy_filenames,
                                 test_clean_filenames,
                                 reverb_path,
                                 BASEPATH,
                                  data_configs["test"]["SAMPLE_RATE"],
                                  data_configs["test"]["MAX_LENGTH"],
                                  data_configs["test"]["T_REVERB"],
                                  data_configs["test"]["BATCH_SIZE"], 
                                  data_configs["test"]["NUM_WORKER"],
                                  data_configs["test"]["PIN_MEMORY"],
                                  data_configs["test"]["TRAINING"])

test_dataset_2 = audio_data_loader(test_noisy_filenames_2,
                                 test_clean_filenames_2,
                                 reverb_path,
                                 BASEPATH,
                                  data_configs["test"]["SAMPLE_RATE"],
                                  data_configs["test"]["MAX_LENGTH"],
                                  data_configs["test"]["T_REVERB"],
                                  data_configs["test"]["BATCH_SIZE"], 
                                  data_configs["test"]["NUM_WORKER"],
                                  data_configs["test"]["PIN_MEMORY"],
                                  data_configs["test"]["TRAINING"])

n_u = len(test_dataset)
n_d = len(test_dataset_2) 
n_t = n_u + n_d
print(n_t)

stft_layer = STFT().to(DEVICE)

enhacement, model_configs = create_enhancement_model_and_configs(model_name = model_name,
                                                                  DEVICE = DEVICE)


if pip_loss:
    asr_encoder = load_asr_encoder(ecoder_ckp
                                , device=DEVICE)
    for param in asr_encoder.parameters():
        param.requires_grad = False

    loss_fn = pip_loss_fn

    save_model_path = model_configs["save_path_pip"]

    enhacement.load_state_dict(torch.load(save_model_path))

    enh_pipline = EnhancemetPipline_train(enhacement, stft_layer, asr_encoder)

else:
    loss_fn = enh_loss_fn
    save_model_path = model_configs["save_path_rev"]

    enhacement.load_state_dict(torch.load(save_model_path))

    enh_pipline = EnhancemetSolo_train(enhacement, stft_layer)

torch.cuda.empty_cache()

start = time.time()

print("Start computing loss for", model_configs["name"],"model ...")
test_total_loss = evaluate_epoch(test_dataset, enh_pipline, loss_fn, DEVICE=DEVICE,
                                 train_with_pipeline = pip_loss)
print(f"upper than 10dB Test Loss: {test_total_loss:.5f}, Time(min): {(time.time()-start)//60}")

start = time.time()
test_total_loss_2 = evaluate_epoch(test_dataset_2, enh_pipline, loss_fn, DEVICE=DEVICE,
                                   train_with_pipeline = pip_loss)
print(f"lesser 10dB Test Loss: {test_total_loss_2:.5f}, Time(min): {(time.time()-start)//60}")

print(f"\n Total Test Loss: {(test_total_loss * n_u + test_total_loss_2 * n_d)/n_t:.5f}.")
























