import argparse

import torch
from models.utils import create_enhancement_model_and_configs
from recipes.utils import load_model_config
from tools.train_utils import run, pip_loss_fn, enh_loss_fn
from dataio.utils import read_reverb_filenames,read_data_filenames
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

ap.add_argument("-train_cfg", "--train_config_path", 
                required=True,
                type=str,
                help="config of parameters for training, such as ./recipes/training.json")

ap.add_argument("-ecoder_ckp", "--Path_of_Hamrah_encoder", 
                required=True,
                type=str,
                help="path for loading checkpoints of Hamrah encoder,  such as ./checkpoints/ASR-Model/asr-encoder.ckpt" )

args = vars(ap.parse_args())


model_name = args["name_of_enhancement_model"]
BASEPATH = args["path_of_data_files"]
data_cfg = args["data_config_path"]
train_cfg = args["train_config_path"]
ecoder_ckp = args["Path_of_Hamrah_encoder"]
    

training_configs = load_model_config(train_cfg)


reverb_path = read_reverb_filenames(base_path = BASEPATH)

train_noisy_filenames, train_clean_filenames,valid_noisy_filenames, valid_clean_filenames,\
      test_noisy_filenames, test_clean_filenames = read_data_filenames(base_path = BASEPATH)


data_configs = load_model_config(data_cfg)

train_dataset = audio_data_loader(train_noisy_filenames,
                                  train_clean_filenames,
                                  reverb_path,
                                  BASEPATH,
                                  data_configs["train"]["SAMPLE_RATE"],
                                  data_configs["train"]["MAX_LENGTH"],
                                  data_configs["train"]["T_REVERB"],
                                  data_configs["train"]["BATCH_SIZE"], 
                                  data_configs["train"]["NUM_WORKER"],
                                  data_configs["train"]["PIN_MEMORY"],
                                  data_configs["train"]["TRAINING"])


valid_dataset = audio_data_loader(valid_noisy_filenames,
                                  valid_clean_filenames,
                                  reverb_path,
                                  BASEPATH,
                                  data_configs["test"]["SAMPLE_RATE"],
                                  data_configs["test"]["MAX_LENGTH"],
                                  data_configs["test"]["T_REVERB"],
                                  data_configs["test"]["BATCH_SIZE"], 
                                  data_configs["test"]["NUM_WORKER"],
                                  data_configs["test"]["PIN_MEMORY"],
                                  data_configs["test"]["TRAINING"])



stft_layer = STFT().to(DEVICE)

enhacement, model_configs = create_enhancement_model_and_configs(model_name = model_name,
                                                                  DEVICE = DEVICE)

if training_configs["IS_PIPELINE"]:
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


optimizer = torch.optim.Adam(enh_pipline.parameters(),
                              lr=training_configs["LEARNING_RATE"])

best_valid_loss = run(
    enh_pipline,
    train_dataset,
    valid_dataset,
    optimizer,
    loss_fn,
    save_model_path= save_model_path,
    step_show= training_configs["STEP_SHOW"],
    n_epoch= training_configs["NUM_EPOCH"],
    grad_acc_step= training_configs["GRAD_STEP"],
    train_with_pipeline = training_configs["IS_PIPELINE"],
    DEVICE=DEVICE
)

print(f"Best loss {best_valid_loss}")