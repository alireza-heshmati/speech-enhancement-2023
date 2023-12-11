import argparse
import torch
import timeit
import glob
import os
import librosa
import soundfile
from models.utils import create_enhancement_model_and_configs
from models.utils import ISTFT, STFT, EnhancementW2W

# use cuda if cuda available 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model_name", "--name_of_enhancement_model",
                required=True,
                type=str,
                help="name of GAGNet models, such as gagnet, gagnet-v2 and gagnet-v4" )

ap.add_argument("-demo_path", "--path_of_demo_files", 
                required=True,
                type=str,
                help="path of demo file of samples, such as ./demo")

ap.add_argument("-samples_file", "--name_of_samples_file", 
                required=True,
                type=str,
                help="name of samples file to load samples in demo file, such as samples")

ap.add_argument("-enhanced_file", "--name_of_enhanced_file", 
                required=True,
                type=str,
                help="name of enhanced file to save enhanced samples in demo file, such as enhanced_samples")

args = vars(ap.parse_args())


model_name = args["name_of_enhancement_model"]
demo_path = args["path_of_demo_files"]
samplesfile = args["name_of_samples_file"]
enhancedfile = args["name_of_enhanced_file"]

path_list = []
for path in glob.glob(os.path.join(demo_path,samplesfile+"/*")):
    path_list.append(path)


stft_layer = STFT().to(DEVICE)
istft_layer = ISTFT().to(DEVICE)


enhacement, model_configs = create_enhancement_model_and_configs(model_name = model_name,
                                                                  DEVICE = DEVICE)

enhacement.load_state_dict(torch.load(model_configs["save_path_rev"]))
enh_model = EnhancementW2W(enhacement, stft_layer, istft_layer)


sr=16000
start_total = timeit.default_timer()

for path_audio in path_list:
    
    sig, sr = librosa.load(path_audio,sr=sr, dtype='float32')

    enh_model.eval()

    torch.cuda.empty_cache()
    with torch.no_grad():
        enhanced_sig = enh_model(torch.from_numpy(sig).unsqueeze(dim = 0).to(DEVICE)).cpu().numpy()

    soundfile.write(os.path.join(demo_path, enhancedfile ,path_audio.split("/")[-1]) , 
                    enhanced_sig[0], sr)


print(f"Total Time (min): {(timeit.default_timer() - start_total)//60}")





