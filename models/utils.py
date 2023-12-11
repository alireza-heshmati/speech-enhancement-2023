import torch
import torch.nn as nn
from recipes.utils import load_model_config
from models.gagnet import GaGNet
from speechbrain.dataio.preprocess import AudioNormalizer

def create_enhancement_model_and_configs(model_name = "gagnet", DEVICE = "cuda"):
    """Create the enhancement model and its configs

    Arguments
    ---------
    model_name : str
        Name of enhancement model ("gagnet", "gagnet-v2" and "gagnet-v4").
    DEVICE : str
        GPU ("cuda") or CPU ("cpu").

    Returns
    -------
    enhacement : class
        The enhancement model.
    model_configs : dict
        The enhancement model configs.

    """

    if model_name == "gagnet":
        config_path = "./recipes/gagnet.json"
    elif model_name == "gagnet-v2":
        config_path = "./recipes/gagnet-v2.json"
    elif model_name == "gagnet-v4":
        config_path = "./recipes/gagnet-v4.json"
    else:
        raise ValueError("the name of the model is not supported!!")

    model_configs = load_model_config(config_path)

    enhacement = GaGNet(
        cin=model_configs["cin"],
        k1=tuple(model_configs["k1"]),
        k2=tuple(model_configs["k2"]),
        c=model_configs["c"],
        kd1=model_configs["kd1"],
        cd1=model_configs["cd1"],
        d_feat=model_configs["d_feat"],
        p=model_configs["p"],
        q=model_configs["q"],
        dilas=model_configs["dilas"],
        fft_num=model_configs["fft_num"],
        is_u2=model_configs["is_u2"],
        is_causal=model_configs["is_causal"],
        is_squeezed=model_configs["is_squeezed"],
        acti_type=model_configs["acti_type"],
        intra_connect=model_configs["intra_connect"],
        norm_type=model_configs["norm_type"],
        ).to(DEVICE)

    return enhacement, model_configs

class ISTFT(nn.Module):
    """Inverse short-time Fourier transform (ISTFT).

    Arguments
    ---------
        input : float (Tensor)
            The input tensor. Expected to be in the format of :func:`~torch.stft`, output.
            That is a complex tensor of shape `(B?, N, T)` where 
            - `B?` is an optional batch dimension
            - `N` is the number of frequency samples, `(n_fft // 2) + 1`
                for onesided input, or otherwise `n_fft`.
            - `T` is the number of frames, `1 + length // hop_length` for centered stft,
                or `1 + (length - n_fft) // hop_length` otherwise.
        n_fft : int
            Size of Fourier transform.
        hop_length : int
            The distance between neighboring sliding window.
        window : str
           The optional window function.
        normalized : bool
            controls whether to return the normalized ISTFT results

    Returns
    -------
        x_istft : float (Tensor)
            ISTFT of the input.

    """
    def __init__(self,
                 n_fft=400,
                 hop_length=160,
                 window="hamming_window",
                 normalized=False):
        super(ISTFT, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.window = getattr(torch, window)(n_fft)
        
    def forward(self, x):
        """This method should implement forwarding operation in the ISTFT.

        Arguments
        ---------
        x : float (Tensor)
            The input of ISTFT.

        Returns
        -------
        x_istft : float (Tensor)
            The output of ISTFT.
        """
        x = torch.view_as_complex(x)

        x_istft = torch.istft(x,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=self.window.to(x.device),
                              normalized=self.normalized)
        
        return x_istft    



class STFT(nn.Module):
    """Short-time Fourier transform (STFT).

    Arguments
    ---------
        input : float (Tensor)
            The input tensor of shape `(B, L)` where `B` is an optional.
        n_fft : int
            Size of Fourier transform.
        hop_length : int
            The distance between neighboring sliding window.
        window : str
           The optional window function.
        normalized : bool
            Controls whether to return the normalized STFT results.
        pad_mode : str
            controls the padding method used.
        return_complex : bool
            Whether to return a complex tensor, or a real tensor with 
            an extra last dimension for the real and imaginary components.

    Returns
    -------
        x_istft : float (Tensor)
            STFT of the input.

    """
    def __init__(self,
                 n_fft=400,
                 hop_length=160,
                 window="hamming_window",
                 normalized=False,
                 pad_mode="constant",
                 return_complex=False):
        super(STFT, self).__init__()
        
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.return_complex = return_complex
        self.pad_mode = pad_mode
        self.window = getattr(torch, window)(n_fft)
        
        
    def forward(self, x):
        """This method should implement forwarding operation in the STFT.

        Arguments
        ---------
        x : float (Tensor)
            The input of STFT.

        Returns
        -------
        x_stft : float (Tensor)
            The output of STFT.
        """
        x_stft = torch.stft(input=x,
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length,
                            window=self.window.to(x.device),
                            normalized=self.normalized,
                            pad_mode=self.pad_mode,
                            return_complex=self.return_complex)
        
        return x_stft 
        
        
        
def load_asr_encoder(path, device):
    """Load ASR encoder and adapt it with device.

    Arguments
    ---------
        path : str
            The path of Hamrah ASR encoder checkpoint.
        device : str
            "cuda" or "cpu".

    Returns
    -------
        encoder : float (Tensor)
            Pre-trained Hamrah ASR encoder, adapting with device.

    """
    network = torch.load(path)
    
    encoder = network["encoder"]
    encoder.compute_features.compute_STFT = torch.nn.Identity()

    if device != "cpu":
        encoder = encoder.to(device)
        encoder.normalize.glob_mean = encoder.normalize.glob_mean.to(device)
        encoder.normalize.glob_std = encoder.normalize.glob_std.to(device)
        
    return encoder
    

def preprocessing_for_GAGNet_train(noisy_stft, target_stft):
    """Pre-processing of GAGNet models for input and target of them.

    Arguments
    ---------
        noisy_stft : float (Tensor)
            STFT of the input of GAGNet models.
        target_stft : float (Tensor)
            STFT of the target of GAGNet models.

    Returns
    -------
        c_noisy_stft : float (Tensor)
            Changed STFT of the input of GAGNet models.
        c_target_stft : float (Tensor)
            Changed STFT of the target of GAGNet models.

    """
    noisy_stft = noisy_stft.permute(0,3,2,1)
    target_stft = target_stft.permute(0,3,1,2)
    noisy_mag = torch.norm(noisy_stft, dim=1) ** 0.5
    noisy_phase = torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
    target_mag = torch.norm(target_stft, dim=1) ** 0.5
    target_phase = torch.atan2(target_stft[:, -1, ...], target_stft[:, 0, ...])
    
    c_noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase),
                               noisy_mag * torch.sin(noisy_phase)), dim=1)
    c_target_stft = torch.stack((target_mag * torch.cos(target_phase),
                                target_mag * torch.sin(target_phase)), dim=1)

    return c_noisy_stft, c_target_stft

def preprocessing_for_GAGNet(noisy_stft):
    """Pre-processing of GAGNet models for input of them.

    Arguments
    ---------
        noisy_stft : float (Tensor)
            STFT of the input of GAGNet models.

    Returns
    -------
        c_noisy_stft : float (Tensor)
            Changed STFT of the input of GAGNet models.

    """
    noisy_stft = noisy_stft.permute(0,3,2,1)
    noisy_mag = torch.norm(noisy_stft, dim=1) ** 0.5
    noisy_phase = torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
    
    c_noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase),
                               noisy_mag * torch.sin(noisy_phase)), dim=1)

    return c_noisy_stft

def postprocessing_for_GAGNet(enhancement_output):
    """Post-processing of GAGNet models for output of them.

    Arguments
    ---------
        enhancement_output : float (Tensor)
            Output of GAGNet models.

    Returns
    -------
        c_enhancement_output : float (Tensor)
            Changed output of GAGNet models.

    """

    est_mag = torch.norm(enhancement_output, dim=-1)**2.0
    est_phase = torch.atan2(enhancement_output[..., -1], enhancement_output[...,0])
    c_enhancement_output = torch.stack((est_mag*torch.cos(est_phase),
                                      est_mag*torch.sin(est_phase)), dim=-1)

    return c_enhancement_output

class EnhancemetPipline_train(nn.Module):
    """Enhancement training with encoder Hamrah.

    Arguments
    ---------
        enhancement : class
            GAGNet models.
        stft_layer : class
            STFT module.
        asr_encoder : class
            Pre-trained Hamrah ASR encoder.

    Returns
    -------
        noisy_embed : float (Tensor)
            Output of Hamrah encoder according to the enhanced input.
        target_embed : float (Tensor)
            Output of Hamrah encoder according to the clean input.
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.

    """
    def __init__(self,
                 enhancement,
                 stft_layer,
                 asr_encoder,
                 ):
        super(EnhancemetPipline_train, self).__init__()
        self.enhancement = enhancement
        self.asr_encoder = asr_encoder
        self.stft_layer = stft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input, clean_target, length_ratio):
        """This method should implement forwarding operation in the EnhancemetPipline_train.

        Arguments
        ---------
        noisy_input : float (Tensor)
            The noisy input of EnhancemetPipline_train.
        clean_target : float (Tensor)
            The clean input of EnhancemetPipline_train.
        length_ratio : float (Tensor)
            The original length ratio of each audio in input related to maximum length.

        Returns
        -------
        noisy_embed : float (Tensor)
            Output of Hamrah encoder according to the enhanced input.
        target_embed : float (Tensor)
            Output of Hamrah encoder according to the clean input.
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.
        """
        
        clean_target= self.normalizer(clean_target.unsqueeze(dim=1),16000)
        noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)
        
        noisy_stft = self.stft_layer(noisy_input)
        clean_stft = self.stft_layer(clean_target)

        noisy_stft, target_stft = preprocessing_for_GAGNet_train(noisy_stft, clean_stft)

        clean_stft = clean_stft.permute([0,2,1,3])

        esti_list = self.enhancement(noisy_stft)
        
        enhancement_output = esti_list[-1].permute([0,2,3,1])

        enhancement_output = postprocessing_for_GAGNet(enhancement_output).permute([0,2,1,3])

        noisy_embed = self.asr_encoder(enhancement_output, length_ratio)
        target_embed = self.asr_encoder(clean_stft, length_ratio)

        return noisy_embed, target_embed, esti_list, target_stft


class EnhancemetSolo_train(nn.Module):
    """Enhancement training without encoder Hamrah.

    Arguments
    ---------
        enhancement : class
            GAGNet models.
        stft_layer : class
            STFT module.

    Returns
    -------
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.

    """
    def __init__(self,
                 enhancement,
                 stft_layer,
                 ):
        super(EnhancemetSolo_train, self).__init__()
        self.enhancement = enhancement
        self.stft_layer = stft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input, clean_target):
        """This method should implement forwarding operation in the EnhancemetSolo_train.

        Arguments
        ---------
        noisy_input : float (Tensor)
            The noisy input of EnhancemetSolo_train.
        clean_target : float (Tensor)
            The clean input of EnhancemetSolo_train.

        Returns
        -------
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.

        """

        clean_target= self.normalizer(clean_target.unsqueeze(dim=1),16000)
        noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)

        noisy_stft = self.stft_layer(noisy_input)
        clean_stft = self.stft_layer(clean_target)

        noisy_stft, target_stft = preprocessing_for_GAGNet_train(noisy_stft, clean_stft)
        esti_list = self.enhancement(noisy_stft)

        return esti_list, target_stft
    

class EnhancementW2W(nn.Module):
    """End-to-End Enhancement module.

    Arguments
    ---------
        enhancement : class
            GAGNet models.
        stft_layer : class
            STFT module.
        istft_layer : class
            ISTFT module.

    Returns
    -------
        noisy_rec : float (Tensor)
            Enhanced of the input audio

    """
    def __init__(self,
                 enhancement,
                 stft_layer,
                 istft_layer
                 ):
        super(EnhancementW2W, self).__init__()
        self.enhancement = enhancement
        self.stft_layer = stft_layer
        self.istft_layer = istft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input):
        """This method should implement forwarding operation in the EnhancementW2W.

        Arguments
        ---------
        noisy_input : float (Tensor)
            The noisy input of EnhancementW2W.

        Returns
        -------
        noisy_rec : float (Tensor)
            Enhanced of the input audio

        """
        
        noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)
        noisy_stft = self.stft_layer(noisy_input)
        noisy_stft = preprocessing_for_GAGNet(noisy_stft)

        rec_stft = self.enhancement(noisy_stft)[-1].permute([0,2,3,1])

        est_stft = postprocessing_for_GAGNet(rec_stft)
        noisy_rec = self.istft_layer(est_stft)
        
        return noisy_rec

