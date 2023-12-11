import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn



def enh_loss_fn(esti_list, label, seq_len):
    """Prepare GAGNet loss for training
    
    Arguments
    ---------
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        label : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.
        seq_len : int
            Number of label frame.

    Returns
    -------
        loss : float (Tensor)
            The GAGNet loss.

    """

    BATCH_SIZE = label.shape[0]
    frame_list = []
    for i in range(BATCH_SIZE):
        frame_list.append(seq_len)
    alpha_list = [0.1 for _ in range(len(esti_list))]
    alpha_list[-1] = 1
    mask_for_loss = []
    utt_num = label.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
        mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss1, loss2 = 0., 0.
    mag_label = torch.norm(label, dim=1)
    for i in range(len(esti_list)):
        mag_esti = torch.norm(esti_list[i], dim=1)
        loss1 = loss1 + alpha_list[i] * (((esti_list[i] - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
        loss2 = loss2 + alpha_list[i] * (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)



def pip_loss_fn(noisy_embeds, clean_embeds):
    """Prepare CS (Cosine Similarity) loss for training
    
    Arguments
    ---------
        noisy_embed : float (Tensor)
            Output of Hamrah encoder according to the enhanced input.
        clean_embeds : float (Tensor)
            Output of Hamrah encoder according to the clean input.

    Returns
    -------
        loss : float (Tensor)
            CS loss.

    """
    loss = 1.0 - F.cosine_similarity(noisy_embeds, clean_embeds, dim=-1)
    return torch.mean(loss)

def train_epoch(dataset, model, optimizer, loss_fn, grad_acc_step=1, step_show=100,\
                 DEVICE = 'cuda', train_with_pipeline = False):
    """train each epoch

    Arguments
    ---------
    dataset : class
        Training dataset

    model : class
        Model for training

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function

    grad_acc_step : int
        number of iteration to update parameters.

    step_show : int
        Number of batches to reduce learning rate and show training results

    DEVICE : str
        CPU or GPU.

    train_with_pipeline : bool
        Train with encoder pipeline or not.

    
    Returns
    -------
    total_loss : float
        Train loss for the epoch
    """
    model.train()

    total_loss = 0
    loss_section = 0
    section = 1

    counter = 0
    ex_counter = 0
    torch.cuda.empty_cache()
    start = time.time()
    for noisy_input, clean_target, length_ratio in tqdm(dataset):
        
        length_ratio = length_ratio.to(DEVICE)
        noisy_input = noisy_input.to(DEVICE)
        clean_target = clean_target.to(DEVICE)
        if train_with_pipeline:
            noisy_embeds, target_embeds , _, _ = model(noisy_input, clean_target, length_ratio)
            loss = loss_fn(noisy_embeds, target_embeds)

        else:
            gag_list, target_stft = model(noisy_input, clean_target)
            loss = loss_fn(gag_list, target_stft, target_stft.shape[-1])

        loss.backward()

        total_loss += loss.detach().cpu().item()
        counter += 1

        # graph is cleared here
        if counter % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()


        if counter  % step_show == 0:
            finish = time.time()

            lr = optimizer.param_groups[0]['lr']
            l = (total_loss - loss_section) / (counter - ex_counter)
            print(f"Section {section}. lr: {lr:.5f}, Loss: {l:.5f}, Time (Min): {round((finish - start) / 60, 3)}")

            loss_section = total_loss
            ex_counter = counter
            section += 1
            start = time.time()

    optimizer.zero_grad()


    total_loss = total_loss / counter
    print(f"Total Train Loss: {total_loss:.5f}")

    return total_loss

def evaluate_epoch(dataset, model, loss_fn, DEVICE = 'cuda', train_with_pipeline = False):
    """Evaluate model with loss

    Arguments
    ---------
    dataset : class
        Training dataset

    model : class
        Model for training
        
    loss_fn : function
        Loss function

    DEVICE : str
        CPU or GPU.

    train_with_pipeline : bool
        Train with encoder pipeline or not.

    
    Returns
    -------
    total_loss : float
        Train loss for the epoch
    """
    model.eval()

    total_loss = 0
    counter = 0
    torch.cuda.empty_cache()
    with torch.no_grad():  
        for noisy_input, clean_target, length_ratio in tqdm(dataset):
            length_ratio = length_ratio.to(DEVICE)

            noisy_input = noisy_input.to(DEVICE)
            clean_target = clean_target.to(DEVICE)

            if train_with_pipeline:
                noisy_embeds, target_embeds , _, _ = model(noisy_input, clean_target, length_ratio)
                loss = loss_fn(noisy_embeds, target_embeds)

            else:
                gag_list, target_stft = model(noisy_input, clean_target)
                loss = loss_fn(gag_list, target_stft, target_stft.shape[-1])


            total_loss += loss.detach().cpu().item()
            counter += 1

    total_loss = total_loss / counter

    return total_loss


# run the training and evaluation.
def run(model,
        train_loader,
        validation_loader,
        optimizer,
        loss_fn,
        save_model_path,
        step_show,
        n_epoch,
        grad_acc_step=1,
        train_with_pipeline = False,
        DEVICE = 'cuda'
        ):
    """execuation of training, evaluating and saving best model

    Arguments
    ---------
    model : class
        Model for training
        
    train_loader : class
        Training data loader
        
    validation_loader : class
        Validation data loader

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function
        
    save_model_path : str
        Path for saving model parameters

    step_show : int
        Number of batches to reduce learning rate and show training results

    n_epoch : str
        Number of epoches

    grad_acc_step : int
        number of iteration to update parameters.

    train_with_pipeline : bool
        Train with encoder pipeline or not.

    DEVICE : str
        CPU or GPU.

    Returns
    -------
    best_loss : float
        Best alidation loss.
        
    """
    
    best_loss = 1e10

    train_lrs = []

    for epoch in range(n_epoch):
        start = time.time()
        train_lrs.append(optimizer.param_groups[0]['lr'])
        print('\n',f"--- start epoch {epoch+1} ---")
        
        train_total_loss = train_epoch(train_loader, model,
                                        optimizer, loss_fn, 
                                        grad_acc_step, 
                                       step_show, DEVICE=DEVICE,
                                         train_with_pipeline = train_with_pipeline)
        
        val_total_loss = evaluate_epoch(validation_loader,
                                         model, loss_fn,
                                           DEVICE=DEVICE,
                                        train_with_pipeline = train_with_pipeline)
        
        finish = time.time()

        print(f"Train_Loss: {train_total_loss:.5f}") 
        print(f"Val_Loss: {val_total_loss:.5f}")
        print(f"Epoch_Time (min): {round((finish - start) / 60, 3)}")

        # save best model
        if val_total_loss < best_loss:
            best_loss = val_total_loss      
            torch.save(model.enhancement.state_dict(), save_model_path)

    return best_loss