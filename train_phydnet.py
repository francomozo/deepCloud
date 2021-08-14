from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset, MontevideoFoldersDataset_w_time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import datetime
import random
import time
from src.dl_models.phydnet import ConvLSTM,PhyCell, EncoderRNN
#from data.moving_mnist import MovingMNIST
from src.dl_models.phydnet import K2M
from skimage.metrics import structural_similarity as ssim
import argparse
import matplotlib.pyplot as plt


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio, writer=None):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    
    loss = 0
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, output_image, _, _ = encoder(input=input_tensor[:, ei, :,:, :],
                                                                     first_timestep=(ei==0),
                                                                     decoding=False)
        loss += criterion(output_image, input_tensor[:, ei+1,:, :, :])

    decoder_input = input_tensor[:,-1,:,:] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
        target = target_tensor[:,di,:,:]
        loss += criterion(output_image, target)
        if use_teacher_forcing:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image

    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length

def trainIters(train_loader, val_loader, encoder, nepochs, print_every=10, eval_every=10, checkpoint=False, model_name='', writer=None):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    criterion = nn.MSELoss()
    
    for epoch in range(0, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003) # ??
        
        for i, (in_frames, out_frames) in enumerate(train_loader):
            input_tensor = in_frames.unsqueeze(2).to(device=device)
            target_tensor = out_frames.unsqueeze(2).to(device=device)
            
            loss = train_on_batch(input_tensor[:,:,:,:,:], target_tensor[:,:,:,:,:], encoder, encoder_optimizer, criterion, teacher_forcing_ratio)                                   
            loss_epoch += loss
            
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS", loss_epoch, epoch)             
                      
        train_losses.append(loss_epoch)        
        if (epoch+1) % print_every == 0:
            print('epoch ', epoch,  ' loss ', loss_epoch, 'train time epoch ', time.time()-t0)
            
        if (epoch+1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, val_loader, epoch, writer) 
            scheduler_enc.step(mse)
            if writer: 
                writer.add_scalar("VAL MSE", mse, epoch)
                writer.add_scalar("VAL MAE", mae, epoch)
                writer.add_scalar("VAL SSIM", ssim, epoch)
                writer.add_scalar("Learning rate", encoder_optimizer.state_dict()["param_groups"][0]["lr"], epoch)
                
            if checkpoint:    
                if mse < best_mse:
                    best_mse = mse
                    model_dict = {
                        'epoch': epoch + 1,
                        'model_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': encoder_optimizer.state_dict(),
                        'train_loss_epoch_mean': loss_epoch,
                        'val_loss_mae': mae,
                        'val_loss_mse': mse,
                        'val_loss_ssim': ssim,
                    }   
                    PATH = 'checkpoints/'
                    ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                    NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'
                    torch.save(model_dict, PATH + NAME)  
                          
    return train_losses

def evaluate(encoder, val_loader, epoch, writer=None):
    total_mse, total_mae, total_ssim,total_bce = 0,0,0,0
    t0 = time.time()
    with torch.no_grad():
        for i, (in_frames, out_frames) in enumerate(val_loader):
            input_tensor = in_frames.unsqueeze(2).to(device=device)
            target_tensor = out_frames.unsqueeze(2).to(device=device)
            
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input,
                                                                           first_timestep=False,
                                                                           decoding=False)
                decoder_input = output_image
                if writer and (i == 0):
                    writer.add_images('predictions_batch', output_image, epoch)
                    
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions) 
            predictions = predictions.swapaxes(0,1)

            mse_batch = np.mean((predictions-target)**2, axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(predictions-target), axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            
            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1]) 
            
            cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (batch_size*target_length)
            total_bce +=  cross_entropy
     
    print('eval mse ', total_mse/len(val_loader),  ' eval mae ', total_mae/len(val_loader),' eval ssim ',total_ssim/len(val_loader), 'eval time= ', time.time()-t0)        
    return total_mse/val_len(loader),  total_mae/len(val_loader), total_ssim/len(val_loader)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

normalize = preprocessing.normalize_pixels(mean0=False)
val_mvd = MontevideoFoldersDataset(
                                    path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',
                                    in_channel=3,
                                    out_channel=1,
                                    min_time_diff=5,
                                    max_time_diff=15,
                                    csv_path=None,
                                    transform=normalize
                                    )


train_mvd = MontevideoFoldersDataset(
                                    path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',
                                    in_channel=3,
                                    out_channel=1,
                                    min_time_diff=5,
                                    max_time_diff=15,
                                    csv_path=None,
                                    transform=normalize
                                    )

batch_size=10

train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1

comment = f' batch_size:{batch_size}'
writer = SummaryWriter(log_dir='runs/phydnet' ,comment=comment)

phycell  =  PhyCell(input_shape=(64,64), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell =  ConvLSTM(input_shape=(64,64), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)

nepochs = 10
print_every = 1
eval_every = 1
model_name = ''

train_losses = trainIters(train_loader=train_loader,
                          val_loader=val_loader,
                          encoder=encoder,
                          nepochs=nepochs,
                          print_every=print_every,
                          eval_every=eval_every,
                          checkpoint=False,
                          model_name=model_name,
                          writer=writer)

if writer:
  writer.close()
