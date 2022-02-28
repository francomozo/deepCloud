import copy
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from src import evaluate, preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.gan import Discriminator
from src.dl_models.unet import UNet2
from src.lib.utils import gradient_penalty, save_checkpoint

# -----------------------------------------------------------------+
# 19 feb 2022
# This is the new default script to train gans on mvd
# Differences:
#   -> the Unet models have 16 filters DONE
#   -> models will be trained to 30, 60 and 90 min PENDING
#   -> the dataloaders use csvs in a different way DONE
#   -> evaluate.evaluate_gan_val evaluation horizon PENDING, debug with pdb
#   -> the static grid images starts before epoch 1 (and possibly
#       generate a different dataset with chosen images for this(didnt do this)) DONE
#   -> modify evaluate.make_val_grid DONE
#   -> disable the grid gt vs pred for tb to save time DONE
#   -> save best model, and model in last epoch DONE
#   -> save the image grid to mem DONE
#   -> print time elapsed training the ith modelDONE

expId = 'baab' # global experiment Id. string TODO: complete the expId

# Params =======================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')
torch.manual_seed(50)

ROOT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/'
PT_PATH = ROOT_PATH + 'checkpoints/MVD/30min/30min_UNET2_mvd_mae_filters16_sigmoid_diffFalse_retrainFalse_34_16-02-2022_11:26_BEST_FINAL.pt' # TODO:complete path depending on the ph
DATA_PATH_TRAIN = ROOT_PATH + '/data/mvd/train/'
CSV_PATH_TRAIN = ROOT_PATH + '/data/mvd/train_cosangs_mvd.csv'

DATA_PATH_VAL = ROOT_PATH + '/data/mvd/validation/'
CSV_PATH_VAL = ROOT_PATH + '/data/mvd/val_cosangs_mvd.csv' 

BATCH_SIZE = 12 #16 # fixed
NUM_EPOCHS = 40 # TODO 
PREDICT_HORIZON = 3 # int corresponding to num output images, ph=30min is 3

SAVE_STATIC_SEQUENCES = True
sub_expId = 0


outputs = {}

models = {
    'aaag5' : {
        'lr' : 0.0001, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 32, 'use_critic_iter' : True},
    'aaag8' : {
        'lr' : 0.0001, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 32, 'use_critic_iter' : False},
    'aaah6' : {
        'lr' : 0.0003, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 32, 'use_critic_iter' : True},
    'aaah8' : {
        'lr' : 0.0003, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 32, 'use_critic_iter' : False},
    'aaah10' : {
        'lr' : 0.0003, 'lambda_gp' : 5, 'critic_iter' : 10,  'features_d' : 16, 'use_critic_iter' : True},
    'aaah13' : {
        'lr' : 0.0003, 'lambda_gp' : 5, 'critic_iter' : 10,  'features_d' : 32, 'use_critic_iter' : True},
    'aaaj4' : {
        'lr' : 5e-05, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 16, 'use_critic_iter' : False},
    'aaaj5' : {
        'lr' : 5e-05, 'lambda_gp' : 5, 'critic_iter' : 5,  'features_d' : 32, 'use_critic_iter' : True},
    'aaaj12' : {
        'lr' : 5e-05, 'lambda_gp' : 5, 'critic_iter' : 10,  'features_d' : 16, 'use_critic_iter' : False},
    'aaan2' : {
        'lr' : 5e-05, 'lambda_gp' : 10, 'critic_iter' : 5,  'features_d' : 16, 'use_critic_iter' : True}
}

total_exps = len(models)
for index, key in enumerate(models.keys()):    
    sub_expId = index + 1
    
    lr = models[key]['lr']
    lambda_gp = models[key]['lambda_gp']
    critic_iter = models[key]['critic_iter']
    features_d = models[key]['features_d']
    use_critic_iter = models[key]['use_critic_iter']
    
    print(f'=================================================================================================================================')
    print(f'Exp {sub_expId}/{total_exps}.')
    print(f'ExpId: {expId}{sub_expId} ({key}). lr({lr})_lambda_gp({lambda_gp})_critic_iter({critic_iter})_features_d({features_d})_use_critic_iter({use_critic_iter}).')
    #try:
    if True:   
        # Hyperparams =======================
        LEARNING_RATE = lr
        LAMBDA_GP = lambda_gp
        CRITIC_ITERATIONS = critic_iter
        FEATURES_D = features_d

        # Dataloaders =======================
        normalize = preprocessing.normalize_pixels()
        train_mvd = MontevideoFoldersDataset(path=DATA_PATH_TRAIN, 
                                             in_channel=3, 
                                             out_channel=PREDICT_HORIZON,
                                             min_time_diff=5, 
                                             max_time_diff=15,
                                             csv_path=CSV_PATH_TRAIN,
                                             transform=normalize, 
                                             output_last=True) 
        train_loader = DataLoader(train_mvd, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        val_mvd = MontevideoFoldersDataset(path=DATA_PATH_VAL, 
                                           in_channel=3, 
                                           out_channel=PREDICT_HORIZON,
                                           min_time_diff=5, 
                                           max_time_diff=15,
                                           csv_path=CSV_PATH_VAL,
                                           transform=normalize, 
                                           output_last=True)

        val_loader = DataLoader(val_mvd)

        # Nets =======================
        disc = Discriminator(channels_img=1, features_d=FEATURES_D).to(device)
        gen = UNet2(n_channels=3, n_classes=1, bilinear=True, bias=False, filters=16).to(device)
        
        gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

        # Initializate optimizer
        opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

        gen.train()
        disc.train()

        # Auxiliar variables =======================
        best_val_loss = 1e3 
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        if os.path.isdir(f'{os.getcwd()}/checkpoints/{expId}') == False: 
            os.mkdir(f'{os.getcwd()}/checkpoints/{expId}')
        
        if SAVE_STATIC_SEQUENCES:
            if os.path.isdir(f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}') == False: 
                os.mkdir(f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}')

        # tb
        # writer_gt = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/gt") DISABLE FOR SPEED PURPOSES
        # writer_pred = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/pred")
        writer_static = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/static")
        writer_loss = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/loss")
        step = 0

        gen_loss_by_epochs = []
        disc_loss_by_epochs = []
        
        experiment_output = {}
        
        start = time.time()
        for epoch in range(NUM_EPOCHS):
            gen_epoch_loss_list = []
            disc_epoch_loss_list = []
            epoch_output = {}
            print(f'==> Epoch {epoch+1}/{NUM_EPOCHS}') 
            
            with torch.no_grad():
                grid = evaluate.make_val_grid(gen, sequences=4, device=device, val_mvd=val_mvd)
                writer_static.add_image("static_imgs", grid, global_step=epoch) # epoch corresponds to the actual epoch
                                                                                # epoch=0 is before training 
                if SAVE_STATIC_SEQUENCES:
                    save_image(grid, f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}/{expId}{sub_expId}_epoch{epoch}.png')
                    torch.save(grid, f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}/{expId}{sub_expId}_epoch{epoch}.pt')

            gen.train() 

            for batch_idx, (in_frames, gt) in enumerate(train_loader):
                
                in_frames = in_frames.to(device)
                gt = gt.to(device)
                
                # Train Critic: max E[critic(real)] - E[critic(fake)]
                pred = gen(in_frames)
                disc_pred = disc(pred).reshape(-1)
                disc_gt = disc(gt).reshape(-1)
                gp = gradient_penalty(disc, gt, pred, device=device)
                loss_disc = (
                    -(torch.mean(disc_gt) - torch.mean(disc_pred)) + LAMBDA_GP * gp
                )
                disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                opt_disc.step()

                # Train Generator: max E[disc(gen_noise)] <-> min -E[disc(gen_noise)]
                if use_critic_iter:
                    if batch_idx % CRITIC_ITERATIONS == 0:
                        disc_pred = disc(pred).reshape(-1)
                        loss_gen = -torch.mean(disc_pred)
                        gen.zero_grad()
                        loss_gen.backward()
                        opt_gen.step()
                else: 
                    disc_pred = disc(pred).reshape(-1)
                    loss_gen = -torch.mean(disc_pred)
                    gen.zero_grad()
                    loss_gen.backward()
                    opt_gen.step()

                if batch_idx % 300 == 0 and batch_idx > 0:
                    with torch.no_grad():
                        writer_loss.add_scalar('batch_loss/Gen', loss_gen, global_step=step)
                        writer_loss.add_scalar('batch_loss/Disc', loss_disc, global_step=step)
                        
                        # img_grid_gt = torchvision.utils.make_grid(gt.detach()[:4], normalize=True)
                        #img_grid_pred = torchvision.utils.make_grid(pred.detach()[:4], normalize=True)  
                        # writer_gt.add_image("gt", img_grid_gt, global_step=step) DISABLE FOR SPEED PURPOSES
                        # writer_pred.add_image("pred", img_grid_pred, global_step=step)
                    step += 1
                
                # for each batch save losses to calculate mean for each epoch 
                gen_epoch_loss_list.append(loss_gen.detach().item())
                disc_epoch_loss_list.append(loss_disc.detach().item())

            # calc train loss by epoch
            gen_loss_by_epochs.append(sum(gen_epoch_loss_list)/len(gen_epoch_loss_list))
            disc_loss_by_epochs.append(sum(disc_epoch_loss_list)/len(disc_epoch_loss_list))

            # validation each epoch
            gen.eval()
            val_loss = evaluate.evaluate_gan_val(gen, val_loader, 
                                                predict_horizon=1, 
                                                device=device,
                                                metric='SSIM')
            val_loss = np.mean(val_loss, axis=0)[0] # single value, varies [-1, 1] 
            
            # for val_loss i want the model to reach 1 in each entry, so im looking to maximize, instead i
            # will modify so that the best model is when the values go to zero, so first substract 1 so that
            # [-2, 0] and then multiply by -1 so [2, 0], 0 eq to 1 and 2 to -1
            val_loss_modified = (val_loss -1)*(-1)
            
            print(f'\t -> Disc_epoch_loss: {disc_loss_by_epochs[-1]:.8f}. Gen_epoch_loss: {gen_loss_by_epochs[-1]:.8f}. Val_loss: {val_loss:.8f}.')

            # tb losses by epoch
            with torch.no_grad():
                # obs:  the next to lines were moved to the beginning of the epoch to print the first sequence without
                #       an epoch of gan model training
                # grid = evaluate.make_val_grid(gen, sequences=3, device=device, val_mvd=val_mvd) 
                # writer_static.add_image("static_imgs", grid, global_step=epoch+1)
                writer_loss.add_scalar('epoch_loss/Gen', loss_gen, global_step=epoch+1)
                writer_loss.add_scalar('epoch_loss/Disc', loss_disc, global_step=epoch+1)
                writer_loss.add_scalar('epoch_loss/Val_SSIM', val_loss, global_step=epoch+1)

            # save best model (on val)
            if val_loss_modified < best_val_loss: # this part save the gen and dict after one epoch of training
                if best_val_loss != 1e3:
                    print(f'\t -> New best model!!!')
                gen_dict = {
                    'epoch': epoch+1,
                    'model_state_dict': copy.deepcopy(gen.state_dict()),
                    'opt_state_dict': copy.deepcopy(opt_gen.state_dict()),
                    'gen_epoch_loss': gen_loss_by_epochs,
                }
                disc_dict = {
                        'epoch': epoch+1,
                        'model_state_dict': copy.deepcopy(disc.state_dict()),
                        'opt_state_dict': copy.deepcopy(opt_disc.state_dict()),
                        'disc_epoch_loss': disc_loss_by_epochs,
                }
                best_val_loss = val_loss_modified


            epoch_output = {
                'disc_loss' : disc_loss_by_epochs[-1],
                'gen_loss' : gen_loss_by_epochs[-1],
                'val_loss' : val_loss,
            }
            
            actual_epoch = epoch+1
            experiment_output[actual_epoch] = epoch_output

        exp = expId  + str(sub_expId)
        outputs[exp] = experiment_output

        # save last epoch static images        
        with torch.no_grad():
            grid = evaluate.make_val_grid(gen, sequences=4, device=device, val_mvd=val_mvd)
            writer_static.add_image("static_imgs", grid, global_step=epoch+1) # last epoch 
                                                                                 
            if SAVE_STATIC_SEQUENCES:
                save_image(grid, f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}/{expId}{sub_expId}_epoch{epoch+1}.png')
                torch.save(grid, f'{os.getcwd()}/reports/gan_exp_outputs/static_sequences/{expId}/{expId}{sub_expId}_epoch{epoch+1}.pt')                

        # save best model checkpoint
        save_checkpoint(gen_dict, disc_dict, expId, sub_expId, obs='BEST')       
        
        # save last epoch model checkpoint
        gen_dict = {
            'epoch': epoch+1,
            'model_state_dict': copy.deepcopy(gen.state_dict()),
            'opt_state_dict': copy.deepcopy(opt_gen.state_dict()),
            'gen_epoch_loss': gen_loss_by_epochs,
        }
        disc_dict = {
                'epoch': epoch+1,
                'model_state_dict': copy.deepcopy(disc.state_dict()),
                'opt_state_dict': copy.deepcopy(opt_disc.state_dict()),
                'disc_epoch_loss': disc_loss_by_epochs,
        }

        save_checkpoint(gen_dict, disc_dict, expId, sub_expId)

        # print elapsed time while training this model
        end = time.time()
        print(f'Time elapse: {int((end - start)/60)} minutes.')
    #except:
    #    print('Experiment failed :(')

with open(f'reports/gan_exp_outputs/{expId}.json', 'w') as fp:
   json.dump(outputs, fp)
