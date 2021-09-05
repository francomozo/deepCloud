import datetime
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import evaluate, preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.gan import Discriminator
from src.dl_models.unet import UNet2
from src.lib.utils import gradient_penalty, save_checkpoint

expId = # global experiment Id. string
objective_loss = 0.05803579

# Params =======================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_2_50_09-07-2021_06:21.pt'
DATA_PATH_TRAIN = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/'
DATA_PATH_VAL = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/'

BATCH_SIZE = 12 # fixed
NUM_EPOCHS = 2 # fixed
predict_horizon = 6 # this is for validation
sub_expId = 0

# tuneable params:
lrs = [1e-4, 3e-4] # floats
critic_iters = [5] # ints
lambda_gps = [5] # ints
features_ds = [32] # ints
load_dict_flags = [True] # bool
use_critic_iters = [True] # bool
csv_on_trains = [True] # bool

hparams = [(lr, lambda_gp, critic_iter, features_d, load_dict_flag, use_critic_iter, csv_on_train) for lr in lrs
                                                                                                   for lambda_gp in lambda_gps
                                                                                                   for critic_iter in critic_iters
                                                                                                   for features_d in features_ds
                                                                                                   for load_dict_flag in load_dict_flags
                                                                                                   for use_critic_iter in use_critic_iters
                                                                                                   for csv_on_train in csv_on_trains]
total_exps = len(hparams)
for index, (lr, lambda_gp, critic_iter, features_d, load_dict_flag, use_critic_iter, csv_on_train) in enumerate(hparams):
    
    sub_expId = index + 1

    print(f'=================================================================================================================================')
    print(f'Exp {sub_expId}/{total_exps}.')
    print(f'ExpId: {expId}{sub_expId}. lr({lr})_lambda_gp({lambda_gp})_critic_iter({critic_iter})_features_d({features_d})_load_dict_flag({load_dict_flag})_use_critic_iter({use_critic_iter})_csv_on_train({csv_on_train}).')
    try:
        if csv_on_train:
            CSV_PATH_TRAIN = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
        else:
            CSV_PATH_TRAIN = None
        csv_on_val = False # This is fixed for fear comparisons between experiments
        if csv_on_val:
            CSV_PATH_VAL = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/val_cosangs_in3_out6.csv' 
        else:
            CSV_PATH_VAL = None

        
        # Hyperparams =======================
        LEARNING_RATE = lr
        LAMBDA_GP = lambda_gp
        CRITIC_ITERATIONS = critic_iter
        FEATURES_D = features_d

        # Dataloaders =======================
        normalize = preprocessing.normalize_pixels()
        train_ds = MontevideoFoldersDataset(path=DATA_PATH_TRAIN, in_channel=3, out_channel=1,
                                            min_time_diff=5, max_time_diff=15,
                                            csv_path=CSV_PATH_TRAIN,
                                            transform=normalize)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        val_mvd = MontevideoFoldersDataset(path=DATA_PATH_VAL, 
                                                in_channel=3, out_channel=predict_horizon,
                                                min_time_diff=5, max_time_diff=15,
                                                transform=normalize, 
                                                csv_path=CSV_PATH_VAL)

        val_loader = DataLoader(val_mvd)

        # Nets =======================
        #gen = UNet2(n_channels=3, n_classes=1, bilinear=True, filters=32).to(device)
        disc = Discriminator(channels_img=1, features_d=FEATURES_D).to(device)
        gen = UNet2(n_channels=3, n_classes=1, bilinear=True, bias=True).to(device)
        
        if load_dict_flag:
            gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

        # Initializate optimizer
        opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

        gen.train()
        disc.train()

        # Auxiliar variables =======================
        criterion = nn.MSELoss()
        best_val_loss = 1e3
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        if os.path.isdir(f'{os.getcwd()}/checkpoints/{expId}') == False: 
            os.mkdir(f'{os.getcwd()}/checkpoints/{expId}')

        # tb
        writer_gt = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/gt")
        writer_pred = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/pred")
        writer_static = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/static")
        writer_loss = SummaryWriter(f"runs/{expId}/{expId}{sub_expId}/loss")
        step = 0

        gen_loss_by_epochs = []
        disc_loss_by_epochs = []
        for epoch in range(NUM_EPOCHS):
            gen_epoch_loss_list = []
            disc_epoch_loss_list = []
            print(f'==> Epoch {epoch+1}/{NUM_EPOCHS}') 
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
                        
                        img_grid_gt = torchvision.utils.make_grid(gt.detach()[:4], normalize=True)
                        img_grid_pred = torchvision.utils.make_grid(pred.detach()[:4], normalize=True)  
                        
                        writer_gt.add_image("gt", img_grid_gt, global_step=step)
                        writer_pred.add_image("pred", img_grid_pred, global_step=step)
                    step += 1
                
                # for each batch save losses to calculate mean for each epoch 
                gen_epoch_loss_list.append(loss_gen.detach().item())
                disc_epoch_loss_list.append(loss_disc.detach().item())

            # calc train loss by epoch
            gen_loss_by_epochs.append(sum(gen_epoch_loss_list)/len(gen_epoch_loss_list))
            disc_loss_by_epochs.append(sum(disc_epoch_loss_list)/len(disc_epoch_loss_list))

            # validation each epoch
            gen.eval()
            val_err_array = evaluate.evaluate_gan_val(gen, val_loader, 
                                                    predict_horizon=predict_horizon, 
                                                    device=device,
                                                    metric='RMSE')
            val_err_array = np.mean(val_err_array, axis=0) # vector of length 'predict_horizon'
            
            print(f'\t -> Disc_epoch_loss: {disc_loss_by_epochs[-1]:.8f}. Gen_epoch_loss: {gen_loss_by_epochs[-1]:.8f}. Val_loss: {val_err_array[0]:.8f}.')
            print(f'\t -> Val_pred_horiz_loss: {val_err_array}')

            # tb losses by epoch
            with torch.no_grad():
                grid = evaluate.make_val_grid(gen, sequences=3, device=device)
                writer_static.add_image("static_imgs", grid, global_step=epoch+1)
                
                writer_loss.add_scalar('epoch_loss/Gen', loss_gen, global_step=epoch+1)
                writer_loss.add_scalar('epoch_loss/Disc', loss_disc, global_step=epoch+1)
                writer_loss.add_scalar('epoch_loss/Val_RMSE', val_err_array[0], global_step=epoch+1)

            # save best model (on val)
            if val_err_array[0] < best_val_loss:
                if best_val_loss != 1e3:
                    print(f'\t -> New best model!!!')
                gen_dict = {
                    'epoch': epoch+1,
                    'model_state_dict': gen.state_dict(),
                    'opt_state_dict': opt_gen.state_dict(),
                    'gen_epoch_loss': gen_loss_by_epochs,
                }
                disc_dict = {
                        'epoch': epoch+1,
                        'model_state_dict': disc.state_dict(),
                        'opt_state_dict': opt_disc.state_dict(),
                        'disc_epoch_loss': disc_loss_by_epochs,
                }
                best_val_loss = val_err_array[0]

        save_checkpoint(gen_dict, disc_dict, expId, sub_expId)       
    except:
        print('Experiment failed :(')
