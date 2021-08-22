import datetime
import random

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

exp_num = 0

# Params =======================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt'

DATA_PATH_TRAIN = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/'
DATA_PATH_VAL = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation'
csv_on_train = True
if csv_on_train:
    CSV_PATH_TRAIN = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
else:
    CSV_PATH_TRAIN = None
csv_on_val = True
if csv_on_val:
    CSV_PATH_VAL = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/val_cosangs_in3_out6.csv' 
else:
    CSV_PATH_VAL = None

predict_horizon = 6 # this is for validation

    
# Hyperparams =======================
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 30
LAMBDA_GP = 5
CRITIC_ITERATIONS = 5
FEATURES_D = 32

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
gen = UNet2(n_channels=3, n_classes=1, bilinear=True, filters=32).to(device)
disc = Discriminator(channels_img=1, features_d=FEATURES_D).to(device)

gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

# Initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

gen.train()
disc.train()

# Auxiliar variables =======================
criterion = nn.MSELoss()
best_val_loss = 1e3

# description of the experiment:
ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
exp_desc = f'lr({LEARNING_RATE})_opt(rmsprop)_lambda_gp({LAMBDA_GP})_load_dict(10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt)_features_d({FEATURES_D})_csv(train_cosangs_in3_out1)'

# tb
#writer_gt = SummaryWriter(f"runs/exp{exp_num}_{ts}/{exp_desc}/gt")
#writer_pred = SummaryWriter(f"runs/exp{exp_num}_{ts}/{exp_desc}/pred")
#writer = SummaryWriter(f"runs/exp{exp_num}_{ts}/{exp_desc}/loss")
#step = 0

gen_loss_by_epochs = []
disc_loss_by_epochs = []
for epoch in range(NUM_EPOCHS):
    gen_epoch_loss_list = []
    disc_epoch_loss_list = []
    gen.train()
    print(f'== Epoch {epoch+1}/{NUM_EPOCHS}') 
    
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
        if batch_idx % CRITIC_ITERATIONS == 0:
            disc_pred = disc(pred).reshape(-1)
            loss_gen = -torch.mean(disc_pred)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # if batch_idx % 100 == 0 and batch_idx > 0:
        #     with torch.no_grad():
        #         writer.add_scalar('Gen Loss', loss_gen, global_step=step)
        #         writer.add_scalar('Disc Loss', loss_disc, global_step=step)
                
        #         img_grid_gt = torchvision.utils.make_grid(gt, normalize=True)
        #         img_grid_pred = torchvision.utils.make_grid(pred, normalize=True)  
                
        #         writer_gt.add_image("gt", img_grid_gt, global_step=step)
        #         writer_pred.add_image("pred", img_grid_pred, global_step=step)
        #     step += 1
        
        # for each batch save losses to calculate mean for each epoch 
        gen_epoch_loss_list.append(loss_gen.detach().item())
        disc_epoch_loss_list.append(loss_disc.detach().item())

    # calc train loss by epoch
    gen_loss_by_epochs.append(sum(gen_epoch_loss_list)/len(gen_epoch_loss_list))
    disc_loss_by_epochs.append(sum(disc_epoch_loss_list)/len(disc_epoch_loss_list))

    # validation each epoch
    val_err_array = evaluate.evaluate_model(gen, val_loader, 
                                            predict_horizon=predict_horizon, 
                                            device=device,
                                            metric='RMSE')
    val_err_mean = np.mean(val_err_array, axis=0)
    
    # print losses (gen, disc, val_loss)
    print(f'\t -> Gen_epoch_loss: {gen_loss_by_epochs[-1]:.8f}, Disc_epoch_loss: {disc_loss_by_epochs[-1]:.8f}. Val_loss: {val_err_mean:.8f}.')
    print(f'\t -> Val_loss_1hr: {val_err_array}')
    
    
#     # save best model (on val)
#     if val_loss < best_val_loss:
#         if best_val_loss != 1e3:
#             print(f'\t -> New best model!!!')
#         gen_dict = {
#             'epoch': epoch+1,
#             'model_state_dict': gen.state_dict(),
#             'opt_state_dict': opt_gen.state_dict(),
#             'gen_epoch_loss': gen_loss_by_epochs,
#         }

#         disc_dict = {
#                 'epoch': epoch+1,
#                 'model_state_dict': disc.state_dict(),
#                 'opt_state_dict': opt_disc.state_dict(),
#                 'disc_epoch_loss': disc_loss_by_epochs,
#         }
#         best_val_loss = val_loss


# save_checkpoint(gen_dict, disc_dict, exp_desc, exp_num)       
