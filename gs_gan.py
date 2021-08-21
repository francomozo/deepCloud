import random

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.gan import Discriminator
from src.dl_models.unet import UNet2
from src.lib.utils import gradient_penalty

# Paras and hyperparams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt'
CSV_PATH='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
BATCH_SIZE = 8
NUM_EPOCHS = 5
#LEARNING_RATE = 1e-4
#LAMBDA_GP = 10
CRITIC_ITERATIONS = 5
#FEATURES_D = 64

# Dataloaders
normalize = preprocessing.normalize_pixels()
train_ds = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',    
                                     in_channel=3,
                                     out_channel=1,
                                     min_time_diff=5, max_time_diff=15,
                                     csv_path=CSV_PATH,
                                     transform=normalize)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# here starts the gs
lrs = [1e-4, 3e-4, 5e-5]
lambda_gps = [10]
optimizers = ['adam', 'rmsprop']
load_dicts = [True, False]
features_ds = [32, 64]

hparams = [(lr, lambda_gp, optim_type, load_dict, features_d) for lr in lrs
                                                              for lambda_gp in lambda_gps
                                                              for optim_type in optimizers
                                                              for load_dict in load_dicts
                                                              for features_d in features_ds]

total_gs = len(hparams)
for index, (LEARNING_RATE, LAMBDA_GP, optim_type, load_dict, FEATURES_D) in enumerate(hparams):
    
    exp_str = f'lr({LEARNING_RATE})_lambda_gp({LAMBDA_GP})_optim_type({optim_type})_load_dict({load_dict})_features_d({FEATURES_D})'
    
    print('========================================================================')
    print(f'Iter {index+1}/{total_gs}:  {exp_str}')

    # Nets
    gen = UNet2(n_channels=3, n_classes=1, bilinear=True, filters=32).to(device)
    disc = Discriminator(channels_img=1, features_d=FEATURES_D).to(device)

    if load_dict:
        gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

    # Initializate optimizer
    if optim_type == 'adam':
        opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
        opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    else:
        opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
        opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

    gen.train()
    disc.train()

    # tb
    #writer_gt = SummaryWriter(f"runs/gan-30_epochs-critic_iters/gt")
    #writer_pred = SummaryWriter(f"runs/gan-30_epochs-critic_iters/pred")
    writer = SummaryWriter(f'runs/gs_gan/' + exp_str)
    step = 0

    for epoch in range(NUM_EPOCHS):
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

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                #print(f'{torch.mean(disc_gt)}, {-torch.mean(disc_pred)}, {LAMBDA_GP * gp}')

                with torch.no_grad():
                    writer.add_scalar('Gen Loss', loss_gen, global_step=step)
                    writer.add_scalar('Disc Loss', loss_disc, global_step=step)
                    
                    
                    # print images to tb, disabled 
                    #img_grid_gt = torchvision.utils.make_grid(gt, normalize=True)
                    #img_grid_pred = torchvision.utils.make_grid(pred, normalize=True)  
                    
                    #writer_gt.add_image("gt", img_grid_gt, global_step=step)
                    #writer_pred.add_image("pred", img_grid_pred, global_step=step)
                step += 1
