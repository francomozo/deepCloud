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
from src.lib.utils import save_gan_checkpoint
# move to package

# Paras and hyperparams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt'
CSV_PATH='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
LAMBDA_GP = 5
CRITIC_ITERATIONS = 5
FEATURES_D = 32

# Dataloaders
normalize = preprocessing.normalize_pixels()
train_ds = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',    
                                     in_channel=3,
                                     out_channel=1,
                                     min_time_diff=5, max_time_diff=15,
                                     csv_path=CSV_PATH,
                                     transform=normalize)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Nets
gen = UNet2(n_channels=3, n_classes=1, bilinear=True, filters=32).to(device)
disc = Discriminator(channels_img=1, features_d=FEATURES_D).to(device)

gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

# Initializate optimizer
#opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
#opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

gen.train()
disc.train()

# description of the experiment:
ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
exp_desc = f'lr({LEARNING_RATE})_opt(rmsprop)_lambda_gp({LAMBDA_GP})_load_dict(10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt)_features_d({FEATURES_D})_csv(train_cosangs_in3_out1)'


# tb
writer_gt = SummaryWriter(f"runs/{ts}/{exp_desc}/gt")
writer_pred = SummaryWriter(f"runs/{ts}/{exp_desc}/pred")
writer = SummaryWriter(f"runs/{ts}/{exp_desc}/loss")
step = 0

gen_loss_by_epochs = []
disc_loss_by_epochs = []
for epoch in range(NUM_EPOCHS):
    gen_epoch_loss_list = []
    disc_epoch_loss_list = []
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
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_disc.item():.4f}, loss G: {loss_gen.item():.4f}"
            )
            #print(f'{torch.mean(disc_gt)}, {-torch.mean(disc_pred)}, {LAMBDA_GP * gp}')

            with torch.no_grad():
                writer.add_scalar('Gen Loss', loss_gen, global_step=step)
                writer.add_scalar('Disc Loss', loss_disc, global_step=step)
                
                # print images to tb, disabled 
                img_grid_gt = torchvision.utils.make_grid(gt, normalize=True)
                img_grid_pred = torchvision.utils.make_grid(pred, normalize=True)  
                
                writer_gt.add_image("gt", img_grid_gt, global_step=step)
                writer_pred.add_image("pred", img_grid_pred, global_step=step)
            step += 1

        gen_epoch_loss_list.append(loss_gen.item())
        disc_epoch_loss_list.append(loss_disc.item())
    
    gen_loss_by_epochs.append(sum(gen_epoch_loss_list)/len(gen_epoch_loss_list))
    disc_loss_by_epochs.append(sum(disc_epoch_loss_list)/len(disc_epoch_loss_list))
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}. Gen_epoch_loss: {gen_loss_by_epochs[-1]}, Disc_epoch_loss: {disc_loss_by_epochs[-1]}')


save_gan_checkpoint(gen, disc, opt_gen, opt_disc, NUM_EPOCHS, gen_loss_by_epochs, disc_loss_by_epochs, exp_desc)
