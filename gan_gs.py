import random

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import preprocessing, train
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet2

# Paras and hyperparams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt'
CSV_PATH='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
LAMBDA_GP = 0
#CRITIC_ITERATIONS = 5


# utils.py
def gradient_penalty(disc, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate disc scores
    mixed_scores = disc(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

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
disc = UNet2(n_channels=1, n_classes=1, bilinear=True, filters=32).to(device)
# gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])
#disc.load_state_dict(torch.load(PT_PATH)["model_state_dict"])
#gen.apply(train.weights_init)
#disc.apply(train.weights_init)

# Initializate optimizer
#opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
#opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)

gen.train()
disc.train()

# tb
writer_gt = SummaryWriter(f"runs/gan/gt")
writer_pred = SummaryWriter(f"runs/gan/pred")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx, (in_frames, gt) in enumerate(train_loader):
        
        in_frames = in_frames.to(device) # this is the noise (B, C, H, W)=(1, 3, 256, 256)
        gt = gt.to(device) # this is the real (1, 1, 256, 256))
        cur_batch_size = gt.shape[0]
        # HERE THE PAPER TRAINS A FEW ITER THE DISC BUT I WONT DO THAT
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        # for _ in range(CRITIC_ITERATIONS):
        #noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        
        pred = gen(in_frames) # (1, 1, 256, 256)
        
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
                img_grid_gt = torchvision.utils.make_grid(gt, normalize=True)
                img_grid_pred = torchvision.utils.make_grid(pred, normalize=True)  
                
                writer_gt.add_image("gt", img_grid_gt, global_step=step)
                writer_pred.add_image("pred", img_grid_pred, global_step=step)
            step += 1
