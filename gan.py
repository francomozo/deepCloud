import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src import preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet2

# Hyperparams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(50)
BATCH_SIZE = 32
PT_PATH = None
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
#CRITIC_ITERATIONS = None
LAMBDA_GP = 10

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
                                     transform=normalize)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Nets
gen = UNet2(n_channels=3, n_classes=1, bilinear=True,filters=32).to(device)
disc = UNet2(n_channels=3, n_classes=1, bilinear=True,filters=32).to(device)
gen.load_state_dict(torch.load(PT_PATH)["model_state_dict"])
disc.load_state_dict(torch.load(PT_PATH)["model_state_dict"])

# Initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
disc.train()


for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    # for batch_idx, (real, _) in enumerate(train_loader):
    for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
        in_frames = in_frames.to(device) # this is the noise
        real = out_frames.to(device) # this is the real
    
        cur_batch_size = real.shape[0]
        # HERE THE PAPER TRAINS A FEW ITER THE DISC BUT I WONT DO THAT
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        # for _ in range(CRITIC_ITERATIONS):
        #noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        
        fake = gen(in_frames)
        disc_real = disc(real).reshape(-1)
        disc_fake = disc(fake).reshape(-1)
        gp = gradient_penalty(disc, real, fake, device=device)
        loss_critic = (
            -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp
        )
        disc.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = disc(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

# PASOS
# crear dataloader for train
# levantar alguna unet para arrancar con esos pesos
# crear las gans
# entrenar
