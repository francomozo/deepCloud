
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet2
from src.lib.utils import gradient_penalty


# Paras and hyperparams
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(50)
PT_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/10min_UNet2_sigmoid_mae_f32_60_04-08-2021_20:43.pt'
CSV_PATH='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_in3_out1.csv'
OUTPUT_FILE_PATH = '/clusteruy/home/franco.mozo/outputs_gan_gs.txt'
BATCH_SIZE = 1
NUM_EPOCHS = 1
CRITIC_ITERATIONS = 5


# GRID SEARCH
lrs = [1e-4, 3e-4, 1e-5, 5e-5]
lambda_gps = [0, 1, 5, 10]
critic_iters = [True, False]
optimizers = ['rmsprop', 'adam']
load_dicts = [False]

hparams = [(lr, lambda_gp, critic_iter, optim_type, load_dict) for lr in lrs
                                               for lambda_gp in lambda_gps
                                               for critic_iter in critic_iters
                                               for optim_type in optimizers
                                               for load_dict in load_dicts]
total_gs = len(hparams)
for index, (LEARNING_RATE, LAMBDA_GP, critic_iter, optim_type, load_dict) in enumerate(hparams):
    print('========================================================================')
    print(
        f'Iter {index+1}/{total_gs}. \
        Hparams: lr={LEARNING_RATE}, lambda_gp={LAMBDA_GP}, critic_iter={critic_iter}, optim_type={optim_type}, load_dict={load_dict}.'
    )

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
    writer_gt = SummaryWriter(f"runs/gan/gs/lr({LEARNING_RATE})-lambda({LAMBDA_GP})-critic_iter({critic_iter})-optim_type({optim_type})-load_dict({load_dict})/gt")
    writer_pred = SummaryWriter(f"runs/gan/gs/lr({LEARNING_RATE})-lambda({LAMBDA_GP})-critic_iter({critic_iter})-optim_type({    optim_type})-load_dict({load_dict})/pred")
    step = 0

    for epoch in range(NUM_EPOCHS):
        losses_gen = []
        losses_disc = []
        for batch_idx, (in_frames, gt) in enumerate(train_loader):
            
            in_frames = in_frames.to(device) # this is the noise (B, C, H, W)=(1, 3, 256, 256)
            gt = gt.to(device) # this is the real (1, 1, 256, 256))
            cur_batch_size = gt.shape[0]
            # Train Critic: max E[critic(real)] - E[critic(fake)]
           
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

            # only update generator every CRITIC_ITERATIONS iterations if critic_iter true
            # else, do it every iteration (there should be a better way to do this)
            if critic_iter == True: # ie: Train more the critic (or discriminator)
                if batch_idx % CRITIC_ITERATIONS == 0:
                    # Train Generator: max E[disc(gen_noise)] <-> min -E[disc(gen_noise)]
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
            
            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                losses_gen.append(loss_gen)
                losses_disc.append(loss_disc)

                with torch.no_grad():
                    img_grid_gt = torchvision.utils.make_grid(gt, normalize=True)
                    img_grid_pred = torchvision.utils.make_grid(pred, normalize=True)  
                    
                    writer_gt.add_image("gt", img_grid_gt, global_step=step)
                    writer_pred.add_image("pred", img_grid_pred, global_step=step)
                step += 1
        # write to file the best and worst losses
        output_file = open(OUTPUT_FILE_PATH,"a")
        
        max_gen, min_gen = max(losses_gen), min(losses_gen)
        max_disc, min_disc = max(losses_disc), min(losses_disc)
        gen_loss, disc_loss = losses_gen[-1], losses_disc[-1]

        string1 = f'Iter {index+1}/{total_gs} ========================================== \n' 
        string2 = f'Hparams: lr={LEARNING_RATE}, lambda_gp={LAMBDA_GP}, critic_iter={critic_iter}, optim_type={optim_type}, load_dict={load_dict}. \n'
        
        string3 = f'\t max_gen={max_gen}, min_gen={min_gen} \n\t max_disc={max_disc}, min_disc={min_disc} \n\n'
        string4 = f'\t last_gen_loss={gen_loss} \n\t last_disc_loss={disc_loss} \n\n'
        
        output_file.write(string1)
        output_file.write(string2)
        output_file.write(string3)
        output_file.write(string4)
        output_file.close()
