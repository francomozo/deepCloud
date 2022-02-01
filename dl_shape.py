import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from src import evaluate, preprocessing
from src.data import MontevideoFoldersDataset
from src.dl_models.gan import Discriminator
from src.dl_models.unet import UNet2
from src.lib.utils import gradient_penalty, save_checkpoint

# Hyperparams =======================
LEARNING_RATE = 1e-4
BATCH_SIZE = 12
NUM_EPOCHS = 3
LAMBDA_GP = 5
CRITIC_ITERATIONS = 5
FEATURES_D = 32
DATA_PATH_TRAIN = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/'
CSV_PATH_TRAIN = None


normalize = preprocessing.normalize_pixels()
train_ds = MontevideoFoldersDataset(path=DATA_PATH_TRAIN, in_channel=3, out_channel=1,
                                    min_time_diff=5, max_time_diff=15,
                                    csv_path=CSV_PATH_TRAIN,
                                    transform=normalize)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


dataIter = iter(train_loader)


out = next(dataIter)
print(out.shape)
