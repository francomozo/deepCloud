# USAGE:
#   Training loops and checkpoint saving
#
import datetime
import time

import numpy as np
import optuna
import torch

from src.lib.utils import print_cuda_memory


def train_model(model,
                criterion,
                optimizer,
                device,
                train_loader,
                epochs,
                val_loader,
                num_val_samples=10,
                checkpoint_every=None,
                verbose=True,
                print_every=100,
                ):

    # TODO: - save best acc model DONE(revision pending)
    #       - decide what the function returns
    #       - docstring

    TRAIN_LOSS = []
    VAL_LOSS = []
    # TIME = []

    BEST_VAL_ACC = 1e5

    for epoch in range(epochs):
        start_epoch = time.time()

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            # start_batch = time.time()

            # data to cuda if possible
            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # end_batch = time.time()
            # TIME.append(end_batch - start_batch)

            TRAIN_LOSS.append(loss.item())

            if verbose and batch_idx > 0 and batch_idx % print_every == 0:
                model.eval()
                # start_val = time.time()

                with torch.no_grad():
                    for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

                        in_frames = in_frames.to(device=device)
                        out_frames = out_frames.to(device=device)

                        frames_pred = model(in_frames)
                        val_loss = criterion(frames_pred, out_frames)

                        VAL_LOSS.append(val_loss.item())

                        if val_batch_idx == num_val_samples:
                            break

                # end_val = time.time()
                # val_time = end_val - start_val
                CURRENT_VAL_ACC = sum(VAL_LOSS)/len(VAL_LOSS)

                # print statistics
                print(
                    f'Epoch({epoch + 1}/{epochs}) | Batch({batch_idx:04d}/{len(train_loader)}) | ', end='')
                # , end='')
                print(
                    f'Train_loss({(loss.item()):06.2f}) | Val_loss({CURRENT_VAL_ACC:.2f})')
                # print(f'Time_per_batch({sum(TIME)/len(TIME):.2f}s) | Val_time({val_time:.2f}s)') # this part maybe dont print
                # TIME = []
                VAL_LOSS = []

        end_epoch = time.time()
        if verbose:
            print(
                f'Time elapsed in current epoch: {(end_epoch - start_epoch):.2f} secs.')

        if CURRENT_VAL_ACC < BEST_VAL_ACC:
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_per_batch': TRAIN_LOSS,
                'train_loss_epoch_mean': sum(TRAIN_LOSS)/len(TRAIN_LOSS)
            }
        else:
            model_dict = None

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            PATH = 'checkpoints/'
            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME = 'model_epoch' + str(epoch + 1) + '_' + str(ts) + '.pt'

            torch.save(model_dict, PATH + NAME)


def train_model_old(model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                    curr_epoch,
                    loss_history,
                    train_for=0,
                    verbose=True,
                    checkpoint_every=0,
                    print_cuda_mem=False,
                    loader_val=None,
                    trial=None):
    """Trains model, prints cuda mem, saves checkpoints, resumes training.

    Args:
        device : torch.device()
        curr_epoch : Current epoch the model is. Zero if it was never trained
        loss_history ([list]): Empty list if the model was never trained
        train_for (int, optional): Number of epochs to train. Defaults to 0.
        verbose (bool, optional): Print epoch counter and time of each epoch. Defaults to True.
        checkpoint_every (int, optional): Save checkpoint every "checkpoint_every" epochs. Defaults to 0.
        print_cuda_mem (bool, optional): Defaults to False.
        loader_val (optional): Pytorch Dataloader for validation
        trial (optional): optuna class for hyperparameters
    """

    loss_history_val = []
    val_step = 0
    num_val_samples = 400
    print_every = 100
    if not train_for:
        return loss_history
    last_epoch = train_for + curr_epoch

    curr_epoch += 1

    if print_cuda_mem:
        print_cuda_memory()

    while True:
        if verbose:
            print("EPOCH:", curr_epoch)
        start = time.time()
        start_batch = time.time()
        secuence_number = 0
        loss_total = 0
        loss_count = 0
        for id, (data, targets) in enumerate(loader):
            model.train()
            start_batch = time.time()
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)
            loss_total += loss.item()
            loss_count += 1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Validation
            secuence_number += data.shape[0]
            if loader_val is not None and secuence_number >= 400:
                model.eval()
                with torch.no_grad():
                    secuence_number = 0
                    loss_val_total = 0
                    loss_val_count = 0
                    for id_val, (data, targets) in enumerate(loader_val):
                        data = data.to(device=device)
                        targets = targets.to(device=device)

                        scores_val = model(data)
                        loss_val = criterion(scores_val, targets)
                        loss_val_total += loss_val.item()
                        loss_val_count += 1
                        if (id_val * data.shape[0] >= num_val_samples):
                            break

                    loss_val_average = loss_val_total/loss_val_count
                    loss_history_val.append(np.array(loss_val_average))
                    if trial is not None:
                        trial.report(loss_val_average, val_step)
                        val_step += 1
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
            end_batch = time.time()
            if verbose and (id+1) % print_every == 0:
                print('Iteration', id+1, '/', len(loader), ',loss = %.4f' % loss.item(), ',epoch_loss = %.4f' %
                      (loss_total/loss_count), ',Iteration time = %.2f' % ((end_batch-start_batch)/print_every), 's')
        if print_cuda_mem:
            print()
            print_cuda_memory()
            print_cuda_mem = False

        # Time
        end = time.time()
        if verbose:
            print(f'Time elapsed in epoch: {(end - start):.2f} secs.')

        loss_history.append(np.array(loss_total/loss_count))

        PATH = "checkpoints/model_epoch" + str(curr_epoch) + ".pt"

        if checkpoint_every:
            if curr_epoch % checkpoint_every == 0:
                torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'loss_history_val': loss_history_val
                }, PATH)

        if curr_epoch == last_epoch:
            return loss_history, loss_history_val

        curr_epoch += 1
