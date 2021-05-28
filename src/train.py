# USAGE:
#   Training loops and checkpoint saving
#
import time

import torch
import optuna
import numpy as np

from src.lib.utils import print_cuda_memory


def train_model(model,
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
    """

    loss_history_val = []
    val_step = 0
    if not train_for:
        return loss_history
    last_epoch = train_for + curr_epoch

    curr_epoch += 1
    # No entiendo xq es necesario este if:  ---borrar---
    # if curr_epoch > 1:
    #     curr_epoch += 1  # If curr_epoch not zero, the argument passed to
        # curr_epoch is the last epoch the
        # model was trained in the loop before

    if print_cuda_mem:
        print_cuda_memory()

    while True:
        if verbose:
            print("EPOCH:", curr_epoch,  end=' ')
        start = time.time()
        image_number = 0
        loss_average = 0
        loss_count = 0
        for (curr_seq, idxs, data, targets) in loader:

            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)
            loss_average += loss.item()
            loss_count += 1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Validation
            image_number += data.shape[0]
            if loader_val is not None and image_number >= 200:
                image_number = 0
                model.eval()
                loss_val_total = 0
                loss_val_count = 0
                for (curr_seq, idxs, data_val, targets_val) in loader_val:
                    data_val = data_val.to(device=device)
                    targets_val = targets_val.to(device=device)

                    scores_val = model(data_val)
                    loss_val = criterion(scores_val, targets_val)
                    loss_val_total += loss_val.item()
                    loss_val_count += 1

                loss_val_average = loss_val_total/loss_val_count
                loss_history_val.append(np.array(loss_val_average))
                model.train()
                if trial is not None:
                    trial.report(loss_val_average, val_step)
                    val_step += 1
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        if print_cuda_mem:
            print()
            print_cuda_memory()
            print_cuda_mem = False

        # Time
        end = time.time()
        if verbose:
            print(f'Time elapsed: {(end - start):.2f} secs.')

        loss_history.append(np.array(loss_average/loss_count))

        PATH = "checkpoints/model_epoch" + str(curr_epoch) + ".pt"

        if checkpoint_every:
            if curr_epoch % checkpoint_every == 0:
                torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                }, PATH)

        if curr_epoch == last_epoch:
            return loss_history, loss_history_val

        curr_epoch += 1
