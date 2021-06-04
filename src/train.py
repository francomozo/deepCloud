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
    num_val_samples = 10
    print_every = 100
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
            print("EPOCH:", curr_epoch)
        start = time.time()
        start_batch = time.time()
        secuence_number = 0
        loss_total = 0
        loss_count = 0
        for id, (data, targets) in enumerate(loader):
            model.train()
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
            if loader_val is not None and secuence_number >= 200:
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
                        if id_val == num_val_samples:
                            break

                    loss_val_average = loss_val_total/loss_val_count
                    loss_history_val.append(np.array(loss_val_average))
                    if trial is not None:
                        trial.report(loss_val_average, val_step)
                        val_step += 1
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

            if verbose and (id+1) % print_every == 0:
                print('Iteration',id+1 ,'/', len(loader), ',epoch_loss = %.4f' %(loss_total/loss_count) , ',Iteration time = %.2f' %(time.time()-start_batch), 's'  )
                start_batch = time.time()

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
