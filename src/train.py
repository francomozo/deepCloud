# USAGE:
#   Training loops and checkpoint saving
#
import time

import torch

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
                print_cuda_mem=False):
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
        for (curr_seq, idxs, data, targets) in loader:

            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        if print_cuda_mem:
            print()
            print_cuda_memory()
            print_cuda_mem = False

        # Time
        end = time.time()
        if verbose:
            print(f'Time elapsed: {(end - start):.2f} secs.')

        loss_history.append(loss.clone().detach().cpu().numpy())

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
            return loss_history

        curr_epoch += 1
