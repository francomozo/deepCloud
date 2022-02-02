# USAGE:
#   Training loops and checkpoint saving
#
import datetime
import time
import os
import copy

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import CenterCrop

from src.lib.utils import print_cuda_memory
from src.lib.utils_irradianceNet import convert_to_full_res, interpolate_borders

from piqa import SSIM


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
                eval_every=100,
                writer=None,
                scheduler=None,
                model_name=None):


    if model_name is None:
        model_name = 'model'
    
    TRAIN_LOSS_GLOBAL = [] # perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = [] # perists through epochs, stores the mean of each epoch

    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] # stores values inside the current epoch
        VAL_LOSS_EPOCH = [] # stores values inside the current epoch
        

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            start_batch = time.time()

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

            end_batch = time.time()
            TIME.append(end_batch - start_batch)

            TRAIN_LOSS_EPOCH.append(loss.detach().item())

            if (batch_idx > 0 and batch_idx % eval_every == 0) or (batch_idx == len(train_loader) - 1):
                model.eval()
                VAL_LOSS_LOCAL = [] # stores values for this validation run
                start_val = time.time()
                with torch.no_grad():
                    for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

                        in_frames = in_frames.to(device=device)
                        out_frames = out_frames.to(device=device)

                        frames_pred = model(in_frames)
                        val_loss = criterion(frames_pred, out_frames)

                        VAL_LOSS_LOCAL.append(val_loss.detach().item())

                        if val_batch_idx == num_val_samples:
                            if (batch_idx == len(train_loader)-1) and writer:
                                # enter if last batch of the epoch and there is a writer
                                writer.add_images('groundtruth_batch', out_frames, epoch)
                                writer.add_images('predictions_batch', frames_pred, epoch)
                            break

                end_val = time.time()
                val_time = end_val - start_val
                CURRENT_VAL_ACC = sum(VAL_LOSS_LOCAL)/len(VAL_LOSS_LOCAL)
                VAL_LOSS_EPOCH.append(CURRENT_VAL_ACC)
                
                CURRENT_TRAIN_ACC = sum(TRAIN_LOSS_EPOCH[batch_idx-eval_every:])/len(TRAIN_LOSS_EPOCH[batch_idx-eval_every:])

                if verbose:
                    # print statistics
                    print(f'Epoch({epoch + 1}/{epochs}) | Batch({batch_idx:04d}/{len(train_loader)}) | ', end='')
                    print(f'Train_loss({(CURRENT_TRAIN_ACC):06.4f}) | Val_loss({CURRENT_VAL_ACC:.4f}) | ', end='')
                    print(f'Time_per_batch({sum(TIME)/len(TIME):.2f}s) | Val_time({val_time:.2f}s)') 
                    TIME = []
                    
                if writer: 
                    # add values to tensorboard 
                    writer.add_scalar("Loss in train GLOBAL",CURRENT_TRAIN_ACC , batch_idx + epoch*(len(train_loader)))
                    writer.add_scalar("Loss in val GLOBAL" , CURRENT_VAL_ACC,  batch_idx + epoch*(len(train_loader)))
        
        # epoch end      
        end_epoch = time.time()
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))
        
        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        if writer: 
            # add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN",TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN" , VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)
          
        if verbose:
            print(f'Time elapsed in current epoch: {(end_epoch - start_epoch):.2f} secs.')

        if CURRENT_VAL_ACC < BEST_VAL_ACC:
            BEST_VAL_ACC = CURRENT_VAL_ACC
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_per_batch': TRAIN_LOSS_EPOCH,
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1]
            }


        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            PATH = 'checkpoints/'
            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME =  model_name + '_epoch' + str(epoch + 1) + '_' + str(ts) + '.pt'

            torch.save(model_dict, PATH + NAME)
            
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
          nn.init.constant_(model.bias.data, 0)

# how to apply

# 1) load model , ex: model = UNet(...)
# 2) model.apply(weights_init)

def train_model_2(model,
                criterion,
                optimizer,
                device,
                train_loader,
                epochs,
                val_loader,
                checkpoint_every=None,
                verbose=True,
                writer=None,
                scheduler=None,
                model_name=None,
                save_images=True):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    if model_name is None:
        model_name = 'model' 

    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        img_size = in_frames.size(2)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

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

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            mse_val_loss = 0
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = mae_loss(frames_pred, out_frames)
                mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
          
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    if img_size < 1000:
                        writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                        writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    else:
                        writer.add_images('groundtruth_batch', out_frames[0], epoch)
                        writer.add_images('predictions_batch', frames_pred[0], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("VALIDATION MSE LOSS",  mse_val_loss/len(val_loader), epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_SSIM(
                        model,
                        train_criterion,
                        val_criterion,
                        optimizer,
                        device,
                        train_loader,
                        epochs,
                        val_loader,
                        checkpoint_every=None,
                        verbose=True,
                        writer=None,
                        scheduler=None,
                        model_name='model',
                        save_images=True):
    
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = 1 - train_criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = val_criterion(frames_pred, out_frames)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_cyclicLR(   model,
                            criterion,
                            optimizer,
                            device,
                            train_loader,
                            epochs,
                            val_loader,
                            checkpoint_every=None,
                            verbose=True,
                            writer=None,
                            scheduler=None,
                            model_name='model',
                            save_images=True):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

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

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
            if scheduler:
                scheduler.step()
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = criterion(frames_pred, out_frames)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_diff(model,
                     criterion,
                     optimizer,
                     device,
                     train_loader,
                     epochs,
                     val_loader,
                     checkpoint_every=None,
                     verbose=True,
                     writer=None,
                     scheduler=None,
                     model_name=None,
                     save_images=True):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    
    if model_name is None:
        model_name = 'model' 

    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            diff_pred = model(in_frames)

            diff = torch.subtract(out_frames[:,0], in_frames[:,2]).unsqueeze(1)
            loss = criterion(diff_pred, diff)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                diff_pred = model(in_frames)
                
                frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1)
                
                val_loss = criterion(out_frames, frames_pred)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_SSIMandMAE(
                            model,
                            SSIM_criterion,
                            MAE_criterion,
                            optimizer,
                            device,
                            train_loader,
                            epochs,
                            val_loader,
                            checkpoint_every=None,
                            verbose=True,
                            writer=None,
                            scheduler=None,
                            model_name='model',
                            save_images=True):
    
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = 1 - SSIM_criterion(frames_pred, out_frames) + MAE_criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = MAE_criterion(frames_pred, out_frames)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_ssim_diff(  model,
                            train_criterion,
                            val_criterion,
                            optimizer,
                            device,
                            train_loader,
                            epochs,
                            val_loader,
                            checkpoint_every=None,
                            verbose=True,
                            writer=None,
                            scheduler=None,
                            model_name=None,
                            save_images=True):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    
    if model_name is None:
        model_name = 'model' 

    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            diff_pred = model(in_frames)

            diff = torch.subtract(out_frames[:,0], in_frames[:,2]).unsqueeze(1)
            loss = 1- train_criterion(diff_pred, diff)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                diff_pred = model(in_frames)
                
                frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1)
                
                val_loss = val_criterion(out_frames, frames_pred)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL


def train_model_MSE(
                        model,
                        train_criterion,
                        val_criterion,
                        optimizer,
                        device,
                        train_loader,
                        epochs,
                        val_loader,
                        checkpoint_every=None,
                        verbose=True,
                        writer=None,
                        scheduler=None,
                        model_name='model',
                        save_images=True):
    
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        criterion (torch.criterion): [description]
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.

    Returns:
        TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL: Lists containing the mean error of each epoch
    """    
    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_LOSS_GLOBAL = []
    
    TIME = []

    BEST_VAL_ACC = 1e5
    
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = [] #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            loss = train_criterion(frames_pred, out_frames)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH.append(loss.detach().item())
            
        TRAIN_LOSS_GLOBAL.append(sum(TRAIN_LOSS_EPOCH)/len(TRAIN_LOSS_EPOCH))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            VAL_LOSS_EPOCH = []
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                val_loss = val_criterion(frames_pred, out_frames)

                VAL_LOSS_EPOCH.append(val_loss.detach().item())
                
                if writer and (val_batch_idx == 0) and save_images and epoch>35:
                    writer.add_images('groundtruth_batch', out_frames[:10], epoch)
                    writer.add_images('predictions_batch', frames_pred[:10], epoch)
                    
        VAL_LOSS_GLOBAL.append(sum(VAL_LOSS_EPOCH)/len(VAL_LOSS_EPOCH))

        if scheduler:
            scheduler.step(VAL_LOSS_GLOBAL[-1])
        
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val_loss({VAL_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION LOSS, EPOCH MEAN", VAL_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)            

        if VAL_LOSS_GLOBAL[-1] < BEST_VAL_ACC:
            if verbose:
                print('New Best Model')
            BEST_VAL_ACC = VAL_LOSS_GLOBAL[-1]
            model_dict = {
                'epoch': epoch + 1,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL[-1],
                'val_loss_epoch_mean': VAL_LOSS_GLOBAL[-1]
            }
            model_not_saved = True

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if model_not_saved:
                if verbose:
                    print('Saving Checkpoint')
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

                torch.save(model_dict, PATH + NAME)
                model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose:
            print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, PATH + NAME)
    
    return TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL

def train_model_full(
                    model,
                    train_loss,
                    optimizer,
                    device,
                    train_loader,
                    val_loader,
                    epochs,
                    checkpoint_every=None,
                    verbose=True,
                    writer=None,
                    scheduler=None,
                    loss_for_scheduler='mae',
                    model_name=None,
                    save_images=True,
                    predict_diff=False,
                    retrain=False,
                    trained_model_dict=None,
                    testing_loop=False):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        train_loss (str): Train criterion to use ('mae','mse','ssim')
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.
        loss_for_scheduler (string): choose validation error to use for scheduler steps
        model_name (string): Prefix Name for the checkpoint model to be saved
        save_images (bool): If true images are saved on the tensorboard
        predict_diff (bool): If True the model predicts the difference between las input image and output
        
    Returns:
        TRAIN_LOSS_GLOBAL: Mean train loss in each epoch 
        VAL_MAE_LOSS_GLOBAL: Lists containing the mean MAE error of each epoch in validation
        VAL_MSE_LOSS_GLOBAL: Lists containing the mean MSE error of each epoch in validation
        VAL_SSIM_LOSS_GLOBAL: Lists containing the mean SSIM error of each epoch in validation
    """    
    if  retrain and not trained_model_dict:
        raise ValueError('To retrain the model dict is needed')

    if  predict_diff and (train_loss in ['ssim', 'SSIM']):
        raise ValueError('Cannot use ssim as train function and predict diff. (Yet)')
    
    PREVIOUS_CHECKPOINT_NAME = None
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    ssim_loss = SSIM(n_channels=1).cuda()
    
    if train_loss in ['mae', 'MAE']:
        train_criterion = mae_loss
    if train_loss in ['mse', 'MSE']:
        train_criterion = mse_loss
    if train_loss in ['ssim', 'SSIM']:
        train_criterion = ssim_loss
    if train_loss in ['mae_ssim', 'MAE_SSIM']:
        train_criterion_mae = mae_loss
        train_criterion_ssim = ssim_loss
    if train_loss in ['forecaster_loss', 'FORECASTER_LOSS']:
        train_criterion = FORECASTER_LOSS()
    if model_name is None:
        model_name = 'model' 
    
    TIME = []

    if retrain:
        TRAIN_LOSS_GLOBAL = trained_model_dict['train_loss_epoch_mean']
        VAL_MAE_LOSS_GLOBAL = trained_model_dict['val_mae_loss']
        VAL_MSE_LOSS_GLOBAL = trained_model_dict['val_mse_loss']
        VAL_SSIM_LOSS_GLOBAL = trained_model_dict['val_ssim_loss']
        
        if trained_model_dict['validation_loss'] in ['mae', 'MAE']:
            BEST_VAL_ACC = VAL_MAE_LOSS_GLOBAL[-1]
        if trained_model_dict['validation_loss'] in ['mse', 'MSE']:
            BEST_VAL_ACC = VAL_MSE_LOSS_GLOBAL[-1]
        if trained_model_dict['validation_loss'] in ['ssim', 'SSIM']:
            BEST_VAL_ACC = VAL_SSIM_LOSS_GLOBAL[-1]
        
        first_epoch = trained_model_dict['epoch']
        print(f'Start from pre trained model, epoch: {first_epoch}, last train loss: {TRAIN_LOSS_GLOBAL[-1]}, best val loss: {BEST_VAL_ACC}')
        
    else:
        TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
        VAL_MAE_LOSS_GLOBAL = []
        VAL_MSE_LOSS_GLOBAL = []
        VAL_SSIM_LOSS_GLOBAL = []
        
        BEST_VAL_ACC = 1e5
        
        first_epoch = 0

    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        # writer.add_graph(model, input_to_model=in_frames, verbose=False)
        img_size = in_frames.size(2)
        
    for epoch in range(first_epoch, epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = 0 #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            
            if testing_loop and batch_idx == 10:
                break
            
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            if train_loss in ['mae', 'MAE', 'mse', 'MSE', 'forecaster_loss', 'FORECASTER_LOSS']:
                if predict_diff:
                    diff = torch.subtract(out_frames[:,0], in_frames[:,2]).unsqueeze(1)
                    loss = train_criterion(frames_pred, diff)
                else: 
                    loss = train_criterion(frames_pred, out_frames)
                    
            if train_loss in ['ssim', 'SSIM']:
                loss = 1 - train_criterion(frames_pred, out_frames)
            if train_loss in ['mae_ssim', 'MAE_SSIM']:
                loss = 1 - train_criterion_ssim(frames_pred, out_frames) + train_criterion_mae(frames_pred, out_frames)
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH += loss.detach().item()
            
        TRAIN_LOSS_GLOBAL.append(TRAIN_LOSS_EPOCH/len(train_loader))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            mse_val_loss = 0
            mae_val_loss = 0
            ssim_val_loss = 0
            
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                
                if testing_loop and val_batch_idx == 10:
                    break
                
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                
                if predict_diff:
                    frames_pred = torch.add(frames_pred[:,0], in_frames[:,2]).unsqueeze(1)
                    frames_pred = torch.clamp(frames_pred, min=0, max=1)
                    mae_val_loss += mae_loss(frames_pred, out_frames).detach().item()
                    mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()
                    ssim_val_loss += ssim_loss(frames_pred, out_frames).detach().item()
                    
                if not predict_diff:
                    mae_val_loss += mae_loss(frames_pred, out_frames).detach().item()
                    mse_val_loss += mse_loss(frames_pred, out_frames).detach().item()
                    frames_pred = torch.clamp(frames_pred, min=0, max=1)
                    ssim_val_loss += ssim_loss(frames_pred, out_frames).detach().item()

        VAL_MAE_LOSS_GLOBAL.append(mae_val_loss/len(val_loader))
        VAL_MSE_LOSS_GLOBAL.append(mse_val_loss/len(val_loader))
        VAL_SSIM_LOSS_GLOBAL.append(ssim_val_loss/len(val_loader))
        
        if scheduler:
            if loss_for_scheduler in ['mae', 'MAE']:
                scheduler.step(VAL_MAE_LOSS_GLOBAL[-1])
            if loss_for_scheduler in ['mse', 'MSE']:
                scheduler.step(VAL_MSE_LOSS_GLOBAL[-1])    
            if loss_for_scheduler in ['ssim', 'SSIM']:
                scheduler.step(1 - VAL_SSIM_LOSS_GLOBAL[-1])
                     
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val MAE({VAL_MAE_LOSS_GLOBAL[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL[-1]:.4f}) | Val SSIM({VAL_SSIM_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION MAE", VAL_MAE_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("VALIDATION MSE",  VAL_MSE_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION SSIM",  VAL_SSIM_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        
        if loss_for_scheduler in ['mae', 'MAE']:
            actual_loss = VAL_MAE_LOSS_GLOBAL[-1]
        if loss_for_scheduler in ['mse', 'MSE']:
            actual_loss = VAL_MSE_LOSS_GLOBAL[-1]
        if loss_for_scheduler in ['ssim', 'SSIM']:
            actual_loss = 1 - VAL_SSIM_LOSS_GLOBAL[-1]
                                
        if actual_loss < BEST_VAL_ACC:
            BEST_VAL_ACC = actual_loss
            
            if verbose: print('New Best Model')    
            best_model_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation_loss': loss_for_scheduler,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL,
                'val_mae_loss': VAL_MAE_LOSS_GLOBAL,
                'val_mse_loss': VAL_MSE_LOSS_GLOBAL,
                'val_ssim_loss': VAL_SSIM_LOSS_GLOBAL
            }
            best_model_not_saved = True
            best_model_epoch = epoch + 1

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if verbose: print('Saving Checkpoint')
            
            actual_model_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation_loss': loss_for_scheduler,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL,
                'val_mae_loss': VAL_MAE_LOSS_GLOBAL,
                'val_mse_loss': VAL_MSE_LOSS_GLOBAL,
                'val_ssim_loss': VAL_SSIM_LOSS_GLOBAL
            }
            
            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'
            torch.save(actual_model_dict, os.path.join(PATH, NAME))
            # delete previous checkpoint saved
            if PREVIOUS_CHECKPOINT_NAME:
                try:
                    os.remove(os.path.join(PATH, PREVIOUS_CHECKPOINT_NAME))
                except OSError as e:  ## if failed, report it back to the user ##
                    print ("Error: Couldnt delete checkpoint")
            PREVIOUS_CHECKPOINT_NAME = NAME
            
            if best_model_not_saved:
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  'BEST_' + model_name + '_' + str(best_model_epoch) + '_' + str(ts) + '.pt'
                torch.save(best_model_dict, os.path.join(PATH, NAME))
                best_model_not_saved = False
                
    # if training finished and best model not saved
    if model_not_saved:
        if verbose: print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'

        torch.save(model_dict, os.path.join(PATH,NAME))
    
    return TRAIN_LOSS_GLOBAL, VAL_MAE_LOSS_GLOBAL, VAL_MSE_LOSS_GLOBAL, VAL_SSIM_LOSS_GLOBAL


def train_irradianceNet(
                    model,
                    train_loss,
                    optimizer,
                    device,
                    train_loader,
                    val_loader,
                    epochs,
                    img_size=512,
                    patch_size=128,
                    checkpoint_every=None,
                    verbose=True,
                    writer=None,
                    scheduler=None,
                    loss_for_scheduler='mae',
                    model_name=None,
                    save_images=True,
                    direct=False,
                    train_w_last=False,
                    geo_data=False,
                    retrain=False,
                    trained_model_dict=None,
                    testing_loop=False):
    """ This train function evaluates on all the validation dataset one time per epoch

    Args:
        model (torch.model): [description]
        train_loss (str): Train criterion to use ('mae','mse','ssim')
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader ([type]): [description]
        checkpoint_every (int, optional): [description]. Defaults to None.
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        scheduler ([type], optional): [description]. Defaults to None.
        loss_for_scheduler (string): choose validation error to use for scheduler steps
        model_name (string): Prefix Name for the checkpoint model to be saved
        save_images (bool): If true images are saved on the tensorboard
        
    Returns:
        TRAIN_LOSS_GLOBAL: Mean train loss in each epoch 
        VAL_MAE_LOSS_GLOBAL: Lists containing the mean MAE error of each epoch in validation
        VAL_MSE_LOSS_GLOBAL: Lists containing the mean MSE error of each epoch in validation
        VAL_SSIM_LOSS_GLOBAL: Lists containing the mean SSIM error of each epoch in validation
    """
    
    if  retrain and not trained_model_dict:
        raise ValueError('To retrain the model dict is needed')
    if train_w_last and direct:
        raise ValueError('To train with only last predict horizon the model shouldnt be direct')
        
    PREVIOUS_CHECKPOINT_NAME = None

    dim = img_size // patch_size
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    ssim_loss = SSIM(n_channels=1).cuda()
    
    if train_loss in ['mae', 'MAE']:
        train_criterion = mae_loss
    if train_loss in ['mse', 'MSE']:
        train_criterion = mse_loss
    if train_loss in ['ssim', 'SSIM']:
        train_criterion = ssim_loss
    if train_loss in ['mae_ssim', 'MAE_SSIM']:
        train_criterion_mae = mae_loss
        train_criterion_ssim = ssim_loss
    if train_loss in ['forecaster_loss', 'FORECASTER_LOSS']:
        train_criterion = FORECASTER_LOSS()
    if model_name is None:
        model_name = 'model' 
    
    TIME = []

    if retrain:
        TRAIN_LOSS_GLOBAL = trained_model_dict['train_loss_epoch_mean']
        VAL_MAE_LOSS_GLOBAL = trained_model_dict['val_mae_loss']
        VAL_MSE_LOSS_GLOBAL = trained_model_dict['val_mse_loss']
        VAL_SSIM_LOSS_GLOBAL = trained_model_dict['val_ssim_loss']
        
        if trained_model_dict['validation_loss'] in ['mae', 'MAE']:
            BEST_VAL_ACC = VAL_MAE_LOSS_GLOBAL[-1]
        if trained_model_dict['validation_loss'] in ['mse', 'MSE']:
            BEST_VAL_ACC = VAL_MSE_LOSS_GLOBAL[-1]
        if trained_model_dict['validation_loss'] in ['ssim', 'SSIM']:
            BEST_VAL_ACC = VAL_SSIM_LOSS_GLOBAL[-1]
        
        first_epoch = trained_model_dict['epoch']
        print(f'Start from pre trained model, epoch: {first_epoch}, last train loss: {TRAIN_LOSS_GLOBAL[-1]}, best val loss: {BEST_VAL_ACC}')
        
    else:
        TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
        VAL_MAE_LOSS_GLOBAL = []
        VAL_MSE_LOSS_GLOBAL = []
        VAL_SSIM_LOSS_GLOBAL = []
        
        BEST_VAL_ACC = 1e5
        
        first_epoch = 0
        
    for epoch in range(first_epoch, epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = 0 #stores values inside the current epoch
        
        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            model.train()
            
            if testing_loop and batch_idx==1:
                break
            
            if not geo_data:
                in_frames = torch.unsqueeze(in_frames, dim=2)
                
            in_frames = in_frames.to(device=device)
            if not train_w_last:
                out_frames = torch.unsqueeze(out_frames, dim=2)
                
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            if not train_w_last:
                if train_loss in ['mae', 'MAE', 'mse', 'MSE', 'forecaster_loss', 'FORECASTER_LOSS']:
                    loss = train_criterion(frames_pred, out_frames)
                        
                if train_loss in ['ssim', 'SSIM']:
                    loss = 1 - train_criterion(frames_pred, out_frames)
                if train_loss in ['mae_ssim', 'MAE_SSIM']:
                    loss = 1 - train_criterion_ssim(frames_pred, out_frames) + train_criterion_mae(frames_pred, out_frames)
            
            else:
                if train_loss in ['mae', 'MAE', 'mse', 'MSE', 'forecaster_loss', 'FORECASTER_LOSS']:
                    loss = train_criterion(frames_pred[:, -1, :,:,:], out_frames)
                if train_loss in ['ssim', 'SSIM']:
                    loss = 1 - train_criterion(frames_pred[:, -1, :,:,:], out_frames)
                if train_loss in ['mae_ssim', 'MAE_SSIM']:
                    loss = 1 - train_criterion_ssim(frames_pred[:, -1, :,:,:], out_frames) + train_criterion_mae(frames_pred[:, -1, :,:,:], out_frames)
            
                
                # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH += loss.detach().item()
            
        TRAIN_LOSS_GLOBAL.append(TRAIN_LOSS_EPOCH/len(train_loader))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            mse_val_loss = 0
            mae_val_loss = 0
            ssim_val_loss = 0
            
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):
                
                if testing_loop and val_batch_idx==1:
                    break
                
                if not geo_data:
                    in_frames = torch.unsqueeze(in_frames, dim=2)
                in_frames = in_frames.to(device=device)
                
                if not train_w_last:
                    out_frames = torch.unsqueeze(out_frames, dim=2)
                out_frames = out_frames.to(device=device)
                
                mae_val_loss_Q = 0
                mse_val_loss_Q = 0
                ssim_val_loss_Q = 0
                
                for i in range(dim):
                    for j in range(dim):
                        n = i * patch_size
                        m = j * patch_size
                        
                        frames_pred_Q = model(in_frames[:,:,:, n:n+patch_size, m:m+patch_size])
                        if not train_w_last:
                            mae_val_loss_Q += mae_loss(frames_pred_Q, out_frames[:,:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                            mse_val_loss_Q += mse_loss(frames_pred_Q, out_frames[:,:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                            if direct:
                                frames_pred_Q = torch.clamp(torch.squeeze(frames_pred_Q, dim=1), min=0, max=1)
                                
                                ssim_val_loss_Q += ssim_loss(frames_pred_Q,
                                                            torch.squeeze(out_frames[:,:,:, n:n+patch_size, m:m+patch_size],
                                                                          dim=1)
                                                            ).detach().item()
                            else:    
                                ssim_val_loss_Q = 0
                        else:
                            mae_val_loss_Q += mae_loss(frames_pred_Q[:,-1,:,:,:],
                                                       out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                            mse_val_loss_Q += mse_loss(frames_pred_Q[:,-1,:,:,:],
                                                       out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()

                            frames_pred_Q = torch.clamp(frames_pred_Q[:,-1,:,:,:], min=0, max=1)
                            
                            ssim_val_loss_Q += ssim_loss(frames_pred_Q,
                                                        out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                        
                mae_val_loss += (mae_val_loss_Q / (dim*dim))
                mse_val_loss += (mse_val_loss_Q / (dim**2))
                ssim_val_loss += (ssim_val_loss_Q / (dim**2))
                    
        VAL_MAE_LOSS_GLOBAL.append(mae_val_loss/len(val_loader))
        VAL_MSE_LOSS_GLOBAL.append(mse_val_loss/len(val_loader))
        VAL_SSIM_LOSS_GLOBAL.append(ssim_val_loss/len(val_loader))
        
        if scheduler:
            if loss_for_scheduler in ['mae', 'MAE']:
                scheduler.step(VAL_MAE_LOSS_GLOBAL[-1])
            if loss_for_scheduler in ['mse', 'MSE']:
                scheduler.step(VAL_MSE_LOSS_GLOBAL[-1])    
            if loss_for_scheduler in ['ssim', 'SSIM']:
                scheduler.step(1 - VAL_SSIM_LOSS_GLOBAL[-1])
                     
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f}) | Val MAE({VAL_MAE_LOSS_GLOBAL[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL[-1]:.4f}) | Val SSIM({VAL_SSIM_LOSS_GLOBAL[-1]:.4f}) | ', end='')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION MAE", VAL_MAE_LOSS_GLOBAL[-1] , epoch)
            writer.add_scalar("VALIDATION MSE",  VAL_MSE_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION SSIM",  VAL_SSIM_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("Learning rate", optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        
        if loss_for_scheduler in ['mae', 'MAE']:
            actual_loss = VAL_MAE_LOSS_GLOBAL[-1]
        if loss_for_scheduler in ['mse', 'MSE']:
            actual_loss = VAL_MSE_LOSS_GLOBAL[-1]
        if loss_for_scheduler in ['ssim', 'SSIM']:
            actual_loss = 1 - VAL_SSIM_LOSS_GLOBAL[-1]
                                
        if actual_loss < BEST_VAL_ACC:
            BEST_VAL_ACC = actual_loss
            
            if verbose: print('New Best Model')    
            best_model_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation_loss': loss_for_scheduler,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL,
                'val_mae_loss': VAL_MAE_LOSS_GLOBAL,
                'val_mse_loss': VAL_MSE_LOSS_GLOBAL,
                'val_ssim_loss': VAL_SSIM_LOSS_GLOBAL
            }
            best_model_not_saved = True
            best_model_epoch = epoch + 1

        if checkpoint_every is not None and (epoch + 1) % checkpoint_every == 0:
            if verbose: print('Saving Checkpoint')
            
            actual_model_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'validation_loss': loss_for_scheduler,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'train_loss_epoch_mean': TRAIN_LOSS_GLOBAL,
                'val_mae_loss': VAL_MAE_LOSS_GLOBAL,
                'val_mse_loss': VAL_MSE_LOSS_GLOBAL,
                'val_ssim_loss': VAL_SSIM_LOSS_GLOBAL
            }
            
            ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
            NAME =  model_name + '_' + str(epoch + 1) + '_' + str(ts) + '.pt'
            torch.save(actual_model_dict, os.path.join(PATH, NAME))
            # delete previous checkpoint saved
            if PREVIOUS_CHECKPOINT_NAME:
                try:
                    os.remove(os.path.join(PATH, PREVIOUS_CHECKPOINT_NAME))
                except OSError as e:  ## if failed, report it back to the user ##
                    print ("Error: Couldnt delete checkpoint")
            PREVIOUS_CHECKPOINT_NAME = NAME
            
            if best_model_not_saved:
                PATH = 'checkpoints/'
                ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
                NAME =  'BEST_' + model_name + '_' + str(best_model_epoch) + '_' + str(ts) + '.pt'
                torch.save(best_model_dict, os.path.join(PATH, NAME))
                best_model_not_saved = False
                
    # if training finished and best model not saved
    if best_model_not_saved:
        if verbose: print('Saving Checkpoint')
        PATH = 'checkpoints/'
        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
        NAME =  model_name + '_' + str(best_model_epoch) + '_' + str(ts) + '.pt'
        torch.save(best_model_dict, os.path.join(PATH, NAME))
    
    return TRAIN_LOSS_GLOBAL, VAL_MAE_LOSS_GLOBAL, VAL_MSE_LOSS_GLOBAL, VAL_SSIM_LOSS_GLOBAL


def train_model_double_val(
                    model,
                    train_loss,
                    optimizer,
                    device,
                    train_loader,
                    val_loader_w_csv,
                    val_loader_wo_csv,
                    epochs,
                    verbose=True,
                    writer=None,
                    testing_loop=False):
    """ This train function evaluates on two validation datasets per epoch. 

    Args:
        model (torch.model): [description]
        train_loss (str): Train criterion to use ('mae','mse','ssim')
        optimizer (torch.optim): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        epochs (int): [description]
        val_loader_w_csv ([type]): [description]
        val_loader_wo_csv ([type]): [description]
        verbose (bool, optional): Print trainning status. Defaults to True.
        writer (tensorboard.writer, optional): Logs loss values to tensorboard. Defaults to None.
        
    Returns:
        TRAIN_LOSS_GLOBAL: Mean train loss in each epoch 
        VAL_MAE_LOSS_GLOBAL: Lists containing the mean MAE error of each epoch in validation
        VAL_MSE_LOSS_GLOBAL: Lists containing the mean MSE error of each epoch in validation
        VAL_SSIM_LOSS_GLOBAL: Lists containing the mean SSIM error of each epoch in validation
    """
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    ssim_loss = SSIM(n_channels=1).cuda()
    
    if train_loss in ['mae', 'MAE']:
        train_criterion = mae_loss
    if train_loss in ['mse', 'MSE']:
        train_criterion = mse_loss
    if train_loss in ['ssim', 'SSIM']:
        train_criterion = ssim_loss
    if train_loss in ['mae_ssim', 'MAE_SSIM']:
        train_criterion_mae = mae_loss
        train_criterion_ssim = ssim_loss
    if train_loss in ['forecaster_loss', 'FORECASTER_LOSS']:
        train_criterion = FORECASTER_LOSS()
    
    TIME = []

    TRAIN_LOSS_GLOBAL = [] #perists through epochs, stores the mean of each epoch
    VAL_MAE_LOSS_GLOBAL_W_CSV = []
    VAL_MSE_LOSS_GLOBAL_W_CSV = []
    VAL_SSIM_LOSS_GLOBAL_W_CSV = []
    
    VAL_MAE_LOSS_GLOBAL_WO_CSV = []
    VAL_MSE_LOSS_GLOBAL_WO_CSV = []
    VAL_SSIM_LOSS_GLOBAL_WO_CSV = []
        
    for epoch in range(epochs):
        start_epoch = time.time()
        TRAIN_LOSS_EPOCH = 0 #stores values inside the current epoch

        for batch_idx, (in_frames, out_frames) in enumerate(train_loader):
            
            if testing_loop and batch_idx == 1:
                break
            
            model.train()

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)

            # forward
            frames_pred = model(in_frames)
            if train_loss in ['mae', 'MAE', 'mse', 'MSE', 'forecaster_loss', 'FORECASTER_LOSS']:
                loss = train_criterion(frames_pred, out_frames)
            if train_loss in ['ssim', 'SSIM']:
                loss = 1 - train_criterion(frames_pred, out_frames)
            if train_loss in ['mae_ssim', 'MAE_SSIM']:
                loss = 1 - train_criterion_ssim(frames_pred, out_frames) + train_criterion_mae(frames_pred, out_frames)
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            TRAIN_LOSS_EPOCH += loss.detach().item()
            
        TRAIN_LOSS_GLOBAL.append(TRAIN_LOSS_EPOCH / len(train_loader))
        
        #evaluation
        model.eval()

        with torch.no_grad():
            mse_val_loss_w_csv = 0
            mae_val_loss_w_csv = 0
            ssim_val_loss_w_csv = 0
            
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader_w_csv):
                
                if testing_loop and val_batch_idx == 1:
                    break
                
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)

                mae_val_loss_w_csv += mae_loss(frames_pred, out_frames).detach().item()
                mse_val_loss_w_csv += mse_loss(frames_pred, out_frames).detach().item()
                frames_pred = torch.clamp(frames_pred, min=0, max=1)
                ssim_val_loss_w_csv += ssim_loss(frames_pred, out_frames).detach().item()
            
            mse_val_loss_wo_csv = 0
            mae_val_loss_wo_csv = 0
            ssim_val_loss_wo_csv = 0
                  
            for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader_wo_csv):
                
                if testing_loop and val_batch_idx == 1:
                    break
                
                in_frames = in_frames.to(device=device)
                out_frames = out_frames.to(device=device)

                frames_pred = model(in_frames)
                
                mae_val_loss_wo_csv += mae_loss(frames_pred, out_frames).detach().item()
                mse_val_loss_wo_csv += mse_loss(frames_pred, out_frames).detach().item()
                frames_pred = torch.clamp(frames_pred, min=0, max=1)
                ssim_val_loss_wo_csv += ssim_loss(frames_pred, out_frames).detach().item()

        VAL_MAE_LOSS_GLOBAL_W_CSV.append(mae_val_loss_w_csv / len(val_loader_w_csv))
        VAL_MSE_LOSS_GLOBAL_W_CSV.append(mse_val_loss_w_csv / len(val_loader_w_csv))
        VAL_SSIM_LOSS_GLOBAL_W_CSV.append(ssim_val_loss_w_csv / len(val_loader_w_csv))
        
        VAL_MAE_LOSS_GLOBAL_WO_CSV.append(mae_val_loss_wo_csv / len(val_loader_wo_csv))
        VAL_MSE_LOSS_GLOBAL_WO_CSV.append(mse_val_loss_wo_csv / len(val_loader_wo_csv))
        VAL_SSIM_LOSS_GLOBAL_WO_CSV.append(ssim_val_loss_wo_csv / len(val_loader_wo_csv))
                     
        end_epoch = time.time()
        TIME = end_epoch - start_epoch
        
        if verbose:
            # print statistics
            print(f'Epoch({epoch + 1}/{epochs}) | ', end='')
            print(f'Train_loss({(TRAIN_LOSS_GLOBAL[-1]):06.4f})')
            print(f'WITH CSV: Val MAE({VAL_MAE_LOSS_GLOBAL_W_CSV[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL_W_CSV[-1]:.4f}) | Val SSIM({VAL_SSIM_LOSS_GLOBAL_W_CSV[-1]:.4f}) |')
            print(f'WITHOUT CSV: Val MAE({VAL_MAE_LOSS_GLOBAL_WO_CSV[-1]:.4f}) | Val MSE({VAL_MSE_LOSS_GLOBAL_WO_CSV[-1]:.4f}) | Val SSIM({VAL_SSIM_LOSS_GLOBAL_WO_CSV[-1]:.4f}) |')
            print(f'Time_Epoch({TIME:.2f}s)') # this part maybe dont print
                    
        if writer: 
            #add values to tensorboard 
            writer.add_scalar("TRAIN LOSS, EPOCH MEAN", TRAIN_LOSS_GLOBAL[-1], epoch)
            writer.add_scalar("VALIDATION MAE W CSV", VAL_MAE_LOSS_GLOBAL_W_CSV[-1] , epoch)
            writer.add_scalar("VALIDATION MSE W CSV",  VAL_MSE_LOSS_GLOBAL_W_CSV[-1], epoch)
            writer.add_scalar("VALIDATION SSIM W CSV",  VAL_SSIM_LOSS_GLOBAL_W_CSV[-1], epoch)
            writer.add_scalar("VALIDATION MAE WO CSV", VAL_MAE_LOSS_GLOBAL_WO_CSV[-1] , epoch)
            writer.add_scalar("VALIDATION MSE WO CSV",  VAL_MSE_LOSS_GLOBAL_WO_CSV[-1], epoch)
            writer.add_scalar("VALIDATION SSIM WO CSV",  VAL_SSIM_LOSS_GLOBAL_WO_CSV[-1], epoch)

    results_dict = {
        "TRAIN LOSS, EPOCH MEAN": TRAIN_LOSS_GLOBAL,
        "VALIDATION MAE W CSV": VAL_MAE_LOSS_GLOBAL_W_CSV,
        "VALIDATION MSE W CSV": VAL_MSE_LOSS_GLOBAL_W_CSV,
        "VALIDATION SSIM W CSV": VAL_SSIM_LOSS_GLOBAL_W_CSV,
        "VALIDATION MAE WO CSV": VAL_MAE_LOSS_GLOBAL_WO_CSV,
        "VALIDATION MSE WO CSV": VAL_MSE_LOSS_GLOBAL_WO_CSV,
        "VALIDATION SSIM WO CSV": VAL_SSIM_LOSS_GLOBAL_WO_CSV
        
    }
    
    return results_dict


class FORECASTER_LOSS(nn.Module):
    def __init__(self):
        super(FORECASTER_LOSS,self).__init__()

    def forward(self, output, ground):
        output = output.view(-1)
        ground = ground.view(-1)
        gap = torch.abs(output-ground)
        weight = (output+ground-gap)/2
        weight = 1-weight/100.0
        weight = torch.exp(weight)
        loss = torch.mean(weight*(output-ground)*(output-ground))
        return loss

# criterion = FORECASTER_LOSS()
# loss = criterion(output, gt)
# loss.backward()
