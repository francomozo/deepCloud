# USAGE:
#   Depracated functions that have usage.
#   Should be used as little as possible.
#

# First version of the SatelliteImagesDataset with sliding window. Takes too long.
# Loades images in each iteration.
class SatelliteImagesDatasetSW_v1(Dataset):
    """ [WARNING]: This function is depracated. Too slow. Use SatelliteImagesDatasetSW instead.
        South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        window (int): Size of the moving window to load the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        
        
    Returns:
        [dict]: {'images': images, 'time_stamps': time_stamps}
    """

    dia_ref = datetime.datetime(2019,12,31)
    
    
    
    def __init__(self, root_dir, window=1, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
        self.window = window
        
    
    def __len__(self):
        return len(self.images_list) - self.window + 1
    
    def __getitem__(self, idx):
        try:
            img_names = [os.path.join(self.root_dir, self.images_list[idx])
                         for idx in np.arange(idx, self.window + idx, 1)]

            images = np.array([np.load(img_name) for img_name in img_names])
            
            if self.transform:
                images = np.array([self.transform(image) for image in images])
                # images = np.array([self.transform(Image.fromarray(image)) for image in images])
            
            img_names = [re.sub("[^0-9]", "", self.images_list[idx]) 
                         for idx in np.arange(idx, self.window + idx, 1)]
           
            time_stamps = [self.dia_ref + datetime.timedelta(days=int(img_name[4:7]), 
                                                             hours=int(img_name[7:9]), 
                                                             minutes=int(img_name[9:11]), 
                                                             seconds=int(img_name[11:]))
                            for img_name in img_names]

            samples = {'images': images,
                      'time_stamps': [utils.datetime2str(ts) for ts in time_stamps]}
            
            return samples        
        except IndexError:
            print('End of sliding window')
            

class SatelliteImagesDataset(Dataset):
    """ [WARNING]: This function is depracated.
        First version of a simple dataset class for one image at a time.
        South America Satellite Images Dataset

    Args:
        root_dir (string): Directory with all images from day n.
        transform (callable, optional): Optional transform to be applied on a sample.
        
    Returns:
        [dict]: {'image': image, 'time_stamp': time_stamp}
    """

    dia_ref = datetime.datetime(2019,12,31)
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_list = np.sort(os.listdir(self.root_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 
                                self.images_list[idx])
        image = np.load(img_name)
        if self.transform:
            image = self.transform(image)
        
        img_name = re.sub("[^0-9]", "", self.images_list[idx])
        time_stamp = self.dia_ref + datetime.timedelta(days=int(img_name[4:7]), hours =int(img_name[7:9]), 
                                                   minutes = int(img_name[9:11]), seconds = int(img_name[11:]) )
        
        sample = {'image': image,
                  'time_stamp': utils.datetime2str(time_stamp)}
        
        return sample
    
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
        
    if writer:
        in_frames, _ = next(iter(train_loader))
        in_frames = in_frames.to(device=device)
        writer.add_graph(model, input_to_model=in_frames, verbose=False)

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
