import os
from pathlib import Path

def get_filenames_to_sync(dcfg):
    '''Get list of all filenames that are not between the last n.
    n is the buffer size of psat.'''
    psat_path = Path(dcfg['paths']['psat'])
    list_of_filenames = sorted([img_path for img_path 
                                in os.listdir(psat_path)
                                if os.path.isfile(
                                  os.path.join(psat_path, img_path))])
    filenames_to_sync = list_of_filenames[:-dcfg['buffers']['psat']]
    filenames_to_sync.append('meta')
    return filenames_to_sync


# def sync_psat2dataset(dcfg):
#     '''Transfer and remove old images from psat to dataset.'''
#     filenames_to_sync = get_filenames_to_sync(dcfg)
#     # write list to file
#     src = Path(dcfg['paths']['psat'])
#     filenames_to_sync = [str(src / f) for f in filenames_to_sync]
#     filesfrom = 'preprocessing/sync_scripts/files_to_sync.txt' 
#     with open(filesfrom, 'w') as f:
#         f.writelines([f + '\n' for f in filenames_to_sync])  # write list of files
#     # rsync to dataser folder
#     dst = Path(dcfg['paths']['dataset'])  
#     dst.mkdir(parents=True, exist_ok=True)  # create dataset dir (if not there)
#     os.system('rsync -aru --no-R --ignore-existing '\
#         f'--files-from={filesfrom} / {dst}')  # sync to dataset
#     for f in filenames_to_sync:  # clean psat folder
#         try:
#             os.remove(f)
#         except IsADirectoryError:
#             pass


def clean_buffer(dcfg):
    '''Transfer and remove old images from psat to dataset.'''
    filenames_to_sync = get_filenames_to_sync(dcfg)
    # write list to file
    src = Path(dcfg['paths']['psat'])
    filenames_to_sync = [str(src / f) for f in filenames_to_sync]
    for f in filenames_to_sync:  # clean psat folder
        try:
            os.remove(f)
        except IsADirectoryError:
            pass

