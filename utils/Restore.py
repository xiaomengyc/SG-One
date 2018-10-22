import os
import torch

__all__ = ['restore']

def restore(snapshot_dir, model):
    restore_dir = snapshot_dir
    filelist = os.listdir(restore_dir)
    filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('.pth.tar')]
    if len(filelist) > 0:
        filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
        snapshot = os.path.join(restore_dir, filelist[0])
    else:
        snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(snapshot))
        except KeyError:
            raise Exception("Loading pre-trained values failed.")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(snapshot))


