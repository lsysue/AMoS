import torch
import torch.nn as nn
import shutil
import os

from models.basenet import BaseFC
# from models.ggnn.ggnn import GGNN
from clustering.finch import FINCH

def create_output_dirs(cfg):
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# Load pretrained model from the specified checkpoint path
def load_pretrained_model(model, pretrain_path, is_master_proc=True):
    if pretrain_path:
        if (is_master_proc):
            print('Loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])
    return model


# Saved model checkpoint to the specified path (only do this for the master
# process if in distributed training)
def save_checkpoint(state, is_itercluster, model_name, output_path, is_master_proc=True, filename='checkpoint.pth.tar'):
    if not is_master_proc:
        return
    """Saves checkpoint to disk"""
    if is_itercluster:
        directory = f"iter_{model_name}"
    else:
        directory = f"{model_name}"
    directory = os.path.join(output_path, directory)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if (is_master_proc):
        print('\n=> checkpoint:{} saved...'.format(filename))
    # if is_best:
    #     shutil.copyfile(filename,  os.path.join(directory, 'model_best.pth.tar'))
    #     if (is_master_proc):
    #         print('=> best_model saved as:{}'.format(os.path.join(directory, 'model_best.pth.tar')))


# Load model checkpoint from the specified path
def load_checkpoint(model, checkpoint_path, classifier=False, is_master_proc=True):
    if os.path.isfile(checkpoint_path):
        if (is_master_proc):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items(): #edit
            if 'module.' in k:
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            elif classifier and ('fc' in k or 'bn_proj' in k):
                continue
            else:
                new_state_dict[k] = v
        # load params
        if classifier:
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict)

        if (is_master_proc):
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        if (is_master_proc):
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return start_epoch, model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, distb):
    margin = 0
    pred = (distb - dista - margin)
    return (pred > 0).sum() * 1.0 / (dista.size()[0])
