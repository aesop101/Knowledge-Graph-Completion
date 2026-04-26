import random
import numpy as np
import torch
import os

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
