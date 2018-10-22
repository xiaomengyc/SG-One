

import torch

def get_model_para_number(model):
    total_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)

    return total_number
