import torch
from data import zidian_x


def mask_pad(data):
    # [b, 50]
    length = data.shape[1]
    mask = data == zidian_x['<PAD>'] # [b, 50]
    mask = mask.reshape(-1, 1, 1, length)
    mask = mask.expand(-1, 1, length, length)

    return mask


def mask_tril(data):
    # [b, 50]
    device = data.device
    length = data.shape[1]
    tril = (1 - torch.tril(torch.ones(1, length, length, dtype=torch.long))).to(device) # [1, 50, 50]
    mask = data == zidian_x['<PAD>'] # [b, 50]
    mask = mask.unsqueeze(1).long() # [b, 1, 50]

    mask = mask + tril # [b, 50, 50]
    mask = (mask > 0).unsqueeze(1) # [b, 1, 50, 50]

    return mask