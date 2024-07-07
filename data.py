import torch
import numpy as np
import random


zidian_x = '0,1,2,3,4,5,6,7,8,9,+,<SOS>,<EOS>,<PAD>'
zidian_x = {word: i for i, word in enumerate(zidian_x.split(','))}
zidian_xr = [word for word, _ in zidian_x.items()]

def get_data():
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    p = np.array([1] * 10)
    p = p / p.sum()

    n = random.randint(1, 20)
    s1 = list(np.random.choice(words, size=n, replace=True, p=p))
    n = random.randint(1, 20)
    s2 = list(np.random.choice(words, size=n, replace=True, p=p))

    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y))
    x = s1 + ['+'] + s2

    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]

    x = [zidian_x[i] for i in x]
    y = [zidian_x[i] for i in y]

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    return x, y

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 100000

    def __getitem__(self, i):
        return get_data()

loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=8, drop_last=True, shuffle=True)

if __name__ == "__main__":
    x, y = get_data()
    print(x)
    print(y)
