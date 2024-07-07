import torch
import os
from data import zidian_x, loader, zidian_xr
from mask import mask_pad, mask_tril
from model import Transformer

from util import parse_args
args = parse_args()

import logging
def setup_logger(log_file):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
setup_logger(log_file=args.log_path+'/log.log')
logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(args.log_path+'/loss')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(args).to(device)


def predict(x):
    # [1, 50]
    model.eval()

    mask_pad_x = mask_pad(x) # [1, 1, 50, 50]

    target = [zidian_x['<SOS>']] + [zidian_x['<PAD>']] * (args.max_length - 1)
    target = torch.LongTensor(target).unsqueeze(0) # [1, 50]

    x = model.embed_x(x) # [1, 50, 32]
    x = model.encoder(x, mask_pad_x) # [1, 50, 32]

    # 遍历生成第1个词到第49个词
    for i in range(args.max_length - 1):
        y = target # [1, 50]
        mask_tril_y = mask_tril(y) # [1, 1, 50, 50]
        y = model.embed_y(y) # [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y) # [1, 50, 32]
        out = model.fc_out(y) # [1, 50, 14]
        out = out[:, i, :] # [1, 14]
        out = out.argmax(dim=1).detach() # [1]
        target[:, i + 1] = out

    return target


def train():
    logger.info('Start Training')
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

    loss_log_no = 1
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # [8, 50]
            pred = model(x, y[:, :-1]) # [8, 50, 14]
            pred = pred.reshape(-1, args.word_dict_length) # [400, 14]
            y = y[:, 1:].reshape(-1) # [400]

            # 忽略PAD
            select = y != zidian_x['<PAD>']
            pred = pred[select]
            y = y[select]

            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % args.log_interval == 0:
                pred = pred.argmax(1) # [select, 14] -> [select]
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                logger.info(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}, acc: {accuracy}, lr: {lr}')
                writer.add_scalar('Training Loss', loss.item(), loss_log_no)
                loss_log_no += 1

        sched.step()
    logger.info('End Training')
    torch.save(model.state_dict(), 'model_state_dict.pth')
    # model.load_state_dict(torch.load('model_state_dict.pth'))


def test(mode='test'):
    # mode: test/user
    logger.info('Start Test')
    if mode == 'user':
        while True:
            x = input('input: ')
            if x == 'Q' or x == 'q':
                logger.info('Quit')
                break
            x = ['<SOS>'] + list(x) + ['<EOS>'] + ['<PAD>'] * args.max_length
            x = x[:50]
            x = torch.LongTensor([zidian_x[i] for i in x])
            logger.info('input: ' + ''.join([zidian_xr[i] for i in list(x)]))
            x = x.to(device)
            logger.info('output: ' + ''.join([zidian_xr[i] for i in list(predict(x.unsqueeze(0))[0])]))
    else:
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            break

        for i in range(x.shape[0]):
            logger.info('input: ' + ''.join([zidian_xr[i] for i in x[i].tolist()]))
            logger.info('gt: ' + ''.join([zidian_xr[i] for i in y[i].tolist()]))
            logger.info('output: ' + ''.join([zidian_xr[i] for i in list(predict(x[i].unsqueeze(0))[0])]))


if __name__ == "__main__":
    train()
    test()