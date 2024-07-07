import torch
from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput


class EncoderLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mh = MultiHead(args)
        self.fc = FullyConnectedOutput(args)

    def forward(self, x, mask):
        # [b, 50, 32]
        score = self.mh(x, x, x, mask) # [b, 50, 32]
        out = self.fc(score) # [b, 50, 32]

        return out


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer_1 = EncoderLayer(args)
        self.layer_2 = EncoderLayer(args)
        self.layer_3 = EncoderLayer(args)

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x # [b, 50, 32]


class DecoderLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.mh1 = MultiHead(args)
        self.mh2 = MultiHead(args)
        self.fc = FullyConnectedOutput(args)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y) # [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x) # [b, 50, 32]
        y = self.fc(y) # [b, 50, 32]

        return y


class Decoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.layer_1 = DecoderLayer(args)
        self.layer_2 = DecoderLayer(args)
        self.layer_3 = DecoderLayer(args)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y # [b, 50, 32]


class Transformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_x = PositionEmbedding(args)
        self.embed_y = PositionEmbedding(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.fc_out = torch.nn.Linear(args.dim, args.word_dict_length)

    def forward(self, x, y):
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        x, y = self.embed_x(x), self.embed_y(y) # [b, 50, 32]
        x = self.encoder(x, mask_pad_x) # [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y) # [b, 50, 32]
        y = self.fc_out(y)  # [b, 50, 14]

        return y
