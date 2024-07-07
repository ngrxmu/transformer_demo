import math
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Demo")
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate (default: 2e-3)')
    parser.add_argument('--seed', type=int, default=89, help='random seed (default: 89)')
    parser.add_argument('--head_num', type=int, default=4, help='multi-head numbers (default: 4)')
    parser.add_argument('--dim', type=int, default=32, help='dimention (default: 32)')
    parser.add_argument('--max_length', type=int, default=50, help='max token length (default: 50)')
    parser.add_argument('--word_dict_length', type=int, default=14, help='word dict length (default: 14)')
    parser.add_argument('--log_path', type=str, default='logs', help='log file (default: logs)')
    parser.add_argument('--log_interval', type=int, default=200, help='how many batches to wait before logging training status (default: 200)')
    
    return parser.parse_args()


def attention(Q, K, V, mask):
    # [b, 4, 50, 8]
    dim = Q.shape[1] * Q.shape[3]

    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) # [b, 4, 50, 50]

    score /= Q.shape[3] ** 0.5

    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    score = torch.matmul(score, V) # [b, 4, 50, 8]

    score = score.permute(0, 2, 1, 3).reshape(-1, 50, dim)

    return score


class MultiHead(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc_Q = torch.nn.Linear(args.dim, args.dim)
        self.fc_K = torch.nn.Linear(args.dim, args.dim)
        self.fc_V = torch.nn.Linear(args.dim, args.dim)

        self.out_fc = torch.nn.Linear(args.dim, args.dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=args.dim, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # [b, 50, 32]
        clone_Q = Q.clone()

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K) # [b, 50, 32]
        V = self.fc_V(V) # [b, 50, 32]
        Q = self.fc_Q(Q) # [b, 50, 32]

        Q = Q.reshape(-1, self.args.max_length, self.args.head_num, self.args.dim // self.args.head_num).permute(0, 2, 1, 3) # [b, 4, 50, 8]
        K = K.reshape(-1, self.args.max_length, self.args.head_num, self.args.dim // self.args.head_num).permute(0, 2, 1, 3) # [b, 4, 50, 8]
        V = V.reshape(-1, self.args.max_length, self.args.head_num, self.args.dim // self.args.head_num).permute(0, 2, 1, 3) # [b, 4, 50, 8]

        score = attention(Q, K, V, mask) # [b, 50, 32]
        score = self.dropout(self.out_fc(score)) # [b, 50, 32]
        score = clone_Q + score # [b, 50, 32]
        return score


class PositionEmbedding(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = torch.nn.Embedding(args.word_dict_length, args.dim)
        self.embed.weight.data.normal_(0, 0.1)

        def get_pe(pos, i, dim):
            # 当前位置 当前维度 词向量编码维度
            fenmu = 1e4 ** (i / dim)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(args.max_length, args.dim)
        for i in range(args.max_length):
            for j in range(args.dim):
                pe[i, j] = get_pe(i, j, args.dim)
        
        pe = pe.unsqueeze(0) # [1, 50, 32]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [8, 50]
        embed = self.embed(x) # [b, 50, 32]
        embed = embed + self.pe # [b, 50, 32]
        return embed


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.dim, out_features=args.dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=args.dim * 2, out_features=args.dim),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=args.dim, elementwise_affine=True)

    def forward(self, x):
        # [b, 50, 32]
        clone_x = x.clone()

        x = self.norm(x)
        out = self.fc(x) # [b, 50, 32]

        out = clone_x + out

        return out
