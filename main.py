import torch
import argparse
import matplotlib.pyplot as plt
from model import JDASA
from utile import set_seed, get_data, get_loaders, train_func
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help="random seed of dataset and model")
parser.add_argument('--dataset_name', type=str, default='MDAD', choices=['MDAD', 'aBiofilm', 'DrugVirus'])
parser.add_argument('--num_test', type=float, default=0.2, help='ratio of test datasets')
parser.add_argument('--ratio', type=float, default=1, help='ratio of positive samples and negative samples')

parser.add_argument('--AE_epoch', type=int, default=10, help='number of autoencoder training epoch')
parser.add_argument('--AE_emb', type=int, default=128, help='embedding of autoencoder output')

parser.add_argument('--hops', type=int, default=2, help="k-hop subgraph[1,2,3]")
parser.add_argument('--s_dim', type=int, default=64, help="feature dimension of subgraph")
parser.add_argument('--hidden_dim', type=int, default=1024)

parser.add_argument('--train_epoch', type=int, default=200, help='number of training times')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size of dataset')
args = parser.parse_args()

set_seed(args.seed)
dataset, splits = get_data(args)
train_loader, test_loader = get_loaders(args, splits)

model = JDASA(args, dataset.num_features).cuda()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

train_func(args, train_loader, test_loader, model, optimizer, loss_fn)

