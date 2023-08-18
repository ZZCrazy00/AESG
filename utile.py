import torch
import random
import numpy as np
import tensorflow as tf
from dataset import MyDataset, HashDataset
from torch.utils.data import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tf.random.set_seed(seed)


def get_data(args):
    dataset = MyDataset(args)
    transform = RandomLinkSplit(is_undirected=True, num_val=0, num_test=args.num_test,
                                add_negative_train_samples=True, neg_sampling_ratio=args.ratio)
    train_data, _, test_data = transform(dataset.data)
    splits = {'train': train_data, 'test': test_data}
    return dataset, splits


def get_pos_neg_edges(data):
    pos_edges = data['edge_label_index'][:, data['edge_label'] == 1].t()
    neg_edges = data['edge_label_index'][:, data['edge_label'] == 0].t()
    return pos_edges, neg_edges


def get_hashed_train_val_test_datasets(args, train_data, test_data):
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    train_dataset = HashDataset(args, train_data, pos_train_edge, neg_train_edge)
    test_dataset = HashDataset(args, test_data, pos_test_edge, neg_test_edge)
    return train_dataset, test_dataset


def get_loaders(args, splits):
    train_data, test_data = splits['train'], splits['test']
    train_dataset, test_dataset = get_hashed_train_val_test_datasets(args, train_data, test_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def train_model(data_loader, model, optimizer, loss_fn):
    model.train()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    sample_indices = torch.randperm(len(labels))[:len(labels)]
    links = data.links[sample_indices]
    labels = labels[sample_indices]
    total_loss = 0
    for batch_count, indices in enumerate(DataLoader(range(len(links)), batch_size=8192, shuffle=True)):
        subgraph_features = data.subgraph_features[sample_indices[indices]].cuda().to(torch.float32)
        node_features = data.x[links[indices]].cuda()
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features)
        loss = loss_fn(logits.view(-1), labels[indices].squeeze(0).cuda().to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


@torch.no_grad()
def test_model(data_loader, model):
    model.eval()
    data = data_loader.dataset
    labels = torch.tensor(data.labels)
    preds = []
    for batch_count, indices in enumerate(DataLoader(range(len(data.links)), shuffle=False)):
        curr_links = data.links[indices]
        subgraph_features = data.subgraph_features[indices].cuda().to(torch.float32)
        node_features = data.x[curr_links].cuda()
        logits = model(subgraph_features, node_features)
        preds.append(logits.view(-1).cpu())
    pred = torch.cat(preds)
    labels = labels[:len(pred)]

    auc = roc_auc_score(labels, pred)
    AP = average_precision_score(labels, pred)
    temp = torch.tensor(pred)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    accuracy, sensitivity, precision, specificity, F1_score, mcc = calculate_metrics(labels, temp.cpu())
    return ['AUC:{:.6f}'.format(auc), 'AP:{:.6f}'.format(AP),
            'acc:{:.6f}'.format(accuracy.item()), 'sen:{:.6f}'.format(sensitivity.item()),
            'pre:{:.6f}'.format(precision.item()), 'spe:{:.6f}'.format(specificity.item()),
            'f1:{:.6f}'.format(F1_score.item()), 'mcc:{:.6f}'.format(mcc.item())]


def train_func(args, train_loader, test_loader, model, optimizer, loss_fn):
    for epoch in range(args.train_epoch):
        loss = train_model(train_loader, model, optimizer, loss_fn)
    result = test_model(test_loader, model)
    print(result)
