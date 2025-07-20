import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import numpy as np
from pecos.utils import smat_util
from tqdm import tqdm
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_root_dir', type=str, default='../../dataset')
    parser.add_argument('--node_emb_path', type=str, default=None)
    args = parser.parse_args()

    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',root=args.data_root_dir,
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    print(f"type(dataset) : {type(dataset)}")
    print(f"dataset : {dataset}")

    print(f"type(data) : {type(data)}")
    print(f"dir(data) : {dir(data)}")
    print(f"data : {data}")

    print(f"data.x.shape : {data.x.shape}") # data.x.shape : torch.Size([169343, 128])
    print(f"data.node_year.shape : {data.node_year.shape}") # torch.Size([169343, 1])
    print(f"data.node_year[:10] : {data.node_year[:10]}")
    print(f"torch.min(data.node_year) : {torch.min(data.node_year)}") # 1971
    print(f"torch.max(data.node_year) : {torch.max(data.node_year)}")  # 2020
    print(f"data.y.shape : {data.y.shape}")
    print(f"type(data.adj_t) : {type(data.adj_t)}") #  <class 'torch_sparse.tensor.SparseTensor'>
    data.adj_t = data.adj_t.to_symmetric()
    coo_adj_t = data.adj_t.coo()

    print(f"coo_adj_t : {type(coo_adj_t)}") # <class 'tuple'>
    print(f"coo_adj_t[0].shape : {coo_adj_t[0].shape}") # row torch.Size([1166243]) -> torch.Size([2315598])
    print(f"coo_adj_t[1].shape : {coo_adj_t[1].shape}") # col torch.Size([1166243]) -> torch.Size([2315598])
    print(f"type(coo_adj_t[2]) : {type(coo_adj_t[2])}") # val <class 'NoneType'>
    print(f"coo_adj_t[0][:30] : {coo_adj_t[0][300:330]}") # in sorted manner 
    print(f"coo_adj_t[1][:30] : {coo_adj_t[1][300:330]}") #
    print(f"coo_adj_t[0].min : {torch.min(coo_adj_t[0])}") # 0
    print(f"coo_adj_t[1].min : {torch.min(coo_adj_t[1])}") # 0
    print(f"coo_adj_t[0].max : {torch.max(coo_adj_t[0])}") # 169341 -> 169342
    print(f"coo_adj_t[1].max : {torch.max(coo_adj_t[1])}") # 169342 -> 169342
    print(f"len(coo_adj_t) : {len(coo_adj_t)}") # 3
    

    src=coo_adj_t[0]
    dst=coo_adj_t[1]
    paper_year=data.node_year
    newsrc=[]
    newdst=[]
    for i in range(len(src)):
        if dst[i]>=2000 and abs(paper_year[src[i]]-paper_year[dst[i]])>min(2020-paper_year[dst[i]],paper_year[dst[i]]-2000):
            newsrc.append(src[i])
            newdst.append(dst[i])  
        # if src[i]>=2000 and abs(paper_year[src[i]]-paper_year[dst[i]])>min(2020-paper_year[src[i]],paper_year[src[i]]-2000):
        #     newsrc.append(src[i])
        #     newdst.append(dst[i])   
    for i in range(len(src)):
        newsrc.append(src[i])
        newdst.append(dst[i])
        
    edge_index=torch.tensor([newsrc,newdst])
    edge_index=SparseTensor.from_edge_index(edge_index)
    #print(f"newsrc.shape : {newsrc.shape}") # newsrc.shape : torch.Size([548500])
    #print(f"newdst.shape : {newdst.shape}") # newdst.shape : torch.Size([548500])
    # new_g=Data(x=data.x, y=data.y, edge_index=edge_index)
    #src,dst=torch.cat((src, newsrc)),torch.cat((dst, newdst)) 

    # print(f"len(src) : {len(src)}") # src.shape : torch.Size([2864098])
    # print(f"len(dst) : {len(dst)}") # dst.shape : torch.Size([2864098])
    data.adj_t=edge_index
    coo_adj_t = data.adj_t.coo()    
    print(f"coo_adj_t[0].shape : {coo_adj_t[0].shape}") # 
    print(f"coo_adj_t[1].shape : {coo_adj_t[1].shape}") # 



    if args.node_emb_path:
        data.x = torch.from_numpy(smat_util.load_matrix(args.node_emb_path).astype(np.float32))
        print("Loaded pre-trained node embeddings of shape={} from {}".format(data.x.shape, args.node_emb_path))

    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()