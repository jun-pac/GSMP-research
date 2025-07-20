import argparse

import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
import os

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size)) # Self connection?
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = nn.functional.normalize(h,p=2,dim=0)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(args.epoch):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = evaluate(g, features, labels, val_mask, model)
        # if(epoch%500==499):
        #     #print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))
        #     #f_log.write("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} \n".format(epoch, loss.item(), acc))
        #     #f_log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="3_sym")
    parser.add_argument("--dt", type=str, default="float",help="data type(float, bfloat16)")
    parser.add_argument("--root",type=str, default='./data/')
    parser.add_argument("--epoch", type=int, default=200)

    args = parser.parse_args()
    print(f"Training GCN ; {args.dataset}")    
    
    assert(os.path.exists(args.root+args.dataset))
    data = load_graphs(args.root+args.dataset)
    g=data[0][0]
    init_label=data[1]['label']
    #print(f"init_label : {init_label.shape}") # torch.Size([1000]) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].type(torch.bool), g.ndata["val_mask"].type(torch.bool), g.ndata["test_mask"].type(torch.bool)

    # print(f"type(data) : {type(data)}")
    # print(f"type(g) : {type(g)}")
    # for k in g.ndata.keys():
    #     print(f"key : {k} | shape : {g.ndata[k].shape}")
    # for k in g.ndata.keys():
    #     print(f"key : {k} | shape : {g.ndata[k].shape} | value : {g.ndata[k]}")
    
    # create GCN model
    in_size = features.shape[1]
    out_size = 10 ########### How to modify this?
    model = GCN(in_size, 16, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    f_log=open('./txtlog/'+'NORM_'+args.dataset+'.txt','a')

    # model training
    #print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    #print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
    f_log.write("{:.4f}\n".format(acc))
    f_log.flush()

