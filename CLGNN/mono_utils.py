# coding=utf-8

import torch
import os
import numpy as np

from rphgnn.utils.graph_utils import add_random_feats, dgl_add_all_reversed_edges, dgl_remove_edges
from tqdm import tqdm
import pickle
import time
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from torch_sparse import SparseTensor


def load_mag(device):
    
    # path = args.use_emb
    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset"))
    g, labels = dataset[0]

    # my
    #g = g.to(device)

    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    paper_year=g.nodes['paper'].data['year']

    # g.nodes["paper"].data["feat"] = features.to(device)


    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)




    adjs = []
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)


    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('author', 'affiliated_with', 'institution'),
        ('author', 'writes', 'paper'),
        ('paper', 'cites', 'paper'),
        ('paper', 'has_topic', 'field_of_study'),
    ]

    adjs[2] = adjs[2].to_symmetric()
    assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype==dtype:
            newsrc=[]
            newdst=[]
            for i in tqdm(range(len(src))):
                if paper_year[src[i]]<=paper_year[dst[i]]:
                    newsrc.append(src[i])
                    newdst.append(dst[i])
            newsrc, newdst=np.array(newsrc), np.array(newdst)
            new_edges[(stype, rtype, dtype)] = (newsrc, newdst)
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, 'r.'+rtype, stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)

    new_g = dgl.heterograph(new_edges)
    for ntype in g.ntypes:
        for k in g.nodes[ntype].data.keys():
            print(f"debug : {ntype}, {k}, {g.nodes[ntype].data[k]}")
            new_g.nodes[ntype].data[k]=g.nodes[ntype].data[k]
    new_g.nodes['paper'].data['paper'] = g.nodes['paper'].data['feat'].to(device)


    target_node_type = "paper"
    feature_node_types = [target_node_type]
    new_g = new_g.to(device)
    return new_g, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid




def load_dgl_mag(embedding_size):
    device = "cpu"
    
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_mag(device)

    g.nodes[target_node_type].data["label"] = labels


    # embedding_size = g.ndata["feat"][target_node_type].size(-1) * 4  
    g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)


def load_dgl_data(dataset, use_all_feat=False, embedding_size=None, use_nrl=False, random_state=None):

    batch_size = 10000
    num_epochs = 510
    patience = 30
    validation_freq = 10
    convert_to_tensor = True

    assert(dataset == "mag")
    hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_mag(embedding_size=embedding_size)        

    convert_to_tensor = False
    num_epochs = 100
    patience = 10




    return hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index), \
           batch_size, num_epochs, patience, validation_freq, convert_to_tensor
