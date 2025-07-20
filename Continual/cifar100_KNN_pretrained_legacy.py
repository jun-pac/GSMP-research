
from os.path import expanduser, isfile
from annoy import AnnoyIndex
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import time
import argparse
import torch
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
from networks import *
import pickle
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, GEMPlugin, LwFPlugin

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def extract_features(dataloader, model):
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda() 
            outputs = model(inputs)
            features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    return np.concatenate(features), np.concatenate(labels)





def main(args):
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.eval()
    resnet50 = resnet50.to(device)


    transform = transforms.Compose([
        transforms.Resize(224),  # Should be matched with input size of Resnet50, which is 224*224*3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t_init=time.time()
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    if (not isfile("./train_split_indices.pkl")) or (not isfile("./test_split_indices.pkl")):
        def split_cifar100_by_labels(dataset, num_splits=10):
            split_indices = []
            labels_per_split = 100 // num_splits
            print(f"hello, {num_splits}")

            for split_idx in range(num_splits):
                print(f"here ; {split_idx}")
                label_start = split_idx * labels_per_split
                label_end = label_start + labels_per_split
                indices = [i for i, (img, label) in enumerate(dataset) if label_start <= label < label_end]
                split_indices.append(indices)
            
            return split_indices
        
        train_split_indices = split_cifar100_by_labels(trainset, num_splits=10)
        test_split_indices = split_cifar100_by_labels(testset, num_splits=10)

        with open('train_split_indices.pkl', 'wb') as f:
            pickle.dump(train_split_indices, f)
        with open('test_split_indices.pkl', 'wb') as f:
            pickle.dump(test_split_indices, f)
        print("Train and test split indices have been saved.")


    with open('train_split_indices.pkl', 'rb') as f:
        train_split_indices = pickle.load(f)
    with open('test_split_indices.pkl', 'rb') as f:
        test_split_indices = pickle.load(f)

    batch_size = 50
    train_loaders = [DataLoader(Subset(trainset, indices), batch_size=batch_size, shuffle=True, num_workers=2) 
                    for indices in train_split_indices]
    test_loaders = [DataLoader(Subset(testset, indices), batch_size=batch_size, shuffle=False, num_workers=2) 
                    for indices in test_split_indices]


    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    all_prev_train_features=np.zeros((50000,2048))
    all_prev_train_labels=np.zeros(50000).astype(int)
    cur_mem_size=0

    for task,trainloader in enumerate(train_loaders):
        print(f"Current task number {task}")
        train_features, train_labels = extract_features(trainloader, resnet50)
        d = train_features.shape[1] # 2048 in Resnet50

        annoy_index = AnnoyIndex(d, 'euclidean')  
        for i in range(train_features.shape[0]):
            all_prev_train_features[cur_mem_size]=train_features[i]
            all_prev_train_labels[cur_mem_size]=train_labels[i]
            cur_mem_size+=1

        t0=time.time()    
        for i in range(len(all_prev_train_features)):
            annoy_index.add_item(i,all_prev_train_features[i])
        annoy_index.build(10) 
        print(f"annoy_index rebuild time : {time.time()-t0:.4f}")

        acc_list=[]
        for j, testloader in enumerate(test_loaders):
            test_features, test_labels = extract_features(testloader, resnet50)
            k = 5 
            predictions = []
            for test_feature in test_features:
                neighbors = annoy_index.get_nns_by_vector(test_feature, k, include_distances=False)
                neighbor_labels = all_prev_train_labels[neighbors]
                predictions.append(np.bincount(neighbor_labels).argmax())
            accuracy = np.mean(np.array(predictions) == test_labels)
            acc_list.append(accuracy)
            print(f"{task}-th task, {j}-th test set accuracy : {accuracy:.4f}")
        results.append(acc_list)


    print(f"=== Results summary ===")
    for i,acclist in enumerate(results):
        print(f"Accuracy after {i}-th task : ",end=' ')
        for acc in acclist:
            print(acc,end=' ')
        print() 

    print(f"=== Results summary2 ===")
    for i,acclist in enumerate(results):
        for acc in acclist:
            print(acc)
        print() 
    print(f"Total training time: {time.time()-t_init:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)

# conda activate vv
# python cifar100_KNN_pretrained.py |&tee out_pretrained.txt

