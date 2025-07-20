
from os.path import expanduser, isfile
from annoy import AnnoyIndex
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from tqdm import tqdm
import time
import argparse
import torch
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from networks import *
import pickle

# Multiple dataloaders
def extract_all_prev_features(dataloaders, model, num_tasks):
    features = []
    labels = []
    with torch.no_grad():
        for idx in range(num_tasks):
            dataloader=dataloaders[idx]
            for inputs, targets in dataloader:
                inputs = inputs.cuda() 
                outputs = model(inputs)
                features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
                labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)


# Single dataloader
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

    transform = transforms.Compose([
        transforms.Resize(size=(args.in_size,args.in_size)),  # Should be matched with input size of Resnet50, which is 224*224*3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t_init=time.time()
    # if args.dataset=="CIFAR100":
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    if (not isfile("./train_split_indices.pkl")) or (not isfile("./test_split_indices.pkl")):
        def split_cifar100_by_labels(dataset, num_splits=10):
            split_indices = []
            labels_per_split = 100 // num_splits

            for split_idx in range(num_splits):
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

    train_loaders = [DataLoader(Subset(trainset, indices), batch_size=args.batch_size, shuffle=True, num_workers=2) for indices in train_split_indices]
    test_loaders = [DataLoader(Subset(testset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2) 
                    for indices in test_split_indices]


    # elif args.dataset=="ImageNet":
    #     trainset = datasets.ImageFolder(root='/fs/ess/PAS1289/ImageNet/imagenet/train', transform=transform)
    #     testset= datasets.ImageFolder(root='/fs/ess/PAS1289/ImageNet/imagenet/val', transform=transform)
    #     if (not isfile("/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl")) or (not isfile("/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl")):
    #         def split_ImageNet1K_by_labels(dataset, num_splits=10):
    #             split_indices = []
    #             labels_per_split = 1000 // num_splits
    #             t_begin=time.time()
    #             split_indices=[[] for split_idx in range(num_splits)]
    #             print(f"Start processing...",flush=True)
    #             for i, (img, label) in enumerate(dataset):
    #                 if i%10000==9999:
    #                     print(f"number {i}/{len(dataset)} time: {time.time()-t_begin:.2f}",flush=True)
    #                 split_indices[label//labels_per_split].append(i)
    #             return split_indices
            
    #         train_split_indices = split_ImageNet1K_by_labels(trainset, num_splits=10)
    #         test_split_indices = split_ImageNet1K_by_labels(testset, num_splits=10)
            
    #         with open('/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl', 'wb') as f:
    #             pickle.dump(train_split_indices, f)
    #         with open('/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl', 'wb') as f:
    #             pickle.dump(test_split_indices, f)
    #         print("Train and test split indices have been saved.")

    #     with open('/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl', 'rb') as f:
    #         train_split_indices = pickle.load(f)
    #     with open('/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl', 'rb') as f:
    #         test_split_indices = pickle.load(f)

    #     print(f"Train split sizes:")
    #     for i in range(10):
    #         print(f"Split {i}: {len(train_split_indices[i])}")
    #     print(f"Test split sizes:")
    #     for i in range(10):
    #         print(f"Split {i}: {len(test_split_indices[i])}")
    #     print()

    #     train_loaders = [DataLoader(Subset(trainset, indices), batch_size=args.batch_size, shuffle=True, num_workers=2) for indices in train_split_indices]
    #     test_loaders = [DataLoader(Subset(testset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2) 
    #                     for indices in test_split_indices]

    learning_rate = 0.001
    


    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    for task in range(len(train_loaders)):
        print(f"Current task number: {task}")
        if(task==7):
            break
        train_list=[1,2,4,8]
        if args.joint:
            train_list=[1,2,3,4,5,6,7,8]
        print("Train_list",train_list)
        if task+1 in train_list:
            print(f"Joint training begin in {task}th task")
            model = models.resnet50(pretrained=False)
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            all_prev_dataset=ConcatDataset([train_loaders[i].dataset for i in range(task+1)])
            all_prev_trainloader=DataLoader(all_prev_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
            for epoch in range(args.epochs):
                t1=time.time()
                for inputs, labels in all_prev_trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                print(f"{epoch}th train epoch finished ...{time.time()-t1:.4f}sec")



        resnet50 = nn.Sequential(*list(model.children())[:-1])
        resnet50.eval()
        resnet50 = resnet50.to(device)
        annoy_index = AnnoyIndex(2048, 'euclidean')  

        # Extracting train features
        t0=time.time()    
        all_prev_train_features, all_prev_train_labels = extract_all_prev_features(train_loaders, resnet50, task+1)
        print(f"len(all_prev_train_features): {len(all_prev_train_features)}")
        print(f"Train feature extraction time: {time.time()-t0:.4f}")


        t0=time.time()    
        for i in range(len(all_prev_train_features)):
            annoy_index.add_item(i,all_prev_train_features[i])
        annoy_index.build(10) 
        print(f"annoy_index rebuild time: {time.time()-t0:.4f}")

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
        print()

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
    parser.add_argument('--dataset', default="CIFAR100", type=str, choices=['CIFAR100', 'ImageNet', 'CGLM', 'CLOC'])
    parser.add_argument('--in-size', default=224, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=500, type=int, metavar='N')
    parser.add_argument('--joint', action='store_true')

    args = parser.parse_args()
    main(args)

# conda activate vv
# python cifar100_KNN_exponential.py |&tee out_w_pretrained.txt
# python cifar100_KNN_exponential.py |&tee out_wo_pretrained.txt



# Further 
# python cifar100_KNN_exponential.py --epoch 5 |&tee out_wo_pretrained_5.txt
# python cifar100_KNN_exponential.py --epoch 10 |&tee out_wo_pretrained_10.txt
# python cifar100_KNN_exponential.py --epoch 15 |&tee out_wo_pretrained_15.txt
# python cifar100_KNN_exponential.py --epoch 20 |&tee out_wo_pretrained_20.txt
# python cifar100_KNN_exponential.py --epoch 25 |&tee out_wo_pretrained_25.txt

# python cifar100_KNN_exponential.py --epoch 40 |&tee out_wo_pretrained_40.txt



# 6h
# python cifar100_KNN_exponential.py --joint --epoch 30 |&tee out_wo_pretrained_joint_30.txt 
