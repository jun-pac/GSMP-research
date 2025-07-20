
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
from SimCLR.data_aug.gaussian_blur import GaussianBlur
from SimCLR.data_aug.view_generator import ContrastiveLearningViewGenerator
from SimCLR.exceptions.exceptions import InvalidDatasetSelection
from SimCLR.models.resnet_simclr_mixed import ResNetSimCLR
from SimCLR.simclr_mixed import SimCLR
from torch.cuda.amp import GradScaler, autocast




def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
    return data_transforms

def get_original_pipeline_transform(size):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    data_transforms = transforms.Compose([transforms.Resize(size=size),
                                            transforms.ToTensor()])
    return data_transforms






def main(args):
    args = parser.parse_args()
    print(args)

    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize(args.in_size),  # Should be matched with input size of Resnet50, which is 224*224*3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    t_init=time.time()
    trainset=datasets.CIFAR100(root='./data', train=True, download=True, transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.in_size),2))
    trainset_original = datasets.CIFAR100(root='./data', train=True, download=True, transform=get_original_pipeline_transform(args.in_size))
    testset_original = datasets.CIFAR100(root='./data', train=False, download=True, transform=get_original_pipeline_transform(args.in_size))

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
        test_split_indices = split_cifar100_by_labels(testset_original, num_splits=10)

        with open('train_split_indices.pkl', 'wb') as f:
            pickle.dump(train_split_indices, f)
        with open('test_split_indices.pkl', 'wb') as f:
            pickle.dump(test_split_indices, f)
        print("Train and test split indices have been saved.")


    with open('train_split_indices.pkl', 'rb') as f:
        train_split_indices = pickle.load(f)
    with open('test_split_indices.pkl', 'rb') as f:
        test_split_indices = pickle.load(f)

    
    

    train_loaders = [DataLoader(Subset(trainset, indices), batch_size=args.batch_size, shuffle=True, num_workers=2) 
                    for indices in train_split_indices]
    train_loaders_original = [DataLoader(Subset(trainset_original, indices), batch_size=args.batch_size, shuffle=True, num_workers=2) for indices in train_split_indices]
    test_loaders = [DataLoader(Subset(testset_original, indices), batch_size=args.batch_size, shuffle=False, num_workers=2) 
                    for indices in test_split_indices]


    # Multiple dataloaders
    def extract_all_prev_features(dataloaders, model, num_tasks):
        features = []
        labels = []
        with torch.no_grad():
            for idx in range(num_tasks):
                dataloader=dataloaders[idx]
                for i, (inputs, targets) in enumerate(dataloader):
                    inputs = inputs.to(args.device) 
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
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
                labels.append(targets.cpu().numpy())
        
        return np.concatenate(features), np.concatenate(labels)


    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    results_wo_head = []

    for task in range(len(train_loaders)):
        print(f"Current task number: {task}")
        
        if task+1 in [1,2,4,8]:
            print(f"Joint training begin in {task}th task")
            model = ResNetSimCLR(out_dim=args.out_dim, pretrained=args.pretrained)
            model = model.to(args.device)

            all_prev_dataset=ConcatDataset([train_loaders[i].dataset for i in range(task+1)])
            all_prev_trainloader=DataLoader(all_prev_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True, drop_last=True)

            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(all_prev_trainloader), eta_min=0, last_epoch=-1)

            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(all_prev_trainloader)


        # resnet50=model
        # resnet50.eval()
        # resnet50 = resnet50.to(args.device)
        # annoy_index = AnnoyIndex(args.out_dim, 'euclidean')  

        # Extracting train features
        # t0=time.time()    
        # all_prev_train_features, all_prev_train_labels = extract_all_prev_features(train_loaders_original, resnet50, task+1)
        # print(f"all_prev_train_features.shape: {all_prev_train_features.shape}",flush=True)
        # print(f"Train feature extraction time: {time.time()-t0:.4f}",flush=True)

        # t0=time.time()    
        # for i in range(len(all_prev_train_features)):
        #     annoy_index.add_item(i,all_prev_train_features[i])
        # annoy_index.build(10) 
        # print(f"annoy_index rebuild time: {time.time()-t0:.4f}",flush=True)
        
        # acc_list=[]
        # for j, testloader in enumerate(test_loaders):
        #     test_features, test_labels = extract_features(testloader, resnet50)
        #     k = 5 
        #     predictions = []
        #     for test_feature in test_features:
        #         neighbors = annoy_index.get_nns_by_vector(test_feature, k, include_distances=False)
        #         neighbor_labels = all_prev_train_labels[neighbors]
        #         predictions.append(np.bincount(neighbor_labels).argmax())
        #     accuracy = np.mean(np.array(predictions) == test_labels)
        #     acc_list.append(accuracy)
        #     print(f"{task}-th task, {j}-th test set accuracy : {accuracy:.4f}")
        # results.append(acc_list)
        # print()


        print("Without projection head")
        resnet50_wo_head = model
        resnet50_wo_head.backbone.fc=nn.Identity()
        resnet50_wo_head.eval()
        resnet50_wo_head = resnet50_wo_head.to(args.device)
        annoy_index_wo_head = AnnoyIndex(2048, args.distance)
        all_prev_train_features_wo_head, all_prev_train_labels_wo_head = extract_all_prev_features(train_loaders_original, resnet50_wo_head, task+1)
        print(f"all_prev_train_features_wo_head.shape: {all_prev_train_features_wo_head.shape}")
        t0=time.time()
        for i in range(len(all_prev_train_features_wo_head)):
            annoy_index_wo_head.add_item(i,all_prev_train_features_wo_head[i])
        annoy_index_wo_head.build(args.annoy) 
        print(f"annoy_index rebuild time: {time.time()-t0:.4f}")
        acc_list_wo_head=[]
        for j, testloader in enumerate(test_loaders):
            test_features, test_labels = extract_features(testloader, resnet50_wo_head)
            k = 5 
            predictions_wo_head = []
            for test_feature in test_features:
                neighbors = annoy_index_wo_head.get_nns_by_vector(test_feature, k, include_distances=False)
                neighbor_labels = all_prev_train_labels_wo_head[neighbors]
                predictions_wo_head.append(np.bincount(neighbor_labels).argmax())
            accuracy = np.mean(np.array(predictions_wo_head) == test_labels)
            acc_list_wo_head.append(accuracy)
            print(f"{task}-th task, {j}-th test set accuracy : {accuracy:.4f}")
        results_wo_head.append(acc_list_wo_head)
        print()


    # print(f"=== Results summary ===")
    # for i,acclist in enumerate(results):
    #     print(f"Accuracy after {i}-th task : ",end=' ')
    #     for acc in acclist:
    #         print(acc,end=' ')
    #     print() 

    # print(f"=== Results summary2 ===")
    # for i,acclist in enumerate(results):
    #     for acc in acclist:
    #         print(acc)
    #     print() 

    print("Without projection head")
    print(f"=== Results summary ===")
    for i,acclist in enumerate(results_wo_head):
        print(f"Accuracy after {i}-th task : ",end=' ')
        for acc in acclist:
            print(acc,end=' ')
        print() 

    print(f"=== Results summary2 ===")
    for i,acclist in enumerate(results_wo_head):
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
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether or not to use pretrained ResNet.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=500, type=int,
                        metavar='N')
    parser.add_argument('--distance', default='euclidean', type=str, choices=['angular', 'euclidean', 'manhattan', 'hamming', 'dot'])
    parser.add_argument('--in-size', default=32, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--annoy', default=10, type=int)
    parser.add_argument('--loss-ratio', default=1, type=float)

    args = parser.parse_args()
    main(args)



# Always use pitzer
# No need to use gaussian blur
# Pretrained is much better
# 'WO projection head' is super important
# conda activate vv

# Not experimented 
# python cifar100_KNN_mixed.py --pretrained --epochs 50 --annoy 1000 --batch-size 10 --in-size 32 --distance angular |&tee output/mixed_test.txt
# python cifar100_KNN_mixed.py --pretrained --epochs 100 --annoy 1000 --batch-size 50 --in-size 224 --distance angular |&tee output/mixed_224_w_ang_100_annoy1000.txt
# python cifar100_KNN_mixed.py --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixed_32_w_ang_50_annoy1000.txt



# Experimented
# python cifar100_KNN_mixed.py --loss-ratio 1 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixedh2_1_32_w_50_a1000.txt
# python cifar100_KNN_mixed.py --loss-ratio 2 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixedh2_2_32_w_50_a1000.txt
# python cifar100_KNN_mixed.py --loss-ratio 0.5 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixedh2_0.5_32_w_50_a1000.txt
# python cifar100_KNN_mixed.py --loss-ratio 10 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixedh2_10_32_w_50_a1000.txt
# python cifar100_KNN_mixed.py --loss-ratio 0.2 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixedh2_0.2_32_w_50_a1000.txt


# python cifar100_KNN_mixed.py --loss-ratio 0.05 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixed_0.05_32_w_50_a1000.txt


# Queued
# python cifar100_KNN_mixed.py --loss-ratio 25 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixed_25_32_w_50_a1000.txt
# python cifar100_KNN_mixed.py --loss-ratio 100 --pretrained --epochs 50 --annoy 1000 --batch-size 500 --in-size 32 --distance angular |&tee output/mixed_100_32_w_50_a1000.txt

