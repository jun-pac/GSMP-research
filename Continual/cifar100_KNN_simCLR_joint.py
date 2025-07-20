
from os.path import expanduser, isfile
from annoy import AnnoyIndex
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
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
from SimCLR.models.resnet_simclr import ResNetSimCLR
from SimCLR.simclr import SimCLR
from torch.cuda.amp import GradScaler, autocast
import os
import tarfile
from PIL import Image


def tensor_to_pil(tensor):
    return Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).numpy())



def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(size,size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
    return data_transforms

def get_original_pipeline_transform(size):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    data_transforms = transforms.Compose([transforms.Resize(size=(size,size)),
                                            transforms.ToTensor()])
    return data_transforms


class TensorDatasetFromTar_custom(Dataset):
    def __init__(self, tar_file, transform=None):
        self.tar_file = tar_file
        self.transform = transform
        self.members = []
        self.label_name = [0]*1000
        self.label_dict = {}
        self.label_cnt = 0
        m=0
        with tarfile.open(tar_file, 'r') as tar:
            # Collect all members ending with ".JPEG" assuming they are tensors
            for member in tar.getmembers():
                if(member.name.endswith(".JPEG")):
                    m+=1
                    self.members.append(member.name)
                    class_name=member.name.split('/')[0]
                    if(class_name not in self.label_dict):
                        self.label_dict[class_name]=self.label_cnt
                        self.label_name[self.label_cnt]=class_name
                        self.label_cnt+=1
        print(f"Total label count : {self.label_cnt}",flush=True)
        print(f"members count: {len(self.members)}",flush=True)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        with tarfile.open(self.tar_file, 'r') as tar:
            member = self.members[idx]
            # Load the tensor
            file = tar.extractfile(member)
            img = torch.load(file)
            img = tensor_to_pil(img)
            # Apply transform if provided
            if self.transform:
                img = self.transform(img)
            # Extract the label from the directory name
            class_name = member.split('/')[0]
            # print(f"sample class_name: {class_name}")
            label = int(self.label_dict[class_name])  # You might need to map this correctly based on your setup
            return img, label




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
    if args.dataset=="CIFAR100":
        trainset=datasets.CIFAR100(root='./data', train=True, download=True, transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.in_size),2))
        trainset_original = datasets.CIFAR100(root='./data', train=True, download=True, transform=get_original_pipeline_transform(args.in_size))
        testset_original = datasets.CIFAR100(root='./data', train=False, download=True, transform=get_original_pipeline_transform(args.in_size))
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


    elif args.dataset=="ImageNet":

        # trainset=datasets.ImageFolder(root='/fs/ess/PAS1289/ImageNet/imagenet/train', transform=ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.in_size),2))
        # trainset_original = datasets.ImageFolder(root='/fs/ess/PAS1289/ImageNet/imagenet/train', transform=get_original_pipeline_transform(args.in_size))
        # testset_original = datasets.ImageFolder(root='/fs/ess/PAS1289/ImageNet/imagenet/val', transform=get_original_pipeline_transform(args.in_size))

        # testset_original = TensorDatasetFromTar_custom('/users/PAS1289/oiocha/Persistent_Message_Passing/Continual/data/ImageNet/valid_images.tar',ContrastiveLearningViewGenerator(get_original_pipeline_transform(args.in_size)))
        # print(f"len(testset_original) : {testset_original.__len__()}",flush=True)
        # print(type(testset_original))
        # temp_loader=DataLoader(testset_original,batch_size=args.batch_size, num_workers=2, drop_last=True)
        # print(f"len(temp_loader.dataset) : {len(temp_loader.dataset)}")
        # cntt=0
        # for i,(img,label) in enumerate(temp_loader):
        #     if cntt>10:
        #         break
        #     print(f"is it work? {i}, {type(img)}, {label}",flush=True)
        #     cntt+=1
        trainset_original = TensorDatasetFromTar_custom('/users/PAS1289/oiocha/Persistent_Message_Passing/Continual/data/ImageNet/train_images.tar',ContrastiveLearningViewGenerator(get_original_pipeline_transform(args.in_size)))
        trainset = TensorDatasetFromTar_custom('/users/PAS1289/oiocha/Persistent_Message_Passing/Continual/data/ImageNet/train_images.tar',ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(args.in_size),2))
        testset_original = TensorDatasetFromTar_custom('/users/PAS1289/oiocha/Persistent_Message_Passing/Continual/data/ImageNet/valid_images.tar',ContrastiveLearningViewGenerator(get_original_pipeline_transform(args.in_size)))



        if (not isfile("/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl")) or (not isfile("/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl")):
            def split_ImageNet1K_by_labels(dataset, num_splits=10):
                split_indices = []
                labels_per_split = 1000 // num_splits
                t_begin=time.time()
                split_indices=[[] for split_idx in range(num_splits)]
                print(f"Start processing...",flush=True)
                for i, (img, label) in enumerate(dataset):
                    if i%10000==9999:
                        print(f"number {i}/{len(dataset)} time: {time.time()-t_begin:.2f}",flush=True)
                    split_indices[label//labels_per_split].append(i)
                return split_indices
            
            train_split_indices = split_ImageNet1K_by_labels(trainset, num_splits=10)
            test_split_indices = split_ImageNet1K_by_labels(testset_original, num_splits=10)
            
            with open('/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl', 'wb') as f:
                pickle.dump(train_split_indices, f)
            with open('/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl', 'wb') as f:
                pickle.dump(test_split_indices, f)
            print("Train and test split indices have been saved.")

        with open('/fs/ess/PAS1289/ImageNet/imagenet_train_split_indices.pkl', 'rb') as f:
            train_split_indices = pickle.load(f)
        with open('/fs/ess/PAS1289/ImageNet/imagenet_test_split_indices.pkl', 'rb') as f:
            test_split_indices = pickle.load(f)

        print(f"Train split sizes:")
        for i in range(10):
            print(f"Split {i}: {len(train_split_indices[i])}")
        print(f"Test split sizes:")
        for i in range(10):
            print(f"Split {i}: {len(test_split_indices[i])}")
        print()
        

        train_loaders = [DataLoader(Subset(trainset, indices), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True) 
                    for indices in train_split_indices]
        train_loaders_original = [DataLoader(Subset(trainset_original, indices), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True) for indices in train_split_indices]
        test_loaders = [DataLoader(Subset(testset_original, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True) 
                        for indices in test_split_indices]
        
        print(f"train_loaders: {train_loaders}")
        print(f"type(train_loaders[0]): {type(train_loaders[0])}")
        print(f"len(train_loaders[0].dataset): {len(train_loaders[0].dataset)}")
    else:
        print("Dataset Not Supported")
        exit()


    
    



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
        print(f"Current task number: {task}, len train dataset: {len(train_loaders[task].dataset)}")
        if(task==7):
            break
        if task+1 in [1,2,3,4,5,6,7,8]:
            print(f"Joint training begin in {task}th task")
            model = ResNetSimCLR(out_dim=args.out_dim, pretrained=args.pretrained)
            model = model.to(args.device)

            all_prev_dataset=ConcatDataset([train_loaders[i].dataset for i in range(task+1)])
            all_prev_trainloader=DataLoader(all_prev_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2,pin_memory=True, drop_last=True)
            # print(f"test: {type(all_prev_trainloader)}, {len(all_prev_trainloader.dataset)}")
            # (img, label)=next(all_prev_trainloader)
            # print(f"debug : {img}, {label}")
            try:
                iter(all_prev_trainloader).__next__()
                print("Dataset iteration works!")
            except NotImplementedError as e:
                print("Iteration error:", e)
            # for i, (img, label) in enumerate(all_prev_trainloader):
            #     print(f"i, img, label; {i} {type(img)} {label}")
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
            print(f"{j}th validation set size: {len(test_features)}")
            predictions_wo_head = []
            for test_feature in test_features:
                neighbors = annoy_index_wo_head.get_nns_by_vector(test_feature, args.k, include_distances=False)
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
    parser.add_argument('--k', default=5, type=int)

    parser.add_argument('--dataset', default="CIFAR100", type=str, choices=['CIFAR100', 'ImageNet', 'CGLM', 'CLOC'])

    args = parser.parse_args()
    main(args)


# Always use pitzer
# No need to use gaussian blur
# Pretrained is much better
# 'WO projection head' is super important
# conda activate vv

# python cifar100_KNN_simCLR_joint.py --epochs 150 --annoy 10000 --batch-size 500 --in-size 32 --distance angular --k 10 |&tee output/joint_SimCLR_32_wop_ang_150_annoy10000_k10.txt



# python cifar100_KNN_simCLR_joint.py --epochs 1500 --annoy 10000 --batch-size 500 --in-size 32 --distance angular --k 10 |&tee output/joint_SimCLR_32_wop_ang_1500_annoy10000_k10.txt
