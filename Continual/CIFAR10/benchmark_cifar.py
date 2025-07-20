################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Eli Verwimp                                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example shows how to train models provided by pytorchcv with the rehearsal
strategy.
"""

from os.path import expanduser

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # Model getter: specify dataset and depth of the network.
    model = pytorchcv_wrapper.resnet("cifar10", depth=20, pretrained=False)

    # Or get a more specific model. E.g. wide resnet, with depth 40 and growth
    # factor 8 for Cifar 10.
    # model = pytorchcv_wrapper.get_model("wrn40_8_cifar10", pretrained=False)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # --- TRANSFORMATIONS
    transform = transforms.Compose(
        [
            ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
        ]
    )

    # --- BENCHMARK CREATION
    cifar_train = CIFAR10(
        root=expanduser("~") + "/.avalanche/data/cifar10/",
        train=True,
        download=True,
        transform=transform,
    )
    cifar_test = CIFAR10(
        root=expanduser("~") + "/.avalanche/data/cifar10/",
        train=False,
        download=True,
        transform=transform,
    )

    # Is this benchmark model..? like resnet?
    # I think its a dataloader
    benchmark = nc_benchmark( 
        cifar_train,
        cifar_test,
        5,
        task_labels=False,
        seed=1234,
        fixed_class_order=[9-i for i in range(10)],
    )

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay) # Maybe continual learning strategy (including model, optimizer)
    cl_strategy = Naive( # continual? or curriculam? 
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=1,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
    )
    # plugins=[ReplayPlugin(mem_size=10000)],

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    cnt=0
    for experience in benchmark.train_stream:
        cnt+=1
        print(f"type(experience): {type(experience)}, num: {cnt}")
        print(f"dir(experience) : {dir(experience)}")
        print(f"experience.dataset : {experience.dataset}")
        print(f"type(experience.dataset) : {type(experience.dataset)}")
        num=0
        for data in experience.dataset:
            num+=1
            if(num%1000==0):
                print(f"cur num of data : {num}")
                # print(f"len of datas : {len(data)}")
                # print(f"types : {type(data[0])},{type(data[1])},{type(data[2])}")
                print(f"shapes : {data[0].shape},{data[1]},{data[2]}")
                # print(f"class in current data : {data[1]}")
            
        print(f"experience.benchmark : {experience.benchmark}")
        print(f"type(experience.benchmark()) : {type(experience.benchmark)}")
        print(f"experience.train() : {experience.train()}")
        print(f"type(experience.train()) : {type(experience.train())}")

        print("Start of experience ", experience.current_experience)

        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))


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


# python benchmark_cifar.py |& tee log/benchmark0