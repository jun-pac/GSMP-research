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

# Maintain mini-batch proportial to the number of experiences

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
from avalanche.benchmarks.classic import SplitCIFAR10

import torch.optim.lr_scheduler
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive
from avalanche.training import Cumulative
from avalanche.training.plugins import EarlyStoppingPlugin

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
    # cifar_train = CIFAR10(
    #     root=expanduser("~") + "/.avalanche/data/cifar10/",
    #     train=True,
    #     download=True,
    #     transform=transform,
    # )
    # cifar_test = CIFAR10(
    #     root=expanduser("~") + "/.avalanche/data/cifar10/",
    #     train=False,
    #     download=True,
    #     transform=transform,
    # )

    # benchmark = nc_benchmark( 
    #     cifar_train,
    #     cifar_test,
    #     5,
    #     task_labels=False,
    #     seed=1234,
    #     fixed_class_order=[i for i in range(10)],
    # )

    benchmark = SplitCIFAR10(
        n_experiences=6,
        first_exp_with_half_classes=True,
        fixed_class_order=[9-i for i in range(10)]
    )
    
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    cl_strategy = Naive(  
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=10,
        eval_mb_size=100,
        device=device,
        plugins=[ReplayPlugin(mem_size=100000,batch_size=20),EarlyStoppingPlugin(patience=1,val_stream_name="train_stream")],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    cnt=0
    for experience in benchmark.train_stream:
        cnt+=1
        print(f"type(experience): {type(experience)}, num: {cnt}")
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


# python cifar10_inclusion.py | tee log/inclusions
# 0.4121 0.3979 0.3781 0.3807 -> 0.3922

'''
> Eval on experience 0 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp000 = 0.0934
        Loss_Exp/eval_phase/test_stream/Task000/Exp000 = 1.5657
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000 = 0.4222
-- Starting eval on experience 1 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 23.44it/s]
> Eval on experience 1 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp001 = 0.0790
        Loss_Exp/eval_phase/test_stream/Task000/Exp001 = 1.6355
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001 = 0.4370
-- Starting eval on experience 2 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 27.77it/s]
> Eval on experience 2 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp002 = -0.0120
        Loss_Exp/eval_phase/test_stream/Task000/Exp002 = 2.0077
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002 = 0.2050
-- Starting eval on experience 3 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 27.83it/s]
> Eval on experience 3 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp003 = 0.1180
        Loss_Exp/eval_phase/test_stream/Task000/Exp003 = 1.7741
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003 = 0.2190
-- Starting eval on experience 4 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 29.30it/s]
> Eval on experience 4 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp004 = 0.1090
        Loss_Exp/eval_phase/test_stream/Task000/Exp004 = 1.2703
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004 = 0.5220
-- Starting eval on experience 5 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 30.22it/s]
> Eval on experience 5 (Task 0) from test stream ended.
        Loss_Exp/eval_phase/test_stream/Task000/Exp005 = 1.0596
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp005 = 0.6270
-- >> End of eval phase << --
        Loss_Stream/eval_phase/test_stream/Task000 = 1.5575
        Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.4121
'''