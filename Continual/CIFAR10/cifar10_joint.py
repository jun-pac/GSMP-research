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

# Naive cumulative training. Upper-bound of performances


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
from avalanche.training import JointTraining, Cumulative
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins import EarlyStoppingPlugin

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    model = pytorchcv_wrapper.resnet("cifar10", depth=20, pretrained=False)

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

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    cl_strategy = Cumulative( 
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=10,
        eval_mb_size=100,
        device=device,
        plugins=[EarlyStoppingPlugin(patience=1,val_stream_name="train_stream")],
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


# python cifar10_joint.py | tee log/joint 
# Result(Top1_Acc_Stream/eval_phase) 0.4479, 0.4481, 0.4443, 0.4261 -> 0.4416

'''
> Eval on experience 0 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp000 = -0.1054
        Loss_Exp/eval_phase/test_stream/Task000/Exp000 = 1.2797
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000 = 0.6008
-- Starting eval on experience 1 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 15.65it/s]
> Eval on experience 1 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp001 = 0.4200
        Loss_Exp/eval_phase/test_stream/Task000/Exp001 = 1.8879
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001 = 0.2970
-- Starting eval on experience 2 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 16.40it/s]
> Eval on experience 2 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp002 = -0.1150
        Loss_Exp/eval_phase/test_stream/Task000/Exp002 = 1.9524
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002 = 0.1210
-- Starting eval on experience 3 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 17.91it/s]
> Eval on experience 3 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp003 = -0.0520
        Loss_Exp/eval_phase/test_stream/Task000/Exp003 = 2.0676
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003 = 0.1420
-- Starting eval on experience 4 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 14.73it/s]
> Eval on experience 4 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp004 = -0.0680
        Loss_Exp/eval_phase/test_stream/Task000/Exp004 = 1.1114
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004 = 0.5330
-- Starting eval on experience 5 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 15.45it/s]
> Eval on experience 5 (Task 0) from test stream ended.
        Loss_Exp/eval_phase/test_stream/Task000/Exp005 = 1.4457
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp005 = 0.3820
-- >> End of eval phase << --
        Loss_Stream/eval_phase/test_stream/Task000 = 1.4863
        Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.4479
'''