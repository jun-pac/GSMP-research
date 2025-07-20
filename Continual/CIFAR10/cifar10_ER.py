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

# Memory constrainted ER

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
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, EarlyStoppingPlugin, GEMPlugin, LwFPlugin

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

    # CREATE THE STRATEGY INSTANCE (Naive, with Replay) # Maybe continual learning strategy (including model, optimizer)
    early_stop = EarlyStoppingPlugin(patience=3,val_stream_name="Acc")
    ewc = EWCPlugin(ewc_lambda=0.001)
    LwF = LwFPlugin(alpha=1, temperature=2)
    replay = ReplayPlugin(mem_size=1000)
    GEM = GEMPlugin(patterns_per_experience=200, memory_strength=0.1)

    cl_strategy = Naive( # continual? or curriculam? 
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=10,
        eval_mb_size=100,
        device=device,
        plugins=[LwF,replay],
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


# python cifar10_ER.py | tee log/ER


# ewc + replay(1000) : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.2086, 0.2501, 0.2386, 0.2346
# ewc : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.1584
# No plugin : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.1591
# replay(1000) [1] : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.2235, 0.2275
# replay(10000) [2] : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.2442, 0.2333
# replay(100000) [3] : Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.2469, 0.2394

# LwF + replay(1000) [2] : 0.2269, 0.2362
# LwF + replay(10000) [3][4] : 0.1952
# LwF [4] : 0.1619, 0.1623, 0.1625, 0.1631

# GEM(10,0.8) [2] : 
# GEM(200,0.1) [3] :
# GEM(200,0.5) [3] : 
# GEM(1000,0.2) [3] : 
# GEM(200,0.8) [2] :
# GEM(200,0.5) + replay(1000) [4] :  

#==================================================================================================
# When first_exp_with_half_classes=True
# Naive finetuning w/o any replay: 0.1000, 0.1000, 0.1000, 0.1000 -> 0.1
'''
> Eval on experience 0 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp000 = 0.5154
        Loss_Exp/eval_phase/test_stream/Task000/Exp000 = 5.7428
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000 = 0.0000
-- Starting eval on experience 1 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 18.41it/s]
> Eval on experience 1 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp001 = 1.0000
        Loss_Exp/eval_phase/test_stream/Task000/Exp001 = 5.3123
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001 = 0.0000
-- Starting eval on experience 2 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 19.74it/s]
> Eval on experience 2 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp002 = 1.0000
        Loss_Exp/eval_phase/test_stream/Task000/Exp002 = 5.5393
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002 = 0.0000
-- Starting eval on experience 3 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 19.29it/s]
> Eval on experience 3 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp003 = 1.0000
        Loss_Exp/eval_phase/test_stream/Task000/Exp003 = 4.9022
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003 = 0.0000
-- Starting eval on experience 4 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 17.71it/s]
> Eval on experience 4 (Task 0) from test stream ended.
        ExperienceForgetting/eval_phase/test_stream/Task000/Exp004 = 1.0000
        Loss_Exp/eval_phase/test_stream/Task000/Exp004 = 4.8596
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp004 = 0.0000
-- Starting eval on experience 5 (Task 0) from test stream --
100%|??????????| 10/10 [00:00<00:00, 18.26it/s]
> Eval on experience 5 (Task 0) from test stream ended.
        Loss_Exp/eval_phase/test_stream/Task000/Exp005 = 0.0380
        Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp005 = 1.0000
-- >> End of eval phase << --
        Loss_Stream/eval_phase/test_stream/Task000 = 4.9365
        Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.1000
'''
# ewc + replay(1000): Top1_Acc_Stream/eval_phase/test_stream/Task000 = 0.1624, 0.1784, 0.1732, 0.1714 -> 0.17135

