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
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
from avalanche.models import pytorchcv_wrapper
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, GEMPlugin, LwFPlugin

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # Model getter: specify dataset and depth of the network.
    model = pytorchcv_wrapper.resnet("cifar100", depth=110, pretrained=False)

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    benchmark = SplitCIFAR100(
        n_experiences=6,
        first_exp_with_half_classes=True,
        fixed_class_order=[99-i for i in range(100)]
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
    ewc = EWCPlugin(ewc_lambda=0.001)
    LwF = LwFPlugin(alpha=1, temperature=2)
    replay = ReplayPlugin(mem_size=1000)
    GEM = GEMPlugin(patterns_per_experience=200, memory_strength=0.1)

    cl_strategy = Naive( 
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        CrossEntropyLoss(),
        train_mb_size=50,
        train_epochs=10,
        eval_mb_size=50,
        device=device,
        plugins=[replay],
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

# conda activate vv
# python cifar100_ER.py | tee log/ER


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
