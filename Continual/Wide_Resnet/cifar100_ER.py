
# Memory constrainted ER


from os.path import expanduser

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
from networks import *

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
    model = Wide_ResNet(28, 20, 0.3, 100) # 28X10

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
        train_epochs=50,
        eval_mb_size=50,
        device=device,
        plugins=[ReplayPlugin(mem_size=20000,batch_size=50)],
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    cnt=0
    for experience in benchmark.train_stream:
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
# python cifar100_ER.py | tee log/ER_2000
# python cifar100_ER.py | tee log/ER_5000
# python cifar100_ER.py | tee log/ER_10000
# python cifar100_ER.py | tee log/ER_20000

