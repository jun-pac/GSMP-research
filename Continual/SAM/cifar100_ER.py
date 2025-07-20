
# Memory constrainted ER

from typing import Callable, Optional, List, Union
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator


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
from sam import SAM

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, GEMPlugin, LwFPlugin

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


class SAM_Naive(Naive):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
    
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch() # Return None type
            # for i in range(len(self.mbatch)):
            #     self.mbatch[i] = self.mbatch[i].to(self.device, non_blocking=True)
            self._before_training_iteration(**kwargs)
            self.loss = self._make_empty_loss()
            
            input=self.mbatch[0]
            # output=torch.nn.functional.one_hot(self.mbatch[1],100).type(torch.float32)

            
            # first forward-backward pass
            self._before_forward(**kwargs)
            self.mb_output=self.model(input)
        
            self._after_forward(**kwargs)
            loss=self.criterion()  # use this loss for any training statistics
            self.loss+=loss

            self._before_backward(**kwargs)
            loss.backward()
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.optimizer.first_step(zero_grad=True)
            self._after_update(**kwargs)

            # second forward-backward pass
            self.mb_output=self.model(input)
            loss=self.criterion()
            loss.backward()  # make sure to do a full forward pass
            self.optimizer.second_step(zero_grad=True)
            
            self._after_training_iteration(**kwargs)
    # def training_epoch(self, **kwargs):
    #     """Training epoch.

    #     :param kwargs:
    #     :return:
    #     """
    #     for self.mbatch in self.dataloader:
    #         if self._stop_training:
    #             break

    #         self._unpack_minibatch()
    #         self._before_training_iteration(**kwargs)

    #         self.optimizer.zero_grad()
    #         self.loss = self._make_empty_loss()
    #         input=self.mbatch[0]
    #         output=torch.nn.functional.one_hot(self.mbatch[1],100).type(torch.float32)

    #         # Forward
    #         self._before_forward(**kwargs)
    #         #self.mb_output=self.model(input)
    #         self.mb_output = self.forward()
    #         self._after_forward(**kwargs)

    #         # Loss & Backward
    #         # self.loss += self._criterion(output, self.mb_output)
    #         # print(f"self._criterion(output, self.mb_output) : {self._criterion(output, self.mb_output)}")
    #         print(f"self.criterion() : {self.criterion()}")
    #         self.loss+=self.criterion()
    #         self._before_backward(**kwargs)
    #         self.backward()
    #         self._after_backward(**kwargs)

    #         # Optimization step
    #         self._before_update(**kwargs)
    #         self.optimizer.step()
    #         self._after_update(**kwargs)

    #         self._after_training_iteration(**kwargs)


        

def main(args):
    # Model getter: specify dataset and depth of the network.
    model = Wide_ResNet(28, 10, 0.3, 100) # 28X10

    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    benchmark = SplitCIFAR100(
        n_experiences=6,
        first_exp_with_half_classes=True,
        fixed_class_order=[99-i for i in range(100)]
    )
    # benchmark = SplitCIFAR100(
    #     n_experiences=50,
    #     fixed_class_order=[99-i for i in range(100)]
    # )

    # optimizer
    base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    # optimizer=torch.optim.SGD(model.parameters(), lr=0.01)

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

    cl_strategy = SAM_Naive( 
        model,
        optimizer,
        CrossEntropyLoss(),
        train_mb_size=50,
        train_epochs=10,
        eval_mb_size=50,
        device=device,
        plugins=[ReplayPlugin(mem_size=1000,batch_size=50)],
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
# python cifar100_ER.py | tee log/ER

