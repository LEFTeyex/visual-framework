r"""
For training model.
Consist of some Trainers.
"""

import argparse
import torch.nn as nn

from pathlib import Path
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from utils.log import add_log_file
from metaclass.metatrainer import MetaTrainClassify
from utils.general import timer, load_all_yaml, save_all_yaml, init_seed, select_one_device

from val_classify import ValClassify

from mine.SmartNet.smartnet import SmartNet

r"""Set Global Constant for file save and load"""
ROOT = Path.cwd()  # **/visual-framework root directory


class TrainClassify(MetaTrainClassify):
    def __init__(self, args):
        super(TrainClassify, self).__init__(args)

        # Get path_dict
        self.path_dict = self.get_save_path(('hyp', 'hyp.yaml'),
                                            ('args', 'args.yaml'),
                                            ('logger', 'logger.log'),
                                            ('writer', 'tensorboard'),
                                            ('last', 'weights/last.pt'),
                                            ('best', 'weights/best.pt'),
                                            ('datasets', 'datasets.yaml'))

        # Add FileHandler for logger
        add_log_file(self.path_dict['logger'])

        # Set tensorboard writer
        self.writer = self.set_tensorboard_writer(self.path_dict['writer'])

        # Set one device
        self.device = select_one_device(self.device)  # requires model, images, labels .to(self.device)
        self.cuda = (self.device != 'cpu')

        # Load hyp yaml
        self.hyp = load_all_yaml(self.hyp)

        # Initialize or auto seed manual and save in self.hyp
        self.hyp['seed'] = init_seed(self.hyp['seed'])

        # Get datasets path dict
        self.datasets = load_all_yaml(self.datasets)

        # Save yaml dict
        save_all_yaml((vars(args), self.path_dict['args']),
                      (self.hyp, self.path_dict['hyp']),
                      (self.datasets, self.path_dict['datasets']))
        args = self.empty()

        # TODO auto compute anchors when anchors is None in self.datasets

        # Load checkpoint
        self.checkpoint = self.load_checkpoint(self.weights)

        # TODO upgrade DP DDP

        # Initialize or load model
        self.model = self.load_model(SmartNet(self.inc, self.datasets['nc'], self.image_size, self.channels,
                                              invalid=0.01, num_add=5, add_cut_percentage=0.9,
                                              act='relu', device=self.device), load=self._load_model)

        # Unfreeze model
        self.unfreeze_model()

        # Freeze layers of model
        self.freeze_layers(self.freeze_names)

        # Set parameter groups list to for the optimizer
        self.param_groups = self.set_param_groups((('weight', nn.Parameter, {'weight_decay': self.hyp['weight_decay']}),
                                                   ))

        # Initialize and load optimizer
        self.optimizer = self.load_optimizer(SGD(self.param_groups,
                                                 lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True),
                                             load=self._load_optimizer)
        self.param_groups = self.empty()

        # Initialize and load lr_scheduler
        self.lr_scheduler = self.load_lr_scheduler(StepLR(self.optimizer, 20), load=self._load_lr_scheduler)
        # TODO set lr_scheduler args to self.hyp

        # Initialize and load GradScaler
        self.scaler = self.load_gradscaler(GradScaler(enabled=self.cuda), load=self._load_gradscaler)

        # Initialize or load start_epoch
        self.start_epoch = self.load_start_epoch(load=self._load_start_epoch)

        # Initialize or load best_fitness
        self.best_fitness = self.load_best_fitness(load=self._load_best_fitness)

        # Empty self.checkpoint when load finished
        self.checkpoint = self.empty()

        # Get dataloader for training testing
        transform = transforms.Compose([transforms.ToTensor()])

        self.train_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['train'], transform),
                                                    shuffle=self.shuffle)

        if self.datasets['test'] is not None:
            self.val_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['val'], transform))
            self.test_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['test'], transform))
        else:
            self.val_dataloader = self.set_dataloader(MNIST(self.datasets['path'], self.datasets['val'], transform))
            self.test_dataloader = None

        # TODO upgrade warmup

        # Get loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Set val class
        self.val_class = ValClassify


def parse_args_classify(known: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True, help='')
    parser.add_argument('--visual_image', type=bool, default=False,
                        help='whether make images visual in tensorboard')
    parser.add_argument('--visual_graph', type=bool, default=False,
                        help='whether make model graph visual in tensorboard')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--freeze_names', type=list, default=[],
                        help='save_name of freezing layers in model')
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda:0 or 0')
    parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--pin_memory', type=bool, default=False, help='')
    parser.add_argument('--datasets', type=str, default=str(ROOT / 'mine/data/datasets/classification/MNIST.yaml'),
                        help='')
    parser.add_argument('--save_name', type=str, default='exp', help='')
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'runs/train/classify'), help='')
    parser.add_argument('--hyp', type=str, default=str(ROOT / 'data/hyp/hyp_classify_train.yaml'), help='')

    parser.add_argument('--inc', type=int, default=1, help='')
    parser.add_argument('--image_size', type=int, default=28, help='')
    parser.add_argument('--channels', type=list, default=[512, 256, 128, 64], help='')
    parser.add_argument('--load_model', type=str, default=None, help='')
    parser.add_argument('--load_optimizer', type=bool, default=False, help='')
    parser.add_argument('--load_lr_scheduler', type=bool, default=False, help='')
    parser.add_argument('--load_gradscaler', type=bool, default=False, help='')
    parser.add_argument('--load_start_epoch', type=str, default=None, help='')
    parser.add_argument('--load_best_fitness', type=bool, default=False, help='')
    namespace = parser.parse_known_args()[0] if known else parser.parse_args()
    return namespace


@timer
def train_classify():
    arguments = parse_args_classify()
    trainer = TrainClassify(arguments)
    trainer.train()


if __name__ == '__main__':
    train_classify()
