import argparse
import collections
import torch
import numpy as np
import dataloader as module_data
import model as module_model
from trainer import Trainer
from utils import prepare_device
from utils.parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    logger.info("Loading dataset")
    dataset = config.init_obj('dataset', module_data)
    data_loader = config.init_obj('dataloader', module_data, dataset=dataset)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f"Trainable parameters: {sum([np.prod(p.size()) for p in trainable_params])}")

    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) if config["lr_scheduler"] else None

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      lr_scheduler=lr_scheduler,
                      validation_text=config["trainer"]["validation_text"]
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
