import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader, grad_clip_thresh=None, lr_scheduler=None):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader) * data_loader.batch_expand_size
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
        self.grad_clip_thresh = grad_clip_thresh
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('total_loss', 'mel_loss', 'duration_loss', 'grad_norm')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        batch_idx = 0
        for batchs in tqdm(self.data_loader):
            for db in tqdm(batchs):
                batch_idx += 1

                # load batch data
                character = db["text"].long().to(self.device)
                mel_target = db["mel_target"].float().to(self.device)
                duration = db["duration"].int().to(self.device)
                mel_pos = db["mel_pos"].long().to(self.device)
                src_pos = db["src_pos"].long().to(self.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output = self.model(
                    character,
                    src_pos,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration
                )

                # Calculate loss
                mel_loss, duration_loss = self.criterion(
                    mel_output,
                    duration_predictor_output,
                    mel_target,
                    duration
                )
                total_loss = mel_loss + duration_loss

                self.train_metrics.update('total_loss', total_loss.item())
                self.train_metrics.update('mel_loss', mel_loss.item())
                self.train_metrics.update('duration_loss', duration_loss.item())
                self.train_metrics.update("grad_norm", self.get_grad_norm())


                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                if self.grad_clip_thresh:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_thresh)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                if batch_idx % self.log_step == 0:
                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                        if self.lr_scheduler is not None:
                            self.writer.add_scalar(
                                "learning_rate", self.lr_scheduler.get_last_lr()[0]
                            )
                        for metric_name in self.train_metrics.keys():
                            self.writer.add_scalar(f"{metric_name}", self.train_metrics.avg(metric_name))
                    self.logger.debug('Train Epoch: {} {} Total Loss: {:.2f} Mel Loss: {:.2f} Duration Loss: {:.2f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        total_loss.item(),
                        mel_loss.item(),
                        duration_loss.item()
                        ))

                if batch_idx == self.len_epoch:
                    break
        log = self.train_metrics.result()
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} steps ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
