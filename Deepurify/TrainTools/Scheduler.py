import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, total_train_epoch):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_train_epoch, eta_min=0, last_epoch=-1)
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if not self.finished or not self.after_scheduler:
            return super(GradualWarmupScheduler, self).step(epoch)
        if epoch is None:
            self.after_scheduler.step(None)
        else:
            self.after_scheduler.step(epoch - self.total_epoch)
