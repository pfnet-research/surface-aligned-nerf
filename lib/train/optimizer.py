import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net, lr=None, weight_decay=None):
    params = []
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    params_smpl = []
    lr_smpl = 1e-5

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        # params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if 'leanable_smpl' in key:
            params_smpl += [{"params": [value], "lr": lr_smpl, "weight_decay": weight_decay}]
            # params_smpl += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            continue

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # print(len(params_smpl))
    # print(len(params))
    optimizer = _optimizer_factory[cfg.train.optim](params + params_smpl)

    # if 'adam' in cfg.train.optim:
    #     optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    # else:
    #     optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer