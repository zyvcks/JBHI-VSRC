import torch.optim as optim

from .vnet import VNet
from .vnet_sdf import sdf_VNet


model_list = ['VNET', 'sdf_VNet']


def create_model(args, ema=False):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.base_lr
    in_channels = args.inChannels
    num_class = args.nclass
    weight_decay = args.weight_decay

    if model_name == 'VNET':
        model = VNet(in_channels=in_channels, nclass=num_class)
    elif model_name == 'sdf_VNet':
        model = sdf_VNet(n_channels=in_channels, n_classes=num_class, normalization='batchnorm', has_dropout=ema)

    else:
        raise Exception("No Model Error")

    if ema:
        print("======Building EMA Model. . . . . . . " + model_name + "======")
        for param in model.parameters():
            param.detach_()

        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("No Optimizer Error")

        return model, optimizer
    else:
        print("======Building Model. . . . . . . " + model_name + "======")
        print(model_name, '--Number of params:{}'.format(
            sum([p.data.nelement() for p in model.parameters()])
        ))

        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("No Optimizer Error")
        return model, optimizer
