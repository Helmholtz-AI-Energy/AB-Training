import torchvision.models as models

from . import components
from .qr_fixing import QRFixingModel


def get_tv_model(pretrained, name):
    if pretrained:
        print(f"=> using pre-trained model '{name}'")
        model = models.__dict__[name](pretrained=True)
    else:
        print(f"=> creating model '{name}'")
        model = models.__dict__[name]()
    return model
