import settings
import torch
import torchvision
from loader.ResNet_Attention import resnet18

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        # load pytorch ResNet
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # state_dict = torch.load('zoo/cifar10_90_net.pth')
        # new_state_dict = model.state_dict()
        # new_state_dict.update(state_dict)
        # model.load_state_dict(new_state_dict)
        
        # load our own model
        model = resnet18(pretrained=True, progress=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
