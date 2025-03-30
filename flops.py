import torch
from timm import create_model
import model.build
import utils
from fvcore.nn import FlopCountAnalysis
from model.ska import SKA
import torch.nn.functional as F
from argparse import ArgumentParser

def forward_flops(self, x: torch.Tensor, w: torch.Tensor):
    w = w.squeeze(0)[..., 0, 0].reshape(-1, 1, 3, 3)
    w = w.repeat(8, 1, 1, 1)
    return F.conv2d(x, w, None, 1, 1, 1, groups=w.shape[0])

SKA.forward = forward_flops

torch.autograd.set_grad_enabled(False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="lsnet_t", type=str)
    parser.add_argument("--resolution", default=224, type=int)
    
    args = parser.parse_args()
    model = args.model
    resolution = args.resolution

    inputs = torch.randn(1, 3, resolution,
                            resolution)
    model = create_model(model, num_classes=1000)
    utils.replace_batchnorm(model)
    model.cuda()
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters / 1e6)
    flops = FlopCountAnalysis(model, inputs.cuda()).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    print("Flops: ", flops.total() / 1e9)
