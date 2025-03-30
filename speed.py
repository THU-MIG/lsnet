import torch
import time
from timm import create_model
import model.build
import utils
from argparse import ArgumentParser

torch.autograd.set_grad_enabled(False)

T0 = 5
T1 = 10

def compute_throughput(model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    while time.time() - start < T0:
        model(inputs)
    timing = []
    torch.cuda.synchronize()
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="lsnet_t", type=str)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("--device", default=0, type=int)
    
    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    resolution = args.resolution
    device = args.device
    torch.cuda.set_device(device)
    
    torch.cuda.empty_cache()
    model = create_model(model, num_classes=1000)
    utils.replace_batchnorm(model)
    model.to(device)
    model.eval()
    compute_throughput(model, device,
                        batch_size, resolution=resolution)
    