import sys, os
sys.path.append(os.path.abspath('src'))

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from utils import get_model_instance


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patch_size = 256


def compute_flops(model: torch.nn.Module, inp: torch.Tensor):
    # warm-up (important for CUDA)
    with torch.no_grad():
        for _ in range(5):
            _ = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(inp)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

    # show top operators (flops column available when with_flops=True)
    # print(prof.key_averages().table(sort_by="flops", row_limit=20))

    # compute total FLOPs from recorded events
    total_flops = sum(getattr(e, "flops", 0) for e in prof.key_averages())
    print("Total FLOPs:", total_flops)


if __name__ == '__main__':
    model = get_model_instance('denoising', 'gaussian', 'REDNet', device, True, 50)
    inp = torch.randn(1, 1, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()

    model = get_model_instance('denoising', 'gaussian', 'DnCNN', device, True, 50)
    inp = torch.randn(1, 1, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('denoising', 'gaussian', 'DnCNN', device, True)
    inp = torch.randn(1, 1, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('denoising', 'gaussian', 'DnCNN', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()

    model = get_model_instance('denoising', 'gaussian', 'Restormer', device, True)
    inp = torch.randn(1, 1, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('denoising', 'gaussian', 'Restormer', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('deblurring', 'defocus', 'Restormer', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('deblurring', 'defocus', 'Restormer (Dual-pixel)', device)
    inp = torch.randn(1, 6, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()

    model = get_model_instance('denoising', 'gaussian', 'MaIR', device, sigma=50)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('denoising', 'real', 'MaIR', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()

    model = get_model_instance('deblurring', 'motion', 'DeblurGANv2 (Inception)', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
    model = get_model_instance('deblurring', 'motion', 'DeblurGANv2 (MobileNet)', device)
    inp = torch.randn(1, 3, patch_size, patch_size, device=device)
    compute_flops(model, inp)
    del model; torch.cuda.empty_cache()
