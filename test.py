import torch

print("PyTorch version:", torch.__version__)
cuda_ok = torch.cuda.is_available()
print("CUDA available?", cuda_ok)
if cuda_ok:
    print(" Number of GPUs:", torch.cuda.device_count())
    print(" Device 0 name:", torch.cuda.get_device_name(0))
    x = torch.randn(2,2).to('cuda')
    print(" Tensor moved to:", x.device)
else:
    print(" No CUDA detected. Check drivers/CUDA install.")