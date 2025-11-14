import torch

print(f"PyTorch version: {torch.__version__}") # Should NOT have +cpu anymore
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}") # Should now be True

if cuda_available:
    print(f"CUDA version PyTorch built with: {torch.version.cuda}") # Should show 12.1
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("-> CUDA still not available. Check installation messages and drivers.")