import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
