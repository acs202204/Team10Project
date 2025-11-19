import torch
import os

if os.name == 'nt':  # For Windows
    os.system('cls')
else:  # For Unix-like systems (Linux, macOS)
    os.system('clear')

print("Torch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

else:
    print("CUDA not available")
