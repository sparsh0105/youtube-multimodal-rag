# for gpu detection and configuration 

import torch

def get_device():
    """
    Returns 'cuda' if GPU is available, else 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_info():
    """
    Optional helper for debugging
    """
    if torch.cuda.is_available():
        print("✅ GPU available")
        print(torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU not available, using CPU")
