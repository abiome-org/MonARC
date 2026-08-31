import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", "/tmp/monarc-torch-offline")
os.environ.setdefault("MONARC_DINO_ALLOW_DOWNLOAD", "0")


def pytest_configure(config):
    try:
        import torch

        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    except ImportError:
        pass
