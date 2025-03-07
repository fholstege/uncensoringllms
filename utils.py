import random
import numpy as np
import torch
import os
from transformers import TextStreamer


def str_to_bool(s):
    """Convert string to boolean."""
    lower = s.lower()
    if lower == "true":
        return True
    else:
        return False

def set_seed(seed=42):
    """
    Set seed for reproducibility across all random processes.
    
    Args:
        seed (int): Seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional deterministic settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducibility of dictionaries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducible results")


def get_model_layers(model):
    """Helper function to get model layers based on architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA, Mistral, etc.
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # Qwen, GPT-2, etc.
        return model.transformer.h
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        # Some other architectures
        return model.transformer.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        # GPT-NeoX based models
        return model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        # Some encoder-decoder architectures
        return model.model.decoder.layers
    else:
        # Try to find layers using more general pattern matching
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, "layers"):
                return attr.layers
            if hasattr(attr, "h") and isinstance(getattr(attr, "h"), torch.nn.ModuleList):
                return attr.h
        
        raise AttributeError(f"Could not find layers in model of type {type(model).__name__}. "
                            "Please specify the correct attribute path to access layers.")


class FileTextStreamer(TextStreamer):
    """Extension of TextStreamer that writes to both console and file."""
    def __init__(self, tokenizer, file_handle, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.file_handle = file_handle
        self.current_text = ""
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end)
        self.current_text += text
        self.file_handle.write(text)
        self.file_handle.flush()