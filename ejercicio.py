# cuantización absmax ingenua 8 bits

import torch

def absmax_quantize(X):

    # Calculate scale

    scale = 127 / torch.max(torch.abs(X))

    # Quantize

    X_quant = (scale * X).round()

    # Dequantize

    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant

# cuantizacion de punto cero

def zeropoint_quantize(X):

    # Calculate value range (denominator)

    x_range = torch.max(X) - torch.min(X)

    x_range = 1 if x_range == 0 else x_range

    # Calculate scale

    scale = 255 / x_range

    # Shift by zero-point

    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs

    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize

    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

# Set device to CPU for now

device = 'cpu'

# Load model and tokenizer

model_id = 'gpt2'

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print model size

print(f"Model size: {model.get_memory_footprint():,} bytes")