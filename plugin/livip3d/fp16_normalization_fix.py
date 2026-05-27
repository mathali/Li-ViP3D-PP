import torch
import torch.nn as nn

# Monkey-patch LayerNorm and BatchNorm to always run in FP32
original_layer_norm_forward = nn.LayerNorm.forward
original_batch_norm1d_forward = nn.BatchNorm1d.forward  
original_batch_norm2d_forward = nn.BatchNorm2d.forward

def fp32_layer_norm_forward(self, x):
    """Force LayerNorm to run in FP32 for numerical stability"""
    dtype = x.dtype
    if dtype == torch.float16:
        x = x.float()
        out = original_layer_norm_forward(self, x)
        return out.to(torch.float16)
    return original_layer_norm_forward(self, x)

def fp32_batch_norm1d_forward(self, x):
    """Force BatchNorm1d to run in FP32 for numerical stability"""
    dtype = x.dtype
    if dtype == torch.float16:
        x = x.float()
        out = original_batch_norm1d_forward(self, x)
        return out.to(torch.float16)
    return original_batch_norm1d_forward(self, x)

def fp32_batch_norm2d_forward(self, x):
    """Force BatchNorm2d to run in FP32 for numerical stability"""
    dtype = x.dtype
    if dtype == torch.float16:
        x = x.float()
        out = original_batch_norm2d_forward(self, x)
        return out.to(torch.float16)
    return original_batch_norm2d_forward(self, x)

# Apply the monkey patches
nn.LayerNorm.forward = fp32_layer_norm_forward
nn.BatchNorm1d.forward = fp32_batch_norm1d_forward
nn.BatchNorm2d.forward = fp32_batch_norm2d_forward

print("Patched LayerNorm and BatchNorm to always use FP32 for numerical stability with dynamic loss scaling")


