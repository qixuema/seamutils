import torch
from torch.nn import Module

def fmt(v):
    if torch.is_tensor(v) and v.dim() == 0:
        v = v.item()
    if isinstance(v, (float, int)):
        return f"{v:.3f}"
    else:
        return str(v)

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def split_sequence_and_generate_mask(batch_sequence, threshold):
    if batch_sequence.shape[-1] == 0:
        return torch.zeros_like(batch_sequence, device=batch_sequence.device)
    
    split_mask = batch_sequence >= threshold
    
    segment_flags = torch.cumsum(split_mask, dim=1)
    
    segment_flags = segment_flags - segment_flags[:, 0].unsqueeze(1)
    
    return segment_flags