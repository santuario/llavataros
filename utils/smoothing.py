import torch
from scipy.signal import savgol_filter

def smooth_output(raw_output, window_length=9, poly_order=2):
    smoothed = savgol_filter(raw_output.cpu().numpy(), window_length=window_length, polyorder=poly_order, axis=0)
    return torch.tensor(smoothed, device=raw_output.device)