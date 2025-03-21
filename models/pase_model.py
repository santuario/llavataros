import torch
import torch.nn as nn
import torchaudio
from pase.pase.models.frontend import wf_builder

class PASEModel(nn.Module):
    def __init__(self, config_path, checkpoint_path, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = wf_builder(config_path).eval().to(self.device)
        self.model.load_pretrained(checkpoint_path, load_last=True, verbose=True)
        self.sample_rate = 16000
        self.fps = 30
        self.frame_length = int(self.sample_rate * 0.033)

    def forward(self, waveform):
        waveform = waveform.to(self.device)
        return self.model(waveform)

    def extract_features(self, waveform, sample_rate, target_length=None):
        waveform = waveform.to(self.device)
        sample_rate = sample_rate.item() if sample_rate.dim() == 0 else sample_rate[0].item()  # Handle batched or scalar
        
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform)

        with torch.no_grad():
            features = self.forward(waveform)  # Shape: (batch_size, 256, num_frames)
        
        if target_length is None:
            target_length = int((waveform.shape[-1] / self.sample_rate) * self.fps)
        
        num_frames = features.shape[-1]
        if num_frames != target_length:
            features = torch.nn.functional.interpolate(
                features,
                size=target_length,
                mode='linear',
                align_corners=False
            )
        
        return features.transpose(1, 2).contiguous()  # Shape: (batch_size, target_length, 256)

    @staticmethod
    def load_wav(filepath, target_sample_rate=16000, device=None):
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.unsqueeze(0)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            ).to(device)
            waveform = resampler(waveform)
        return waveform.to(device)