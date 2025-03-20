import torch
import torch.nn as nn
import torchaudio
from pase.pase.models.frontend import wf_builder


class PASEModel(nn.Module):
    def __init__(self, config_path, checkpoint_path, device=None):
        super().__init__()
        
        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = wf_builder(config_path).eval().to(self.device)
        self.model.load_pretrained(checkpoint_path, load_last=True, verbose=True)
        
        # Audio and frame properties
        self.sample_rate = 16000
        self.fps = 30
        self.frame_length = int(self.sample_rate * 0.033)  # 33ms for 30 fps

    def forward(self, waveform):
        """
        Forward pass through PASE model.
        Ensures consistent device usage.
        """
        waveform = waveform.to(self.device)
        return self.model(waveform)

    def extract_features(self, waveform):
        """
        Extracts PASE features and aligns them to 30 fps.
        
        Args:
            waveform (torch.Tensor): Input waveform tensor of shape [1, 1, num_samples]

        Returns:
            torch.Tensor: Aligned features of shape [expected_frames, 256]
        """
        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)

        # Resample waveform if the sample rate is different
        waveform_rate = waveform.shape[-1]
        if waveform_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=waveform_rate, new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform)

        # Extract features
        with torch.no_grad():
            features = self.forward(waveform)  # Shape: (1, 256, num_frames)

        # Align features to 30 fps
        num_frames = features.shape[-1]
        expected_frames = int((waveform.shape[-1] / self.sample_rate) * self.fps)

        if num_frames != expected_frames:
            # Only interpolate if there's a frame mismatch
            features = torch.nn.functional.interpolate(
                features.unsqueeze(0),
                size=expected_frames,
                mode='linear',
                align_corners=False
            ).squeeze(0)

        # Reshape to [expected_frames, 256]
        return features.transpose(1, 2).contiguous()

    @staticmethod
    def load_wav(filepath, target_sample_rate=16000, device=None):
        """
        Loads a WAV file and resamples it if needed.

        Args:
            filepath (str): Path to the WAV file
            target_sample_rate (int): Desired sample rate
            device (torch.device): Target device

        Returns:
            torch.Tensor: Waveform tensor [1, 1, num_samples]
        """
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            ).to(device)
            waveform = resampler(waveform)
        
        return waveform.to(device)