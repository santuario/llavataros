import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from utils.transcriber import WhisperTranscriber
from utils.aligner import MontrealForcedAligner

class GestureDataset(Dataset):
    def __init__(self, dataset_dir, alignments_dir):
        self.dataset_dir = dataset_dir
        
        # Step 1: Transcribe audio files
        transcriber = WhisperTranscriber(model_size="base")
        self.transcript_map = transcriber.transcribe_dataset(dataset_dir)
        
        # Step 2: Align transcripts with audio
        self.aligner = MontrealForcedAligner(dataset_dir, alignments_dir, self.transcript_map)
        self.aligner.align()
        
        # Step 3: Load dataset files
        self.files = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        npy_file = os.path.join(self.dataset_dir, self.files[idx])
        wav_file = npy_file.replace('.npy', '.wav')
        
        # Load motion data
        motion = np.load(npy_file)  # Shape: (num_frames, num_points, 7)
        motion = torch.tensor(motion, dtype=torch.float32)  # [px, py, pz, qx, qy, qz, qw]
        motion = motion.view(motion.shape[0], -1)  # Flatten to (num_frames, num_points * 7)
        
        # Load audio
        waveform, _ = torchaudio.load(wav_file)
        
        # Load transcript
        transcript_path = self.transcript_map[wav_file]
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
        
        # Get word timings
        word_timings = self.aligner.get_word_timings(wav_file)
        
        # Speaker ID (simplified)
        speaker_id = torch.tensor([0], dtype=torch.long)
        
        return waveform, transcript, word_timings, motion, speaker_id

def collate_fn(batch):
    waveforms, transcripts, word_timings_list, motions, speaker_ids = zip(*batch)
    
    waveform_lengths = torch.tensor([w.shape[1] for w in waveforms], dtype=torch.long)
    max_waveform_length = max(waveform_lengths)
    padded_waveforms = torch.zeros(len(waveforms), 1, max_waveform_length)
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :, :waveform.shape[1]] = waveform
    
    motion_lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)
    max_motion_length = max(motion_lengths)
    motion_feature_dim = motions[0].shape[1]
    padded_motions = torch.zeros(len(motions), max_motion_length, motion_feature_dim)
    for i, motion in enumerate(motions):
        padded_motions[i, :motion.shape[0], :] = motion
    
    speaker_ids = torch.stack(speaker_ids)
    
    return padded_waveforms, transcripts, word_timings_list, padded_motions, speaker_ids, waveform_lengths, motion_lengths

def get_data_loader(dataset_dir, alignments_dir, batch_size):
    dataset = GestureDataset(dataset_dir, alignments_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)