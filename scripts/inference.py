import torch
import yaml
import numpy as np
import os
from models.pase_model import PASEModel
from models.llama_model import LLaMAModel
from models.transformer_xl import CrossAttentiveTransformerXL
from utils.smoothing import smooth_output
from utils.transcriber import WhisperTranscriber
from utils.aligner import MontrealForcedAligner
from utils.render_animation import render_3d_animation_with_audio
import torchaudio

class Inference:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.pase = PASEModel(config['paths']['pase_config'], config['paths']['pase_checkpoint']).to(self.device)
        self.llama = LLaMAModel(config['paths']['llama_model'], self.device)
        self.model = CrossAttentiveTransformerXL(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Initialize transcriber and aligner
        self.transcriber = WhisperTranscriber(model_size="base")
        self.aligner = MontrealForcedAligner(
            config['paths']['dataset_dir'],
            config['paths']['alignments_dir'],
            {}
        )
    
    def infer(self, audio_path, transcript=None, output_npy_path=None, output_mov_path=None):
        # Load audio
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        
        # Generate transcript if not provided
        if transcript is None:
            transcript_path = self.transcriber.transcribe(audio_path, self.config['paths']['dataset_dir'])
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        
        # Align transcript to get word timings
        transcript_map = {audio_path: transcript_path}
        self.aligner.transcript_map = transcript_map
        self.aligner.align()
        word_timings = self.aligner.get_word_timings(audio_path)
        
        # Extract features
        audio_feats = self.pase.extract_features(waveform)
        text_feats = self.llama.extract_features(transcript, word_timings, self.config['model']['fps'])
        
        # Speaker ID (simplified)
        speaker_ids = torch.tensor([0], dtype=torch.long).to(self.device)
        
        # Inference
        with torch.no_grad():
            pred, _ = self.model(audio_feats, text_feats, speaker_ids)
            smoothed_pred = smooth_output(pred, self.config['training']['window_length'], self.config['training']['poly_order'])
        
        # Reshape the prediction to (num_frames, num_points, 7)
        num_frames = smoothed_pred.shape[0]
        num_points = self.config['model']['num_joints']
        pred_reshaped = smoothed_pred.cpu().numpy().reshape(num_frames, num_points, -1)  # Shape: (num_frames, 9, 7)
        
        # Save the prediction as a .npy file
        if output_npy_path is None:
            output_npy_path = audio_path.replace('.wav', '_animation.npy')
        np.save(output_npy_path, pred_reshaped)
        print(f"Saved animation to {output_npy_path}")
        
        # Generate .mov preview if output_mov_path is provided
        if output_mov_path:
            class Args:
                point_path = output_npy_path
                wav_path = audio_path
                out_path = output_mov_path
                fps = self.config['model']['fps']
            
            args = Args()
            render_3d_animation_with_audio(args)
        
        return smoothed_pred

if __name__ == "__main__":
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    inference = Inference(config, "llavataros_model.pth")
    audio_path = "data/datasets/zaos1/001_Neutral_0_mirror_x1.0.wav"
    output_npy_path = "data/animations/001_Neutral_0_mirror_x1.0_animation.npy"
    output_mov_path = "data/animations/001_Neutral_0_mirror_x1.0_animation.mov"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    
    prediction = inference.infer(audio_path, output_npy_path=output_npy_path, output_mov_path=output_mov_path)
    print("Inference output shape:", prediction.shape)