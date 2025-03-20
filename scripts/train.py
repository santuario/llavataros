import torch
import yaml
from models.pase_model import PASEModel
from models.llama_model import LLaMAModel
from models.transformer_xl import CrossAttentiveTransformerXL
from utils.data_loader import get_data_loader
from utils.smoothing import smooth_output

class Trainer:
    def __init__(self, config, test_run=False):
        self.config = config
        self.test_run = test_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.pase = PASEModel(config['paths']['pase_config'], config['paths']['pase_checkpoint'], self.device)
        self.llama = LLaMAModel(config['paths']['llama_model'], self.device)
        self.model = CrossAttentiveTransformerXL(config, self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Data loader
        self.data_loader = get_data_loader(
            config['paths']['dataset_dir'],
            config['paths']['alignments_dir'],
            config['training']['batch_size'] if not test_run else 2
        )
    
    def loss_function(self, pred, target, lengths):
        max_len = target.size(1)
        mask = torch.arange(max_len, device=self.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(target)
        
        if pred.size(1) < target.size(1):
            pred = torch.nn.functional.pad(pred, (0, 0, 0, target.size(1) - pred.size(1)))
        elif pred.size(1) > target.size(1):
            pred = pred[:, :target.size(1), :]
        
        num_features = target.size(-1)
        pos_dim = 3
        rot_dim = num_features - pos_dim
        
        pos_loss = torch.mean(torch.abs(pred[:, :, :pos_dim] - target[:, :, :pos_dim]) * mask[:, :, :pos_dim])
        rot_loss = torch.mean(torch.abs(pred[:, :, pos_dim:] - target[:, :, pos_dim:]) * mask[:, :, pos_dim:])
        pred_diff = torch.diff(pred, dim=1)
        target_diff = torch.diff(target, dim=1)
        velocity_loss = torch.mean(torch.abs(pred_diff - target_diff) * mask[:, :-1, :])
        pred_acc = torch.diff(pred, dim=1, n=2)
        target_acc = torch.diff(target, dim=1, n=2)
        accel_loss = torch.mean(torch.abs(pred_acc - target_acc) * mask[:, :-2, :])
        kinetic_loss = torch.mean(((pred[:, :, pos_dim:] - target[:, :, pos_dim:]) ** 2) * mask[:, :, pos_dim:])
        
        total_loss = (
            self.config['training'].get('pos_weight', 1.0) * pos_loss +
            self.config['training'].get('rot_weight', 1.0) * rot_loss +
            self.config['training'].get('vel_weight', 0.5) * velocity_loss +
            self.config['training'].get('acc_weight', 0.2) * accel_loss +
            self.config['training'].get('kin_weight', 0.5) * kinetic_loss
        )
        return total_loss
    
    def train(self):
        self.model.train()
        epochs = 2 if self.test_run else self.config['training']['epochs']
        max_batches = 2 if self.test_run else None
        
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            
            for batch in self.data_loader:
                if max_batches and batch_count >= max_batches:
                    break
                
                waveform, transcripts, word_timings_list, target_poses, speaker_ids, waveform_lengths, motion_lengths = batch
                
                waveform = waveform.to(self.device)
                target_poses = target_poses.to(self.device)
                speaker_ids = speaker_ids.to(self.device)
                waveform_lengths = waveform_lengths.to(self.device)
                motion_lengths = motion_lengths.to(self.device)
                
                max_motion_length = motion_lengths.max().item()
                audio_feats = self.pase.extract_features(waveform, target_length=max_motion_length)
                
                batch_size = len(transcripts)
                text_feats_list = []
                for i in range(batch_size):
                    try:
                        text_feats = self.llama.extract_features(
                            transcripts[i],
                            word_timings_list[i],
                            self.config['model']['fps']
                        )
                        # Interpolate text_feats to match max_motion_length
                        if text_feats.size(0) != max_motion_length:
                            text_feats = torch.nn.functional.interpolate(
                                text_feats.unsqueeze(0).transpose(1, 2),
                                size=max_motion_length,
                                mode='linear',
                                align_corners=False
                            ).transpose(1, 2).squeeze(0)
                    except ValueError as e:
                        print(f"Error processing transcript {i}: {e}. Using fallback.")
                        text_feats = self.llama.extract_features(transcripts[i], [], self.config['model']['fps'])
                    text_feats_list.append(text_feats)
                
                text_feats = torch.stack(text_feats_list).to(self.device)
                
                self.optimizer.zero_grad()
                pred, _ = self.model(
                    audio_feats,
                    text_feats,
                    speaker_ids,
                    audio_lengths=motion_lengths,
                    text_lengths=motion_lengths
                )
                loss = self.loss_function(pred, target_poses, motion_lengths)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            if not self.test_run:
                self.scheduler.step(avg_loss)
        
        save_path = "llavataros_model.pth" if self.test_run else f"llavataros_model_epoch_{epochs}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config, test_run=True)
    trainer.train()