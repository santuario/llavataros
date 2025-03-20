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
        self.pase = PASEModel(config['paths']['pase_config'], config['paths']['pase_checkpoint']).to(self.device)
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
        """
        Compute loss with masking for padded regions.
        Args:
            pred: Model predictions, shape (batch_size, seq_len, 7 * num_joints)
            target: Target poses, shape (batch_size, max_motion_length, 7 * num_joints)
            lengths: Original sequence lengths, shape (batch_size,)
        """
        # Create mask to exclude padded regions (True for padded, False for valid)
        max_len = target.size(1)
        mask = torch.arange(max_len, device=self.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(target)  # Shape: (batch_size, max_len, 7 * num_joints)
        
        # Ensure pred and target have compatible sequence lengths
        if pred.size(1) < target.size(1):
            pred = torch.nn.functional.pad(pred, (0, 0, 0, target.size(1) - pred.size(1)))
        elif pred.size(1) > target.size(1):
            pred = pred[:, :target.size(1), :]
        
        # Separate position (px, py, pz) and rotation (qx, qy, qz, qw) assuming 7 values per joint
        num_features = target.size(-1)
        pos_dim = 3  # px, py, pz
        rot_dim = num_features - pos_dim  # qx, qy, qz, qw per joint
        
        # Position loss
        pos_loss = torch.mean(torch.abs(pred[:, :, :pos_dim] - target[:, :, :pos_dim]) * mask[:, :, :pos_dim])
        
        # Rotation loss
        rot_loss = torch.mean(torch.abs(pred[:, :, pos_dim:] - target[:, :, pos_dim:]) * mask[:, :, pos_dim:])
        
        # Velocity loss (first derivative)
        pred_diff = torch.diff(pred, dim=1)
        target_diff = torch.diff(target, dim=1)
        velocity_loss = torch.mean(torch.abs(pred_diff - target_diff) * mask[:, :-1, :])
        
        # Acceleration loss (second derivative)
        pred_acc = torch.diff(pred, dim=1, n=2)
        target_acc = torch.diff(target, dim=1, n=2)
        accel_loss = torch.mean(torch.abs(pred_acc - target_acc) * mask[:, :-2, :])
        
        # Kinetic loss (squared difference on rotations)
        kinetic_loss = torch.mean(((pred[:, :, pos_dim:] - target[:, :, pos_dim:]) ** 2) * mask[:, :, pos_dim:])
        
        # Combine losses with weights (optional: tune these in config)
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
                
                # Unpack batch with lengths
                waveform, transcript, word_timings, target_poses, speaker_ids, waveform_lengths, motion_lengths = batch
                
                # Move to device
                waveform = waveform.to(self.device)
                target_poses = target_poses.to(self.device)
                speaker_ids = speaker_ids.to(self.device)
                waveform_lengths = waveform_lengths.to(self.device)
                motion_lengths = motion_lengths.to(self.device)
                
                # Extract features
                audio_feats = self.pase.extract_features(waveform, target_length=motion_lengths.max().item())
                text_feats = self.llama.extract_features(transcript[0], word_timings, self.config['model']['fps'])
                
                # Forward pass with sequence lengths
                self.optimizer.zero_grad()
                pred, _ = self.model(
                    audio_feats,
                    text_feats,
                    speaker_ids,
                    audio_lengths=waveform_lengths,
                    text_lengths=motion_lengths  # Proxy for text_feats length
                )
                loss = self.loss_function(pred, target_poses, motion_lengths)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update running loss
                total_loss += loss.item()
                batch_count += 1
                
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")
            
            # Epoch summary
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Step scheduler
            if not self.test_run:
                self.scheduler.step(avg_loss)
        
        # Save model
        save_path = "llavataros_model.pth" if self.test_run else f"llavataros_model_epoch_{epochs}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config, test_run=True)
    trainer.train()