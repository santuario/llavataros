import torch
import torch.nn as nn

class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embedding_dim)
    
    def forward(self, speaker_ids):
        return self.embedding(speaker_ids)

class CrossAttentiveTransformerXL(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.num_joints = config['model']['num_joints']
        self.device = device
        
        # Input projections
        self.audio_proj = nn.Linear(256 + 8, self.d_model)  # PASE+ (256) + speaker (8)
        self.text_proj = nn.Linear(4096 + 8, self.d_model)  # LLaMA2 (4096) + speaker (8)
        
        # Dynamic positional encoding (computed in forward)
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, self.d_model))  # Increased max length
        
        # Speaker embedding
        self.speaker_embedder = SpeakerEmbedding(num_speakers=100)
        
        # Transformer layers with padding mask support
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['model']['n_heads'],
            dim_feedforward=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            batch_first=True  # Use batch_first for simplicity
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['model']['n_layers'])
        
        # Cross-attention with padding mask support
        self.cross_attention = nn.MultiheadAttention(
            self.d_model, config['model']['n_heads'], dropout=config['model']['dropout'], batch_first=True
        )
        
        # Output layer (match target_poses dimensionality)
        self.output_layer = nn.Linear(self.d_model, 7 * self.num_joints)  # [px, py, pz, qx, qy, qz, qw] per joint
        
        self.to(device)

    def forward(self, audio_feats, text_feats, speaker_ids, audio_lengths=None, text_lengths=None, memory=None):
        # Determine batch size and sequence length
        batch_size = audio_feats.size(0) if audio_feats is not None else text_feats.size(0)
        
        # Get speaker embeddings
        speaker_emb = self.speaker_embedder(speaker_ids)  # Shape: (batch_size, embedding_dim)
        speaker_emb = speaker_emb.unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        
        # Process inputs with padding masks
        if text_feats is None:  # PASE+ only
            seq_len = audio_feats.size(1)
            x = torch.cat([audio_feats, speaker_emb.expand(-1, seq_len, -1)], dim=-1)
            x = self.audio_proj(x)  # Shape: (batch_size, seq_len, d_model)
            key_padding_mask = self._get_padding_mask(audio_lengths, seq_len) if audio_lengths is not None else None
        elif audio_feats is None:  # LLaMA2 only
            seq_len = text_feats.size(1)
            x = torch.cat([text_feats, speaker_emb.expand(-1, seq_len, -1)], dim=-1)
            x = self.text_proj(x)  # Shape: (batch_size, seq_len, d_model)
            key_padding_mask = self._get_padding_mask(text_lengths, seq_len) if text_lengths is not None else None
        else:  # Combination with cross-attention
            audio_seq_len = audio_feats.size(1)
            text_seq_len = text_feats.size(1)
            audio_x = self.audio_proj(torch.cat([audio_feats, speaker_emb.expand(-1, audio_seq_len, -1)], dim=-1))
            text_x = self.text_proj(torch.cat([text_feats, speaker_emb.expand(-1, text_seq_len, -1)], dim=-1))
            audio_mask = self._get_padding_mask(audio_lengths, audio_seq_len) if audio_lengths is not None else None
            text_mask = self._get_padding_mask(text_lengths, text_seq_len) if text_lengths is not None else None
            x, _ = self.cross_attention(audio_x, text_x, text_x, key_padding_mask=text_mask)
            key_padding_mask = audio_mask
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len > self.pos_encoding.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum positional encoding length {self.pos_encoding.size(1)}")
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer with padding mask
        output = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        return self.output_layer(output), x  # Shape: (batch_size, seq_len, 7 * num_joints), memory

    def _get_padding_mask(self, lengths, max_len):
        """Create a padding mask where True indicates padding."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask