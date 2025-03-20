import torch
import torchaudio
import yaml
from pase.pase.models.frontend import wf_builder
from transformers import LlamaTokenizer, LlamaModel
import whisper
import subprocess
import os
from dotenv import load_dotenv
from models.transformer_xl import CrossAttentiveTransformerXL
from utils.transcriber import WhisperTranscriber
from utils.aligner import MontrealForcedAligner


# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face token from the environment
HF_AUTH_TOKEN = ""
if not HF_AUTH_TOKEN:
    raise ValueError("HF_AUTH_TOKEN not found in .env file. Please set it and try again.")


def test_pase(config):
    print("Testing PASE+ model...")
    pase = wf_builder(config['paths']['pase_config']).eval()
    pase.load_pretrained(config['paths']['pase_checkpoint'], load_last=True, verbose=True)
    
    # Ensure consistent device usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and input to the same device
    x = torch.randn(1, 1, 100000).to(device)
    pase = pase.to(device)  # Ensure model is on the same device

    # Perform inference
    with torch.no_grad():
        y = pase(x)

    # Print output shape
    print(f"PASE+ output shape: {y.shape}")
    print("PASE+ test passed!")

def test_llama(config):
    print("Testing LLaMA2 model...")
    device = torch.device("cuda")
    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(
        config["paths"]["llama_model"],
        use_auth_token=HF_AUTH_TOKEN,
    )
    model = LlamaModel.from_pretrained(
        config["paths"]["llama_model"],
        use_auth_token=HF_AUTH_TOKEN,
    ).to(device)
    model.eval()
    
    # Test with a sample text
    text = "Hello, how are you?"
    tokens = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    print(f"LLaMA2 embeddings shape: {embeddings.shape}")  # Expected: (1, num_tokens, 4096)
    print("LLaMA2 test passed!")

def test_whisper():
    print("Testing Whisper transcriber...")
    transcriber = WhisperTranscriber(model_size="base")
    audio_path = "data/datasets/zaos1/001_Neutral_0_mirror_x_1_0.wav"
    transcript_path = transcriber.transcribe(audio_path, "data/datasets/zaos1")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    print(f"Whisper transcript: {transcript}")
    print("Whisper test passed!")

def test_mfa(config):
    print("Testing Montreal Forced Aligner...")
    transcriber = WhisperTranscriber(model_size="base")
    transcript_map = transcriber.transcribe_dataset(config['paths']['dataset_dir'])
    aligner = MontrealForcedAligner(
        config['paths']['dataset_dir'],
        config['paths']['alignments_dir'],
        transcript_map
    )
    aligner.align()
    audio_path = "data/datasets/zaos1/001_Neutral_0_mirror_x_1_0.wav"
    word_timings = aligner.get_word_timings(audio_path)
    print(f"MFA word timings (first 5): {word_timings[:5]}")
    print("MFA test passed!")


def test_transformer(config):
    print("Testing Transformer-XL model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentiveTransformerXL(config, device)
    
    # Create dummy inputs for testing
    batch_size = 2
    seq_len = 10
    
    # Dummy audio features (PASE+ features: 256 dimensions)
    audio_feats = torch.randn(batch_size, seq_len, 256, device=device)
    
    # Dummy text features (LLaMA2 embeddings: 4096 dimensions)
    text_feats = torch.randn(batch_size, seq_len, 4096, device=device)
    
    # Dummy speaker IDs
    speaker_ids = torch.randint(0, 100, (batch_size,), device=device)
    
    print("Input shapes:", audio_feats.shape, text_feats.shape, speaker_ids.shape)
    
    # Test the model with both audio and text features
    output, memory = model(audio_feats, text_feats, speaker_ids)
    print("Output shape (audio+text):", output.shape)
    
    # Test the model with audio features only
    output, memory = model(audio_feats, None, speaker_ids)
    print("Output shape (audio only):", output.shape)
    
    # Test the model with text features only
    output, memory = model(None, text_feats, speaker_ids)
    print("Output shape (text only):", output.shape)
    
    print("Transformer-XL test passed!")

def main():

    with open("config/model_config.yaml", "r") as f:
        pase_config = yaml.safe_load(f)
    # Run tests
    test_pase(pase_config)
    test_llama(pase_config)
    test_whisper()
    test_mfa(pase_config)
    test_transformer(pase_config)

if __name__ == "__main__":
    main()