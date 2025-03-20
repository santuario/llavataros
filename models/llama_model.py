import torch
import os
from dotenv import load_dotenv
from transformers import LlamaTokenizer, LlamaModel

# Load environment variables from .env file
load_dotenv()

class LLaMAModel:
    def __init__(self, model_name, device, cache_dir=None):
        # Retrieve the Hugging Face token from the environment
        self.use_auth_token = ""
        if not self.use_auth_token:
            raise ValueError("HF_AUTH_TOKEN not found in .env file. Please set it and try again.")

        # Handle authentication for gated models
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token,
                cache_dir=cache_dir
            )
            self.model = LlamaModel.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token,
                cache_dir=cache_dir
            ).to(device)
            self.model.eval()
            self.device = device
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("If this is a gated repository, please ensure you have proper access and authentication.")
            raise

    def extract_features(self, transcript, word_timings, fps=30):
        # Validate inputs
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Transcript must be a non-empty string.")
        if not word_timings or not isinstance(word_timings, list):
            raise ValueError("word_timings must be a non-empty list of (word, start, end) tuples.")

        # Tokenize the transcript
        inputs = self.tokenizer(transcript, return_tensors="pt", truncation=True, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0]  # Shape: (num_tokens, 2)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (num_tokens, 4096)

        # Map words to tokens
        word_to_token_mapping = []
        current_word_idx = 0
        transcript_words = transcript.split()
        char_pos = 0

        for token_idx, (start_char, end_char) in enumerate(offsets):
            token_text = self.tokenizer.decode(input_ids[0][token_idx], skip_special_tokens=True).strip()
            if token_text:  # Ignore special tokens
                while current_word_idx < len(transcript_words):
                    word = transcript_words[current_word_idx]
                    word_start = transcript.find(word, char_pos)
                    word_end = word_start + len(word)
                    if word_start <= start_char < word_end:
                        word_to_token_mapping.append((current_word_idx, token_idx))
                        char_pos = word_end
                        current_word_idx += 1
                        break
                    else:
                        current_word_idx += 1

        # Align embeddings to 30 fps
        if not word_timings:
            raise ValueError("word_timings is empty.")
        N = int(word_timings[-1][2] * fps)  # Total frames based on end time of last word
        aligned_embeddings = torch.zeros((N, embeddings.shape[-1]), device=self.device)

        for word_idx, (word, start, end) in enumerate(word_timings):
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            # Find the corresponding token indices for this word
            token_indices = [t_idx for w_idx, t_idx in word_to_token_mapping if w_idx == word_idx]
            if token_indices and token_indices[0] < len(embeddings):
                # Use the first token's embedding (you could average if multiple tokens)
                token_idx = token_indices[0]
                aligned_embeddings[start_frame:end_frame] = embeddings[token_idx]

        return aligned_embeddings