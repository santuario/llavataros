import os
import whisper
from pathlib import Path
from tqdm import tqdm

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size (str): Size of the Whisper model (e.g., "tiny", "base", "small", "medium", "large").
        """
        self.model = whisper.load_model(model_size)
        self.model.eval()

    def transcribe(self, audio_path, output_dir):
        """
        Transcribe an audio file and save the transcript as a .txt file.
        
        Args:
            audio_path (str): Path to the .wav file.
            output_dir (str): Directory to save the transcript.
        
        Returns:
            str: Path to the generated transcript file.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        transcript_path = output_dir / f"{audio_path.stem}.txt"
        
        # Skip if transcript already exists
        if transcript_path.exists():
            return str(transcript_path)
        
        # Transcribe audio
        result = self.model.transcribe(str(audio_path), language="en")  # Assuming English audio
        transcript = result["text"]
        
        # Save transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        return str(transcript_path)

    def transcribe_dataset(self, dataset_dir):
        """
        Transcribe all .wav files in the dataset directory.
        
        Args:
            dataset_dir (str): Directory containing .wav files.
        
        Returns:
            dict: Mapping of .wav file paths to their transcript file paths.
        """
        dataset_dir = Path(dataset_dir)
        wav_files = list(dataset_dir.glob("*.wav"))
        transcript_map = {}
        
        print("Transcribing audio files...")
        for wav_file in tqdm(wav_files):
            transcript_path = self.transcribe(wav_file, dataset_dir)
            transcript_map[str(wav_file)] = transcript_path
        
        return transcript_map