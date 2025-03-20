import os
import subprocess
from pathlib import Path
import textgrid

class MontrealForcedAligner:
    def __init__(self, dataset_dir, output_dir, transcript_map):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.transcript_map = transcript_map
        self.dictionary_path = "/app/mfa_models/english_us_arpa.dict"
        self.acoustic_model_path = "/app/mfa_models/english_us_arpa"

    def align(self):
        # Ensure transcripts exist
        for wav_path, txt_path in self.transcript_map.items():
            if not Path(txt_path).exists():
                raise FileNotFoundError(f"Transcript not found: {txt_path}")
        
        # Run MFA alignment with Conda environment activation
        command = (
            "source /opt/conda/etc/profile.d/conda.sh && "
            "conda activate mfa && "
            f"mfa align {str(self.dataset_dir)} {self.dictionary_path} {self.acoustic_model_path} {str(self.output_dir)} --clean"
        )
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")

    def get_word_timings(self, audio_file):
        audio_name = Path(audio_file).stem
        # Transform the file stem to match MFA's naming convention
        parts = audio_name.rsplit('.', 1)
        if len(parts) > 1:
            audio_name_mfa = f"{parts[0]}_{parts[1]}"
        else:
            audio_name_mfa = audio_name
        # Look for the TextGrid file directly in the output directory
        textgrid_path = self.output_dir / f"{audio_name_mfa}.TextGrid"
        
        # Debug: Check if the TextGrid file exists
        print(f"Looking for TextGrid file at: {textgrid_path}")
        if not textgrid_path.exists():
            print(f"TextGrid file does not exist at: {textgrid_path}")
            print(f"Listing contents of {self.output_dir}:")
            os.system(f"ls -R {self.output_dir}")
            raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")
        
        # Parse TextGrid file
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        word_tier = None
        for tier in tg.tiers:
            if tier.name.lower() == "words":
                word_tier = tier
                break
        
        if not word_tier:
            raise ValueError(f"No 'words' tier found in TextGrid: {textgrid_path}")
        
        # Extract word timings
        word_timings = []
        for interval in word_tier.intervals:
            if interval.mark and interval.mark.strip():  # Ignore empty intervals
                word_timings.append((interval.mark, interval.minTime, interval.maxTime))
        
        return word_timings