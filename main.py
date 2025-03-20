import argparse
from scripts.train import Trainer
from scripts.inference import Inference
import yaml

def main():
    parser = argparse.ArgumentParser(description="LLanimation: Llama Driven Gesture Animation")
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="Mode to run: train or infer")
    parser.add_argument("--audio_path", type=str, help="Path to audio file for inference")
    parser.add_argument("--transcript", type=str, default=None, help="Transcript for inference (optional)")
    parser.add_argument("--test", action="store_true", help="Run a small training test")
    parser.add_argument("--output_npy_path", type=str, default=None, help="Path to save the generated .npy animation file")
    parser.add_argument("--output_mov_path", type=str, default=None, help="Path to save the generated .mov preview file")
    args = parser.parse_args()

    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        trainer = Trainer(config, test_run=args.test)
        trainer.train()
    elif args.mode == "infer":
        if not args.audio_path:
            raise ValueError("Audio path is required for inference")
        inference = Inference(config, "llanimation_model.pth")
        prediction = inference.infer(
            args.audio_path,
            transcript=args.transcript,
            output_npy_path=args.output_npy_path,
            output_mov_path=args.output_mov_path
        )
        print("Inference completed. Output shape:", prediction.shape)

if __name__ == "__main__":
    main()