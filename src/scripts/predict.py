import librosa
import torch
import soundfile as sf
import numpy as np

from tqdm import tqdm

from src.datasets.data_processing import preprocess_audio

from src.models.PAE import PAEInputFlattened
from src.utils.config import yaml_config_parser


def main():
    cfg = yaml_config_parser()

    model = PAEInputFlattened(cfg)

    model_checkpoint = ""

    checkpoint_path = "logs/checkpoints/" + model_checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_state_dict_keys = set(model.state_dict().keys())

    # Create a new state_dict, excluding unexpected keys
    new_state_dict = {
        key.replace('model.', ''): value 
        for key, value in checkpoint['state_dict'].items()
        if key.replace('model.', '') in model_state_dict_keys
    }

    model.load_state_dict(new_state_dict)

    model.eval()

    sr = 16000

    audio_sample = ""
    input_audios = preprocess_audio(audio_sample, 8000, 44100, sr, 2, 1)
    output_audios = []

    print(f"Generating output for {len(input_audios)} windows...")

    for i, input_audio in tqdm(enumerate(input_audios), total=len(input_audios)):
        input_tensor = torch.tensor(input_audio).unsqueeze(0)

        with torch.no_grad():
            #tqdm.write("Generating output...")
            output_tensor, _, _, _ = model(input_tensor)

        output_audio = output_tensor.squeeze().cpu().numpy()
        output_audios.append(output_audio)

    output_audio = np.concatenate(output_audios)
    sf.write('output.wav', output_audio, sr)


if __name__ == "__main__":
    main()
