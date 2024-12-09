import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import scipy.signal as signal
import soundfile as sf


def preprocess_audio(file_path, lowpass_cutoff=8000, original_rate=44100, target_rate=16000, window_length=2):

    audio, sr = sf.read(file_path)
    audio = np.mean(audio, axis=1) #Convert audio to mono
    assert sr == original_rate, f"Expected {original_rate} Hz, but got {sr} Hz"
    
    # low-pass filter
    nyquist = sr / 2
    norm_cutoff = lowpass_cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    # Downsample the audio to 16 KHz
    downsampled_audio = signal.resample_poly(filtered_audio, target_rate, sr)
    
    # normalize to -1 to 1
    downsampled_audio = downsampled_audio / np.max(np.abs(downsampled_audio))
    
    # make 2-second windows
    window_size = target_rate * window_length
    step_size = window_size // 2  # 50% overlap

    windows = [
        downsampled_audio[i:i + window_size]
        for i in range(0, len(downsampled_audio) - window_size + 1, step_size)
    ]
    return windows

class AudioDataset(Dataset):
    def __init__(self, dataset_root, split, lowpass_cutoff=8000, original_rate=44100, target_rate=16000, window_length=2):
        self.audio_dir = dataset_root
        self.split = split
        self.lowpass_cutoff = lowpass_cutoff
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.window_length = window_length
        
        # load and preprocess audio files
        self.data = []
        for file_name in os.listdir(dataset_root):
            if file_name.endswith(".wav"):
                file_path = os.path.join(dataset_root, file_name)
                self.data.extend(preprocess_audio(file_path, self.lowpass_cutoff, self.original_rate, self.target_rate, self.window_length))
        
        # Split into train, test, and validation
        num_train = int(0.8 * len(self.data))
        num_valid = int(0.1 * len(self.data))
        num_test = len(self.data) - num_train - num_valid
        train_data, valid_data, test_data = random_split(self.data, [num_train, num_valid, num_test])
        self.data = {"train": train_data, "valid": valid_data, "test": test_data}[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)

# Example 
if __name__ == "__main__":
    dataset_root = os.path.join(os.getcwd(), "data")
    train_dataset = AudioDataset(dataset_root, split='train')
    valid_dataset = AudioDataset(dataset_root, split='valid')
    test_dataset = AudioDataset(dataset_root, split='test')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Access a sample
    sample = train_dataset[0]
    print(f"Sample shape: {sample.shape}")
