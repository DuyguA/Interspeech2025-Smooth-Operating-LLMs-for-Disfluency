import torch
import librosa
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

def extract_and_save_wav2vec_features(audio_path, output_path, sampling_rate=16000):
    """
    Extract Wav2Vec-Conformer features for an audio file .
    
    Args:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the extracted tensor.
        sampling_rate (int): Sampling rate of the audio (default: 16 kHz).
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Tokenize and move inputs to GPU
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.half().cuda()  # FP16 input


    # Save the tensor to disk
    torch.save(input_values, output_path)
    print(f"Saved features to {output_path}")


import os

def batch_extract_and_save_features(file_list, output_dir, sampling_rate=16000):
    """
    Process a batch of audio files and save their embeddings in FP16.

    Args:
        file_list (list): List of paths to audio files.
        output_dir (str): Directory to save the extracted tensors.
        sampling_rate (int): Sampling rate of the audio (default: 16 kHz).
    """
    os.makedirs(output_dir, exist_ok=True) 

    for path in file_list:
        audio_path = "stutter/disfluent_audio/" + path + ".wav"
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Define output path
        output_path = os.path.join(output_dir, f"{path}_features.pt")

        # Extract and save features
        extract_and_save_wav2vec_features(audio_path, output_path, sampling_rate)

train_files = open("train_manifest.txt", "r").read().split("\n")[:-1]
test_files = open("train_manifest.txt", "r").read().split("\n")[:-1]
file_list = train_files + test_files
output_dir = "/ephemeral/features/"  # Directory to save tensors
batch_extract_and_save_features(file_list, output_dir)

