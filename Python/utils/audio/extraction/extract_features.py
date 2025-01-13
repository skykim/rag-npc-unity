# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# extract_features.py

import numpy as np
from utils.audio.load_audio import load_and_preprocess_audio, load_audio_from_bytes
from utils.audio.extraction.extract_features_utils import extract_mfcc_features, extract_zcr

def extract_audio_features(audio_input, sr=88200, from_bytes=False):
    if from_bytes:
        y, sr = load_audio_from_bytes(audio_input, sr)
    else:
        y, sr = load_and_preprocess_audio(audio_input, sr)
    
    frame_length = int(0.01667 * sr)  
    hop_length = frame_length // 2  
    min_frames = 9  

    num_frames = (len(y) - frame_length) // hop_length + 1

    if num_frames < min_frames:
        print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
        return None, None

    combined_features = extract_and_combine_features(y, sr, frame_length, hop_length)
    
    return combined_features, y

def extract_and_combine_features(y, sr, frame_length, hop_length):
    all_features = []

    mfcc_features, original_mfcc_frame_count = extract_mfcc_features(y, sr, frame_length, hop_length)
    all_features.append(mfcc_features)
    zcr_features = extract_zcr(y, frame_length, hop_length, original_mfcc_frame_count)
    all_features.append(zcr_features) 
        
    combined_features = np.hstack(all_features)

    return combined_features