import librosa
import numpy as np

def extract_mfcc_features(y, sr, frame_length, hop_length, num_mfcc=26):
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)
    reduced_mfcc_features = reduce_features(mfcc_features)
    return reduced_mfcc_features.T, mfcc_features.shape[1]  

def extract_zcr(y, frame_length, hop_length, original_mfcc_frame_count):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = zcr[:original_mfcc_frame_count]  
    zcr_reduced = reduce_features(zcr[np.newaxis, :]).flatten()
    zcr_reduced = handle_first_last_frames(zcr_reduced[:, np.newaxis])
    return zcr_reduced

def cepstral_mean_variance_normalization(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    return (mfcc - mean) / (std + 1e-10)

def extract_overlapping_mfcc(chunk, sr, num_mfcc, frame_length, hop_length, ):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc, n_fft=frame_length, hop_length=hop_length)
    mfcc = cepstral_mean_variance_normalization(mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return combined_mfcc    


def handle_first_last_frames(features):
    if features.shape[0] > 1:  
        if np.all(features[0] == 0):
            features[0] = (features[1] + features[2]) / 2
        if np.all(features[-1] == 0):
            features[-1] = (features[-2] + features[-3]) / 2
    
    return features

def reduce_features(features):
    num_frames = features.shape[1]
    paired_frames = features[:, :num_frames // 2 * 2].reshape(features.shape[0], -1, 2)
    reduced_frames = paired_frames.mean(axis=2)
    
    if num_frames % 2 == 1:
        last_frame = features[:, -1].reshape(-1, 1)
        reduced_final_features = np.hstack((reduced_frames, last_frame))
    else:
        reduced_final_features = reduced_frames
    
    return reduced_final_features

