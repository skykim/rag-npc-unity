# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# audio_processing.py

import numpy as np
import torch

def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def pad_audio_chunk(audio_chunk, frame_length, num_features):
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        padding = np.pad(
            audio_chunk,
            pad_width=((0, pad_length), (0, 0)),
            mode='reflect'
        )
        audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
    return audio_chunk

def decode_audio_chunk(audio_chunk, model, device):
    src_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)
        output_sequence = model.decoder(encoder_outputs)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    return decoded_outputs

def blend_chunks(chunk1, chunk2, overlap):
    """Linearly blends the overlapping region between two chunks."""
    # Adjust overlap if either chunk has fewer frames than overlap
    actual_overlap = min(overlap, len(chunk1), len(chunk2))
    
    # If actual_overlap is 0, return concatenation without blending
    if actual_overlap == 0:
        return np.vstack((chunk1, chunk2))
    
    blended_chunk = np.copy(chunk1)
    for i in range(actual_overlap):
        alpha = i / actual_overlap  # Blend factor from 0 to 1
        blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
    
    # Combine blended chunk with non-overlapping part of chunk2
    return np.vstack((blended_chunk, chunk2[actual_overlap:]))

def process_audio_features(audio_features, model, device, config):
    # Configuration settings
    frame_length = config['frame_size']  # Number of frames per chunk (e.g., 64)
    overlap = config.get('overlap', 12)  # Number of overlapping frames between chunks
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Process chunks with the specified overlap
    start_idx = 0
    while start_idx < num_frames:
        end_idx = min(start_idx + frame_length, num_frames)
        
        # Select and pad chunk if needed
        audio_chunk = audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        
        # Decode the current audio chunk
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device)
        decoded_outputs = decoded_outputs[:end_idx - start_idx]
        
        # Blend with the last chunk if it exists
        if all_decoded_outputs:
            last_chunk = all_decoded_outputs.pop()
            blended_chunk = blend_chunks(last_chunk, decoded_outputs, overlap)
            all_decoded_outputs.append(blended_chunk)
        else:
            all_decoded_outputs.append(decoded_outputs)
        
        # Move start index forward by (frame_length - overlap)
        start_idx += frame_length - overlap

    # Process any remaining frames to ensure total frame count matches input
    current_length = sum(len(chunk) for chunk in all_decoded_outputs)
    if current_length < num_frames:
        remaining_frames = num_frames - current_length
        final_chunk_start = num_frames - remaining_frames
        audio_chunk = audio_features[final_chunk_start:num_frames]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device)
        all_decoded_outputs.append(decoded_outputs[:remaining_frames])

    # Concatenate all chunks and trim to the original frame count
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)[:num_frames]
    
    # Normalize or apply any post-processing
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    final_decoded_outputs[:, :61] /= 100  

    return final_decoded_outputs
