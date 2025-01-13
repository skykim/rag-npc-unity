# audio_utils.py

from scipy.signal import butter, lfilter
import numpy as np


def bandpass_filter(y, sr, lowcut=300, highcut=3400, order=1):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    return y_filtered

def loudness_normalization(y, target_peak_dB=-3.0, percentile=95):
    # Compute a robust peak by taking the percentile of the absolute values
    robust_peak = np.percentile(np.abs(y), percentile)
    
    # Convert the robust peak to dB
    current_peak_dB = 20 * np.log10(robust_peak)
    
    # Compute the gain required to match the target peak dB
    gain = 10**((target_peak_dB - current_peak_dB) / 20)
    
    # Apply the gain to normalize the audio signal
    y_normalized = y * gain
    
    return y_normalized


