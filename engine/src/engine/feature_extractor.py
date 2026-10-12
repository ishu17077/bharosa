"""
Bharosa (भरोसा) — 8kHz Telephony Voice Anti-Spoofing Feature Extractor
Extracts 32-dimensional spectro-temporal forensic features engineered specifically
to identify neural vocoder phase smearing and synthetic artifacts that survive G.711 / AMR compression.
"""

import numpy as np
from scipy import signal, fft

# 32 Feature Names in exact order
FEATURE_NAMES = [
    # 0-11: Linear Frequency Cepstral Coefficients (LFCC 1-12)
    "lfcc_1", "lfcc_2", "lfcc_3", "lfcc_4", "lfcc_5", "lfcc_6",
    "lfcc_7", "lfcc_8", "lfcc_9", "lfcc_10", "lfcc_11", "lfcc_12",
    # 12-19: Spectral Statistical Moments
    "spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
    "spectral_flatness", "spectral_crest", "spectral_rolloff_85", "spectral_rolloff_95",
    # 20-23: Pitch & Glottal Biomechanics
    "f0_mean_hz", "f0_jitter_pct", "harmonic_to_noise_ratio_db", "energy_shimmer_pct",
    # 24-27: Neural Vocoder Artifact Indices
    "phase_discontinuity_index", "spectral_flux", "high_band_energy_ratio", "transposed_conv_ripple",
    # 28-31: Telephony & Temporal Modulation
    "g711_quant_distortion", "zcr_mean", "zcr_variance", "syllabic_modulation_4hz"
]


def pre_emphasis(signal_data: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to amplify high frequencies."""
    if len(signal_data) == 0:
        return signal_data
    return np.append(signal_data[0], signal_data[1:] - coeff * signal_data[:-1])


def linear_filterbank(num_filters: int = 20, n_fft: int = 512, sample_rate: int = 8000,
                      f_min: float = 100.0, f_max: float = 3800.0) -> np.ndarray:
    """Construct linearly spaced triangular filterbank for 8kHz telephony audio."""
    half_fft = n_fft // 2 + 1
    f_points = np.linspace(f_min, f_max, num_filters + 2)
    bin_points = np.floor((f_points / (sample_rate / 2)) * (half_fft - 1)).astype(int)

    fbank = np.zeros((num_filters, half_fft))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        if f_m - f_m_minus > 0:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        if f_m_plus - f_m > 0:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank


def extract_lfcc(samples: np.ndarray, sample_rate: int = 8000, num_ceps: int = 12) -> np.ndarray:
    """Extract 12 Linear Frequency Cepstral Coefficients (LFCC) which outperform MFCC in anti-spoofing."""
    n_fft = 512
    # Frame and window
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    
    # Hann window
    window = np.hanning(len(samples))
    windowed = samples * window
    
    # Power spectrum
    spectrum = np.abs(fft.rfft(windowed, n=n_fft)) ** 2
    spectrum = np.maximum(spectrum, 1e-12)

    # Filterbank
    fbank = linear_filterbank(num_filters=20, n_fft=n_fft, sample_rate=sample_rate)
    filter_energies = np.dot(fbank, spectrum)
    filter_energies = np.maximum(filter_energies, 1e-12)
    log_energies = np.log(filter_energies)

    # Discrete Cosine Transform (DCT-II)
    lfcc = fft.dct(log_energies, type=2, norm='ortho')[:num_ceps]
    return lfcc


def extract_pitch_and_hnr(samples: np.ndarray, sample_rate: int = 8000) -> tuple:
    """Estimate fundamental frequency (F0), jitter, and Harmonic-to-Noise Ratio via autocorrelation."""
    if len(samples) < 256:
        return 130.0, 1.0, 15.0, 2.0

    # Center-clipping or autocorrelation
    norm_samples = samples - np.mean(samples)
    std = np.std(norm_samples)
    if std < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    corr = signal.correlate(norm_samples, norm_samples, mode='full')
    corr = corr[len(corr) // 2:]
    corr = corr / (corr[0] + 1e-12)

    # Search pitch lag between 60Hz (lag ~ 133) and 350Hz (lag ~ 23)
    min_lag = int(sample_rate / 350)
    max_lag = int(sample_rate / 60)

    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return 130.0, 1.0, 15.0, 2.0

    peak_rel_idx = np.argmax(search_region)
    peak_val = search_region[peak_rel_idx]
    actual_lag = min_lag + peak_rel_idx

    f0 = sample_rate / actual_lag if actual_lag > 0 else 130.0

    # Harmonic to noise ratio: HNR = 10 * log10(R(lag) / (1 - R(lag)))
    clamped_peak = np.clip(peak_val, 0.01, 0.99)
    hnr_db = 10.0 * np.log10(clamped_peak / (1.0 - clamped_peak + 1e-8))

    # Jitter & Shimmer estimation across sub-windows
    sub_frames = np.array_split(samples, 4)
    sub_f0s = []
    sub_energies = []
    for sf in sub_frames:
        if len(sf) > 100:
            c = signal.correlate(sf - np.mean(sf), sf - np.mean(sf), mode='full')
            c = c[len(c) // 2:]
            c = c / (c[0] + 1e-12)
            if len(c) > max_lag:
                sr = c[min_lag:max_lag]
                if len(sr) > 0:
                    sub_f0s.append(sample_rate / (min_lag + np.argmax(sr)))
            sub_energies.append(np.sqrt(np.mean(sf ** 2)))

    f0_jitter_pct = (np.std(sub_f0s) / (np.mean(sub_f0s) + 1e-8)) * 100.0 if len(sub_f0s) > 1 else 1.2
    energy_shimmer_pct = (np.std(sub_energies) / (np.mean(sub_energies) + 1e-8)) * 100.0 if len(sub_energies) > 1 else 3.5

    return float(f0), float(f0_jitter_pct), float(hnr_db), float(energy_shimmer_pct)


def extract_features_32d(samples: np.ndarray, sample_rate: int = 8000) -> np.ndarray:
    """
    Extract complete 32-dimensional forensic feature vector for 8kHz audio.
    Input: raw float32 samples (between -1.0 and 1.0).
    Output: np.ndarray shape (32,)
    """
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) == 0:
        return np.zeros(32, dtype=np.float32)

    features = np.zeros(32, dtype=np.float32)

    # 1. Pre-emphasis
    emp_samples = pre_emphasis(samples)

    # 2. Extract 12 LFCCs (Indices 0 - 11)
    lfcc = extract_lfcc(emp_samples, sample_rate=sample_rate, num_ceps=12)
    features[0:12] = lfcc

    # 3. FFT Magnitude and Power Spectrum
    n_fft = 512
    spec = np.abs(fft.rfft(emp_samples, n=n_fft))
    power = spec ** 2
    freqs = fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    sum_power = np.sum(power) + 1e-12
    prob_dist = power / sum_power

    # 4. Spectral Statistical Moments (Indices 12 - 19)
    # 12: Spectral Centroid
    centroid = np.sum(freqs * prob_dist)
    features[12] = centroid / 4000.0  # Normalized by Nyquist

    # 13: Spectral Spread (Bandwidth)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * prob_dist))
    features[13] = spread / 2000.0

    # 14: Spectral Skewness
    skewness = np.sum(((freqs - centroid) ** 3) * prob_dist) / ((spread ** 3) + 1e-8)
    features[14] = np.clip(skewness / 10.0, -1.0, 1.0)

    # 15: Spectral Kurtosis
    kurtosis = np.sum(((freqs - centroid) ** 4) * prob_dist) / ((spread ** 4) + 1e-8) - 3.0
    features[15] = np.clip(kurtosis / 20.0, -1.0, 1.0)

    # 16: Spectral Flatness (Wiener Entropy)
    geom_mean = np.exp(np.mean(np.log(power + 1e-12)))
    arith_mean = np.mean(power) + 1e-12
    flatness = geom_mean / arith_mean
    features[16] = float(flatness)

    # 17: Spectral Crest Factor
    crest = np.max(power) / arith_mean
    features[17] = np.clip(crest / 50.0, 0.0, 1.0)

    # 18-19: Spectral Roll-off (85% and 95%)
    cum_power = np.cumsum(power)
    idx_85 = np.where(cum_power >= 0.85 * sum_power)[0]
    idx_95 = np.where(cum_power >= 0.95 * sum_power)[0]
    features[18] = (freqs[idx_85[0]] if len(idx_85) > 0 else 3400.0) / 4000.0
    features[19] = (freqs[idx_95[0]] if len(idx_95) > 0 else 3800.0) / 4000.0

    # 5. Pitch & Glottal Dynamics (Indices 20 - 23)
    f0, jitter, hnr, shimmer = extract_pitch_and_hnr(samples, sample_rate)
    features[20] = f0 / 300.0
    features[21] = np.clip(jitter / 5.0, 0.0, 1.0)
    features[22] = np.clip(hnr / 30.0, 0.0, 1.0)
    features[23] = np.clip(shimmer / 10.0, 0.0, 1.0)

    # 6. Vocoder Phase & High-Frequency Artifacts (Indices 24 - 27)
    # 24: Phase Discontinuity Index
    # Analytic signal via Hilbert transform for instantaneous phase
    analytic = signal.hilbert(samples)
    inst_phase = np.unwrap(np.angle(analytic))
    phase_diff2 = np.diff(inst_phase, n=2)
    features[24] = np.clip(np.std(phase_diff2) / 2.0, 0.0, 1.0)

    # 25: Spectral Flux
    half = len(samples) // 2
    spec1 = np.abs(fft.rfft(samples[:half], n=256))
    spec2 = np.abs(fft.rfft(samples[half:], n=256))
    flux = np.sum((spec2 - spec1) ** 2) / (np.sum(spec1 ** 2) + 1e-12)
    features[25] = np.clip(flux, 0.0, 1.0)

    # 26: High Band (2.5k - 3.8k) to Low Band (0.3k - 1.5k) Energy Ratio
    high_mask = (freqs >= 2500) & (freqs <= 3800)
    low_mask = (freqs >= 300) & (freqs <= 1500)
    high_energy = np.sum(power[high_mask]) + 1e-12
    low_energy = np.sum(power[low_mask]) + 1e-12
    features[26] = np.clip((high_energy / low_energy) / 2.0, 0.0, 1.0)

    # 27: Transposed Convolution Ripple (Periodic peaks in upper spectrum)
    high_spec = spec[high_mask]
    if len(high_spec) > 8:
        autocorr_spec = np.correlate(high_spec - np.mean(high_spec), high_spec - np.mean(high_spec), mode='full')
        autocorr_spec = autocorr_spec[len(autocorr_spec) // 2:]
        features[27] = np.clip(np.max(autocorr_spec[1:6]) / (autocorr_spec[0] + 1e-8), 0.0, 1.0)
    else:
        features[27] = 0.0

    # 7. Telephony & Temporal Modulation (Indices 28 - 31)
    # 28: G.711 mu-law quantization error
    mu = 255.0
    companded = np.sign(samples) * np.log(1.0 + mu * np.abs(samples)) / np.log(1.0 + mu)
    quantized = np.round(companded * 128.0) / 128.0
    quant_error = np.mean((companded - quantized) ** 2)
    features[28] = np.clip(quant_error * 1000.0, 0.0, 1.0)

    # 29-30: Zero Crossing Rate mean & variance
    zcr = np.abs(np.diff(np.sign(samples))) > 0
    # Split into 8 sub-blocks to measure variance
    zcr_blocks = np.array_split(zcr, 8)
    zcr_means = [np.mean(b) for b in zcr_blocks if len(b) > 0]
    features[29] = float(np.mean(zcr_means)) if len(zcr_means) > 0 else 0.0
    features[30] = float(np.std(zcr_means) * 10.0) if len(zcr_means) > 0 else 0.0

    # 31: Syllabic Modulation (~4Hz envelope power)
    env = np.abs(analytic)
    down_env = signal.resample(env, 128)
    env_fft = np.abs(fft.rfft(down_env - np.mean(down_env)))
    env_freqs = fft.rfftfreq(128, d=(len(samples) / sample_rate) / 128)
    mod_4hz_mask = (env_freqs >= 2.0) & (env_freqs <= 6.0)
    features[31] = np.clip(np.sum(env_fft[mod_4hz_mask]) / (np.sum(env_fft) + 1e-8), 0.0, 1.0)

    return features

