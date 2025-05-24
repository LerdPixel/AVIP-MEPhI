import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class VoiceAnalyzer:
    def __init__(self, src_dir: str, results_dir: str = "results"):
        self.src_dir = src_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def analyze_all(self):
        files = glob.glob(os.path.join(self.src_dir, "*.wav"))
        report_data = []

        for filepath in files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            y, sr = librosa.load(filepath, sr=None, mono=True)
            print(f"Analyzing: {name} (sr={sr}, duration={len(y)/sr:.2f}s)")

            self.plot_spectrogram(y, sr, name)
            self.plot_f0_contour(y, sr, name)

            fmin, fmax = self.get_min_max_freq(y, sr)
            f0_med, harmonics = self.estimate_f0_and_harmonics(y, sr)
            formants = self.estimate_formants(y, sr)

            self.plot_spectral_peaks(y, sr, harmonics, formants, name)

            report_data.append({
                'name': name,
                'fmin': fmin,
                'fmax': fmax,
                'f0': f0_med,
                'harmonics': harmonics,
                'formants': formants.tolist()
            })

        self.save_report(report_data)

    def plot_spectrogram(self, y: np.ndarray, sr: int, name: str):
        D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time',
                                 y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram: {name}")
        plt.tight_layout()
        self._save_plot(f"spec_{name}.png")

    def plot_f0_contour(self, y: np.ndarray, sr: int, name: str):
        f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr,
                         frame_length=2048, hop_length=512)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)

        plt.figure(figsize=(8, 4))
        plt.plot(times, f0, label="F0 contour")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"F0 Contour: {name}")
        plt.legend()
        plt.tight_layout()
        self._save_plot(f"f0_{name}.png")

    def plot_spectral_peaks(self, y, sr, harmonics, formants, name):
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        spectrum = D.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        spec_db = librosa.amplitude_to_db(spectrum, ref=np.max)

        plt.figure(figsize=(8, 4))
        plt.plot(freqs, spec_db, label="Average spectrum (dB)")

        for i, h in enumerate(harmonics, start=1):
            plt.axvline(h, linestyle="--", label=f"Harmonic {i}: {h:.1f} Hz")

        for i, f in enumerate(formants, start=1):
            plt.axvline(f, color="red", linestyle=":", label=f"Formant {i}: {f:.1f} Hz")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title(f"Spectral Peaks: {name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._save_plot(f"peaks_{name}.png")

    def get_min_max_freq(self, y, sr, threshold_db=-60):
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        avg_spectrum = S_db.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        mask = avg_spectrum > threshold_db
        if not mask.any():
            return 0.0, 0.0

        return freqs[mask].min(), freqs[mask].max()

    def estimate_f0_and_harmonics(self, y, sr, fmin=50, fmax=800):
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                         frame_length=2048, hop_length=512)
        f0_median = np.nanmedian(f0)

        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        spectrum = D.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        harmonics = []
        k = 1
        while True:
            harmonic_freq = f0_median * k
            if harmonic_freq >= sr / 2:
                break
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            if spectrum[idx] > 0.5 * spectrum.max():
                harmonics.append(harmonic_freq)
            k += 1
        return f0_median, harmonics

    def estimate_formants(self, y, sr, n_formants=3, lpc_order=16):
        a = librosa.lpc(y, order=lpc_order)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]
        angles = np.angle(roots)
        freqs = sorted(angles * sr / (2 * np.pi))
        return np.array(freqs[:n_formants])

    def save_report(self, data):
        report_path = os.path.join(self.results_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(f"File: {item['name']}\n")
                f.write(f"Min freq: {item['fmin']:.1f} Hz, Max freq: {item['fmax']:.1f} Hz\n")
                f.write(f"Fundamental (median): {item['f0']:.1f} Hz\n")
                f.write("Overtones: " + ", ".join(f"{h:.1f}" for h in item['harmonics']) + "\n")
                f.write("Formants: " + ", ".join(f"{f:.1f}" for f in item['formants']) + " Hz\n")
                f.write("\n")

    def _save_plot(self, filename):
        plt.savefig(os.path.join(self.results_dir, filename), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    analyzer = VoiceAnalyzer("voices", "results")
    analyzer.analyze_all()
