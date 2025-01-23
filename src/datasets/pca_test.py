import soundfile as sf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_audio(file_path):
    """Load the music from a file."""
    signal, sample_rate = sf.read(file_path)
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    return signal, sample_rate

def perform_pca(signal, n_components=None):
    """Perform PCA on the audio signal."""
    window_size = 1024
    step_size = 512
    num_windows = (len(signal) - window_size) // step_size + 1

    windows = np.array([
        signal[i * step_size: i * step_size + window_size]
        for i in range(num_windows)
    ])

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(windows)
    return pca, transformed

def find_intrinsic_dimension(eigenvalues, threshold=0.95):
    """Find the intrinsic dimension using cumulative explained variance."""
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    intrinsic_dim = np.argmax(cumulative_variance >= threshold) + 1
    return intrinsic_dim

def plot_eigenvalues(eigenvalues):
    """Plot the eigenvalues to visualize explained variance."""
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues, marker='o')
    plt.title('Eigenvalues of the Covariance Matrix')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.tight_layout()
    plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()  

if __name__ == "__main__":
    audio_file = "../../data/twinkletwinkle-44100khz.wav"
    signal, sample_rate = load_audio(audio_file)
    pca, transformed = perform_pca(signal)
    eigenvalues = pca.explained_variance_
    intrinsic_dim = find_intrinsic_dimension(eigenvalues)
    print(f"Intrinsic Dimension of {audio_file}: {intrinsic_dim}")

    # Canoptionally plot the eigenvalues
    # plot_eigenvalues(eigenvalues)



    