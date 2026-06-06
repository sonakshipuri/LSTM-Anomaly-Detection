# LSTM Autoencoder for Industrial Anomaly Detection

An unsupervised anomaly detection system built using an LSTM Autoencoder on the SKAB (Skoltech Anomaly Benchmark) industrial sensor dataset. The model learns normal operating behavior and identifies anomalies using reconstruction error without relying on labels during training.

## Results

| Model            | F1 Score | AUC-ROC |
| ---------------- | -------- | ------- |
| LSTM Autoencoder | 0.792    | 0.885   |
| Isolation Forest | 0.630    | 0.878   |

**Improvement over baseline:** +16.2% F1

**Best Configuration**

* Window Size: 30
* Hidden Dimension: 32
* Epochs: 50
* Threshold: 95th percentile of training reconstruction errors

---

## Dataset

SKAB (Skoltech Anomaly Benchmark) – Valve-1 subset

* 16 CSV files
* 18,162 timesteps
* 8 industrial sensor signals
* 11,853 normal samples
* 6,309 anomalous samples

Sensor Features:

* Accelerometer1RMS
* Accelerometer2RMS
* Current
* Pressure
* Temperature
* Thermocouple
* Voltage
* Volume Flow RateRMS

---

## Model Architecture

Input Sequence (30 × 8)

↓ Encoder

2-Layer LSTM (Hidden = 32)

↓ Latent Representation

Final Hidden State

↓ Decoder

2-Layer LSTM

↓ Output Layer

Linear Layer (32 → 8)

↓ Reconstruction

Reconstructed Sensor Sequence

Anomaly Score = Mean Squared Reconstruction Error

---

## Methodology

1. Load and concatenate all Valve-1 files.
2. Fit MinMaxScaler using only normal samples.
3. Generate sliding windows from the time-series data.
4. Train the LSTM Autoencoder exclusively on normal windows.
5. Compute reconstruction errors on all windows.
6. Use the 95th percentile of training errors as the anomaly threshold.
7. Compare performance against an Isolation Forest baseline.

---

## Hyperparameter Search

Grid Search:

* Window Sizes: {10, 20, 30}
* Hidden Dimensions: {32, 64}

Best Result:

| Window | Hidden | F1    | AUC   |
| ------ | ------ | ----- | ----- |
| 30     | 32     | 0.778 | 0.877 |

The selected configuration was retrained for 50 epochs to obtain the final reported metrics.

---

## Key Findings

* Reconstruction errors for anomalous windows were significantly higher than those of normal windows.
* Error distributions showed clear separation between train and test samples.
* The LSTM Autoencoder consistently outperformed Isolation Forest in F1 score.
* Temporal sequence modeling improved anomaly detection compared to a non-sequential baseline.

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

---

## Future Improvements

* Attention-based LSTM Autoencoders
* Variational Autoencoders (VAE)
* Transformer-based anomaly detection
* Bayesian hyperparameter optimization
* Real-time streaming anomaly detection

---

## License

MIT License
