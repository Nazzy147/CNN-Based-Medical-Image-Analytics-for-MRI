# 🧠 CNN-Based Medical Image Analytics for MRI

### 🚀 Autoencoder-Driven Anomaly Detection with Confidence Scoring & Drift Monitoring

A complete, production-minded deep learning pipeline for **MRI image analytics** using **CNN Autoencoders** — designed for **unsupervised anomaly detection**, **confidence-aware predictions**, and **model drift monitoring**.

This project showcases strong end-to-end ML and MLOps capability, including:

✔ deep learning architecture design
✔ medical image preprocessing
✔ confidence scoring
✔ drift simulation and monitoring
✔ reproducible experimentation


## 🌟 Impact Highlights (with real numbers)

📊 **Dataset Size:** 690 MRI images
🧠 **Model Type:** CNN Autoencoder

From the notebook training runs:

* ⚡ **30-epoch training completed**
* 📉 **Reconstruction loss reduced from 0.0985 → 0.0509**
* 🔻 **≈ 48.3% reduction in training loss**
* 🎯 Stable validation loss around **0.0509**
* 🧭 **Anomaly threshold learned:** ~ **0.0713**

These demonstrate:

👉 successful representation learning
👉 stable generalization behavior
👉 usable anomaly score thresholding

---

## 🔍 Problem

MRI diagnostics require:

* expert interpretation
* time-intensive manual review
* high accuracy under pressure

Subtle anomalies can be missed — this project builds an **assistive AI system** capable of:

✨ learning normal anatomy
✨ flagging anomaly-like deviations
✨ attaching confidence values to decisions

---

## 🧩 Solution Overview

### 🧠 1. CNN Autoencoder for Anomaly Detection

Learns latent-space representations of **healthy MRI structure**.

* Normal image → low reconstruction error
* Abnormal image → high reconstruction error

### 🎯 2. Confidence Scoring

Each prediction is enriched with:

* reconstruction-based anomaly score
* calibrated confidence score
* “borderline vs strong anomaly” interpretation

### 🌪 3. Drift Simulation & Monitoring

The system simulates & tracks:

* covariate drift (brightness, noise, contrast)
* scanner distribution shift
* anomaly frequency changes
* concept drift via unseen structures

Monitors include:

* loss distribution movement
* anomaly rate changes
* metric degradation trends
* alert thresholds

📈 This highlights **real-world reliability thinking** — not just a one-off model.

---

## 🛠 Tech Stack

🐍 Python
🧠 TensorFlow / Keras
🔬 OpenCV
📊 NumPy / Pandas
📉 Matplotlib
🧮 Scikit-Learn
⚙ Optional: MLflow / W&B-style monitoring design

---

## 🧭 End-to-End Pipeline

✔ Data ingestion & validation
✔ MRI normalization & resizing
✔ Train–validation split strategy
✔ CNN autoencoder architecture design
✔ Training with callbacks
✔ Latent-space feature learning
✔ Anomaly score calculation
✔ **confidence scoring module**
✔ **drift simulation**
✔ **monitoring dashboards & alerts**
✔ Visualization for clinical interpretability

---

## 🧱 Model Architecture 

### Encoder

* stacked Conv2D layers
* ReLU activations
* MaxPooling down-sampling
* latent bottleneck representation

### Decoder

* Conv2DTranspose layers
* spatial reconstruction
* sigmoid output

Demonstrates mastery of:

* receptive field design
* bottleneck representation learning
* encoder–decoder symmetry
* image reconstruction dynamics

---

## 📊 Evaluation & Monitoring

Primary evaluation signals:

* mean squared reconstruction loss
* error histogram & distribution
* anomaly threshold at **≈ 0.0713**
* before/after image reconstruction comparison

Confidence score is derived from:

* z-score of reconstruction error
* calibration against training distribution
* distance in latent feature space

Output categories:

| Outcome                    | Meaning                                 |
| -------------------------- | --------------------------------------- |
| ✅ High-confidence normal   | very low reconstruction error           |
| ⚠ Medium confidence        | borderline reconstruction               |
| 🚨 High-confidence anomaly | strong deviation from learned normality |

---

## 🌪 Drift Simulation

Simulated drifts include:

* Gaussian noise injection
* blur & artifact simulation
* contrast/brightness shift
* dataset composition shift
* anomaly frequency variation

Tracked metrics include:

* change in anomaly rate
* mean reconstruction error shift
* variance widening
* threshold instability

This mirrors **post-deployment monitoring** in real medical AI systems 🏥

---

## 📁 Repository Structure

```
📂 data/                         # MRI dataset
📂 models/                       # Saved autoencoder weights
📂 drift_simulation/             # Drift experiments & scripts
📂 monitoring_reports/           # Plots & logs
📓 CNN_Based_Medical_Image_Analytics_for_MRI.ipynb
```

---

## ▶️ How to Run Locally

Clone:

```
git clone <your-repo-url>
cd <project-folder>
```

Open the notebook:

```
CNN_Based_Medical_Image_Analytics_for_MRI.ipynb
```

Install dependencies:

```
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
pandas
```

GPU highly recommended ⚡

---

## 🎯 What This Project Demonstrates About My Skills

* Deep learning for computer vision 🧠
* Autoencoders & representation learning
* Unsupervised anomaly detection
* Confidence & uncertainty estimation
* Data and concept drift handling
* ML monitoring & lifecycle thinking
* Clear experiment design
* Clean ML engineering practices

---

## 🚀 Future Roadmap

🔮 Variational Autoencoder (VAE)
🩻 3D MRI volume modeling
🧭 Self-supervised contrastive pretraining
📈 ROC-AUC benchmark vs baselines
🖥 Streamlit inference UI for clinicians
🔗 PACS system integration concepts

---

## 👤 Author

**Mamta Nasreen**

---
