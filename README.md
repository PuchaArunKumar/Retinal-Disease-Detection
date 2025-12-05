🩺 Retinal Disease Detection using CLIP + DenseNet Fusion
Deep Learning Project – Diabetic Retinopathy Grading

This repository contains an advanced deep learning model for automatic Retinal Disease Detection, specifically focusing on Diabetic Retinopathy (DR) severity classification.

The system integrates CLIP (for global semantic feature extraction) and DenseNet121 (for fine-grained retinal lesion detection) into a fusion-based architecture, achieving high accuracy and strong clinical reliability.

🚀 Features
✅ Fusion Model (CLIP + DenseNet121)

Dual feature extraction pipelines

Combines global semantic and local structural information

Improved accuracy over single-backbone models

More robust to lighting and image quality variations

✅ Multi-Class DR Severity Classification

Classifies images into 5 DR levels:

0: No DR

1: Mild

2: Moderate

3: Severe

4: Proliferative DR

Softmax multi-class prediction

✅ Explainability (Grad-CAM)

Identifies clinically meaningful areas such as:

Microaneurysms

Hemorrhages

Exudates

Vascular abnormalities

Enhances interpretability and supports medical validation

✅ Strong Performance Metrics

85.45% Accuracy

Quadratic Weighted Kappa (QWK): 0.92

Detailed class-wise precision/recall/F1 scores

Confusion matrix for performance visualization

✅ Baseline Comparison

Models compared against the proposed hybrid model:

Model	Accuracy / QWK
CLIP + DenseNet Fusion	85.45%, QWK = 0.92
DenseNet121	97.30%
AlexNet	97.90%
ResNet50	85.28%
EfficientNet-B3	~0.821 QWK
Vision Transformer	98.79%
📊 Results Summary
Overall Performance
Metric	Score
Accuracy	85.45%
Macro F1-Score	0.73
Weighted F1-Score	0.85
Quadratic Weighted Kappa (QWK)	0.92
Class-Wise Metrics
Class	Precision	Recall	F1-Score
No DR (0)	0.9682	0.9861	0.9771
Mild (1)	0.6667	0.5778	0.6190
Moderate (2)	0.7429	0.8595	0.7969
Severe (3)	0.6667	0.4348	0.5263
Proliferative (4)	0.8846	0.6571	0.7541
📁 Dataset

We use the APTOS 2019 Blindness Detection dataset.

🔗 https://www.kaggle.com/competitions/aptos2019-blindness-detection/

Dataset Details

5 DR severity levels

Real-world retinal fundus images

Varying resolutions and lighting conditions

Preprocessing includes:

Resizing & normalization

Brightness adjustments

Rotation

Noise addition

Random cropping

Class imbalance handling

🧠 Model Architecture
### 1️⃣ CLIP Encoder (Global Feature Extraction)

Learns semantic-level representations

Handles illumination, color, and device variations

Strengthens global contextual understanding

2️⃣ DenseNet121 (Local Lesion Extraction)

Focuses on fine-grained retinal abnormalities:

Microaneurysms

Hemorrhages

Hard/soft exudates

Blood vessel distortions

3️⃣ Feature Fusion Layer

Concatenates embeddings from CLIP & DenseNet

Projection layer aligns feature dimensions

Produces unified representation for classification

4️⃣ Classification Head

Fully connected neural layers

Softmax output for 5 classes

Trained using CrossEntropy loss

5️⃣ Explainability (Grad-CAM)

Highlights important disease regions

Verifies clinical relevance of predictions

Builds trust for medical professionals

🧪 Training Pipeline
Stage 1: Frozen Encoder Training

Freeze CLIP & DenseNet

Train classification + projection layers

Stage 2: Fine-Tuning

Unfreeze DenseNet121

Selectively unfreeze CLIP layers

Gradual learning-rate reduction

Hyperparameters

Optimizer: Adam

Learning rate: 1e-4 → 1e-5

Batch size: 16–32

Loss function: CrossEntropy
