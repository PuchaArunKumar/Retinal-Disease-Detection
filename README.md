project:
  title: "Retinal Disease Detection using CLIP + DenseNet Fusion"
  type: "Deep Learning Project"
  task: "Diabetic Retinopathy Grading"

description: |
  This project implements a dual-feature deep learning system for
  diabetic retinopathy classification. The architecture fuses CLIP
  (global semantic features) and DenseNet121 (local lesion features)
  to achieve high accuracy and strong clinical reliability.

features:
  fusion_model:
    name: "CLIP + DenseNet121 Fusion"
    details:
      - Dual feature extraction pipelines
      - Combines semantic and structural retinal features
      - More accurate than single backbone CNNs
      - Robust to lighting & image quality variations

  multi_class_classification:
    classes:
      - "0 : No DR"
      - "1 : Mild"
      - "2 : Moderate"
      - "3 : Severe"
      - "4 : Proliferative DR"
    notes:
      - Softmax multi-class disease grading

  explainability:
    method: "Grad-CAM"
    focuses_on:
      - Microaneurysms
      - Hemorrhages
      - Exudates
      - Vascular abnormalities

results:
  overall:
    accuracy: 85.45
    macro_f1: 0.73
    weighted_f1: 0.85
    qwk: 0.92

  class_wise:
    - class: "0 - No DR"
      precision: 0.9682
      recall: 0.9861
      f1: 0.9771

    - class: "1 - Mild"
      precision: 0.6667
      recall: 0.5778
      f1: 0.6190

    - class: "2 - Moderate"
      precision: 0.7429
      recall: 0.8595
      f1: 0.7969

    - class: "3 - Severe"
      precision: 0.6667
      recall: 0.4348
      f1: 0.5263

    - class: "4 - Proliferative"
      precision: 0.8846
      recall: 0.6571
      f1: 0.7541

baseline_models:
  - model: "CLIP + DenseNet (Proposed)"
    performance: "85.45% Accuracy, QWK = 0.92"

  - model: "DenseNet121"
    performance: "97.30% Accuracy"

  - model: "AlexNet"
    performance: "97.90% Accuracy"

  - model: "ResNet50"
    performance: "85.28% Accuracy"

  - model: "EfficientNet-B3"
    performance: "~0.821 QWK"

  - model: "Vision Transformer (ViT)"
    performance: "98.79% Accuracy"

dataset:
  name: "APTOS 2019 Blindness Detection"
  url: "https://www.kaggle.com/competitions/aptos2019-blindness-detection/"
  details:
    - Real retinal fundus images
    - 5-class DR severity labels
    - Raw clinical image variations (illumination, focus, blur)
  preprocessing:
    - Resize and normalization
    - Rotation
    - Brightness adjustment
    - Gaussian noise
    - Random cropping

architecture:
  clip_encoder:
    description: |
      Extracts global semantic features and retains robustness to
      noise, illumination, and camera variations.

  densenet121:
    description: |
      Captures fine-grained retinal abnormalities such as
      microaneurysms, hemorrhages, exudates, and vessel distortions.

  fusion_layer:
    description: |
      Concatenates CLIP and DenseNet embeddings and aligns the dimensions
      using a projection layer to form a unified representation.

  classifier:
    description: "Fully connected neural layers with softmax output."

training_pipeline:
  stage_1:
    description: "Freeze CLIP and DenseNet; train only classifier + projection layers."

  stage_2:
    description: "Fine-tune DenseNet; selectively unfreeze CLIP; apply lower learning rate."

  hyperparameters:
    optimizer: "Adam"
    learning_rate_initial: 1e-4
    learning_rate_finetune: 1e-5
    batch_size: 16
    loss_function: "CrossEntropy"

usage:
  steps:
    - "Clone the repository"
    - "Install dependencies"
    - "Place dataset inside /data/"
    - "Run training notebook or training script"
    - "Evaluate using evaluation script"
    - "Generate Grad-CAM visualizations"
    - "Predict on custom images"

future_work:
  - Add glaucoma and AMD detection
  - Improve model lightweight deployment for mobile screening
  - Integrate advanced attention-based explainability
  - Expand dataset diversity
  - Add OOD detection for safety

