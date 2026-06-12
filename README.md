# crop-disease-detector
Federated multimodal ML framework for cross-regional rice leaf blast detection. Fuses MobileNetV2 visual features with climate data (temp, humidity, rainfall). Trained on Punjab, generalizes to Kerala with 94.8% F1. Includes ablation study, SHAP explainability &amp; Grad-CAM.
## Overview

This project presents a federated multimodal machine learning framework 
for detecting Rice Leaf Blast disease across geographically distinct 
regions of India.

Unlike standard image classifiers, this system fuses:
- 🌾 **Visual features** extracted via MobileNetV2 (1280-dim embeddings)
- 🌦️ **Climate data** (temperature, humidity, rainfall) — synthetically 
  generated using literature-grounded agro-meteorological parameters 
  from Punjab and Kerala

The core research problem: models trained in one region often *fail* 
when deployed in another due to climate-induced distribution shift. 
This project solves that using **Federated Learning**.

## Key Results

| Model | Local F1 | Cross-Regional F1 |
|---|---|---|
| Vision-Only | ~99% | 86.3% |
| Climate-Only | High | 50.0% (collapses) |
| Multimodal Fusion | 100% | 86.3% |
| **Federated Global ★** | **99.2%** | **94.8%** |

## Pipeline

1. Dataset partitioned into two regional splits (Punjab & Kerala)
2. MobileNetV2 used headless for deep visual feature extraction
3. Climate features injected using region/label-specific distributions
4. Ablation study comparing Vision-Only vs Climate-Only vs Fusion
5. Federated Learning aggregates regional models into a global model
6. Evaluated with SHAP explainability + Grad-CAM visualizations

## Tech Stack
`Python` `TensorFlow` `Scikit-learn` `MobileNetV2` `Random Forest` 
`Federated Learning` `SHAP` `Grad-CAM` `Google Colab` `Kaggle API`
