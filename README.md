# Skyland - Optimal Drone Landing System 🚁🌍

![Drone Landing](https://img.shields.io/badge/Status-Active-brightgreen) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-orange)

Hi Again, This is Santhosh, your friendly neighbourhood coder. Alright, let’s do this one last time - this is Skyland.
Welcome to **Skyland**, an innovative project designed to enable safe and optimal drone landings using **semantic segmentation** of aerial images. This system leverages state-of-the-art deep learning models to identify safe landing zones while avoiding obstacles such as buildings, trees, and other hazards.

---

## 📌 Overview

Skyland is a deep learning-based system that processes aerial images to segment terrain into safe and unsafe zones for drone landings. The project uses **semantic segmentation models** like **UNet** and **PSPNet**, trained on the **Semantic Drone Dataset**, to accurately classify terrain types. The system also includes an algorithm to autonomously identify the safest landing spots based on the segmentation results.

### Key Features:
- **Semantic Segmentation**: Accurately classifies terrain into 24 classes using deep learning models.
- **Safe Landing Algorithm**: Identifies optimal landing zones by filtering out unsafe areas like buildings, trees, and people.
- **Model Comparison**: Evaluates multiple models (Manual UNet, Fine-Tuned UNet, and PSPNet) to determine the best-performing architecture.
- **Data Preparation**: Includes data visualization, augmentation, and preprocessing steps for robust model training.

---

## 🛠️ Models and Performance

We experimented with three deep learning models for semantic segmentation:

1. **Manual UNet**:
   - Fully convolutional network with encoder-decoder architecture.
   - **Test Accuracy**: 79.35%

2. **Fine-Tuned UNet**:
   - UNet with a **ResNet-50 backbone** pre-trained on ImageNet.
   - **Test Accuracy**: 85.99% (Best Performing Model)

3. **Fine-Tuned PSPNet**:
   - A state-of-the-art semantic segmentation model pre-trained on ImageNet.
   - **Test Accuracy**: 81.13%

The **Fine-Tuned UNet** outperformed the other models and was selected as the final model for the Skyland system.

---

## 🚀 How It Works

1. **Data Preparation**:
   - Aerial Semantic Segmentation Drone Dataset a freely available dataset that can be downloaded via kaggle - [https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset]
   - Data visualization, augmentation, and preprocessing are handled within the code.

2. **Model Training**:
   - The models are trained on the dataset to segment aerial images into 24 terrain classes.
   - Metrics like accuracy, IoU (Intersection over Union), and loss are tracked during training.

3. **Safe Landing Algorithm**:
   - The segmented image is processed to identify safe landing zones.
   - Safe classes (e.g., paved areas, grass) are prioritized, while avoid classes (e.g., roofs, trees, people) are excluded.

4. **Output**:
   - The system outputs a map highlighting safe landing zones for the drone.

---

## 📊 Results

- **Fine-Tuned UNet** achieved the highest accuracy of **85.99%**.
- Safe landing zones are accurately identified, avoiding obstacles like buildings and trees.
- Visualizations of segmented images and safe zones are provided in the `results/` folder.

---

## 🚀 Future Work

- Train models for higher epochs to improve accuracy.
- Integrate real-time drone landing capabilities.
- Expand the dataset to include more diverse terrains and conditions.

---

## 📧 Contact

With great code comes great responsibility... this is my gift, my curse. Who am I? Your friendly neighbourhood coder.
- **Santhosh M**  
- **Email**: santhoshmayilraj@gmail.com  
- **GitHub**: [santhoshmayilraj](https://github.com/santhoshmayilraj)  

---

**Skyland** is your go-to solution for safe and efficient drone landings. Let's take drone technology to new heights! 🚀
```

---
