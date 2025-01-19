# Lane Detection Using PyTorch

## 🚀 Project Overview

This project focuses on **Lane Detection**, a critical application in **autonomous vehicles**. Leveraging the **DeepLabV3 architecture with ResNet101** backbone and a custom **MiniLaneDetectionModel**, the project demonstrates a robust and scalable approach to detecting lanes in road images. 

With seamless integration of the **Roboflow platform** for dataset management and preprocessing, the project showcases advanced techniques in **deep learning, transfer learning, and model deployment** using **Gradio**.

---

## 📂 Dataset Overview

### Dataset Details
The dataset used in this project is associated with the **"lane_detection-rd6mu" project** on **Roboflow**. It contains images annotated for lane detection tasks, split into training, validation, and test sets. 

- **Dataset Source**: [Download the Dataset](https://unhnewhaven-my.sharepoint.com/:f:/g/personal/vpada4_unh_newhaven_edu/EmdcQizHx7dEnx2JGc_mEM8BlSt4ifzzq1rLI47YkbVAkw?e=BMkoUa)
- **Key Features**:
  - Integration with Roboflow API for streamlined dataset preprocessing.
  - Custom PyTorch-compatible dataset using the `ImageFolder` class from `torchvision`.

---

## 🔧 Project Architecture

### 1. **Lane Detection Model**
- **Model**: Pre-trained **DeepLabV3** architecture with **ResNet101** backbone.
- **Key Features**:
  - Transfer learning for leveraging features from diverse datasets.
  - Custom output layer tailored for lane detection tasks.
  - Capable of extracting intricate features for high accuracy.

### 2. **MiniLaneDetectionModel**
- **Model**: A compact neural network for lane detection, designed for simplicity and efficiency.
- **Key Features**:
  - Feature extraction module: Two convolutional layers with max pooling.
  - Classification module: Two fully connected layers with ReLU activation.
  - Debugging: Intermediate outputs printed for better understanding of internal dimensions.

---

## 🧪 Model Training and Evaluation

### Training Details
- **Loss Function**: Mean Squared Error (MSE) for accurate lane position predictions.
- **Optimizer**: Adam optimizer for efficient weight updates.
- **Data Augmentation**: Applied resizing and random horizontal flipping for robustness.
- **Learning Rate Tuning**:
  - Experimented with learning rates (`0.01`, `0.1`).
  - Utilized StepLR scheduler to adjust learning rate during training.

### Evaluation Metrics
1. **Regression Metrics**:
   - **Mean Squared Error (MSE)**: Measures average squared differences.
   - **Mean Absolute Error (MAE)**: Quantifies average absolute differences.
2. **Binary Classification Metrics**:
   - **F1 Score**: Balances precision and recall for lane presence detection.
   - **Accuracy**: Measures overall correctness of predictions.

#### Results:
| **Metric**         | **Value**  |
|---------------------|------------|
| Mean Squared Error  | 0.4034     |
| Mean Absolute Error | 0.4735     |
| F1 Score            | 0.9333     |
| Accuracy            | 0.8750     |

---

## 🖥️ Deployment with Gradio

The project employs **Gradio** to deploy the lane detection model, creating an interactive web interface for real-time lane detection:

- Users can upload images to visualize detected lanes overlaid on the original image.
- The **Gradio interface** uses binary segmentation masks to highlight detected lanes.

### How to Run:
1. Install dependencies using the provided `requirements.txt`.
2. Run the Gradio app:
   ```bash
   python app.py
##📝 Key Features
- Robust Dataset Preparation:
  - Seamless integration with Roboflow API for preprocessing and dataset splitting.
- Versatile Models:
  - Advanced architecture (DeepLabV3 + ResNet101).
  - Lightweight custom model for efficient lane detection.
- Hyperparameter Tuning:
  - Experimented with learning rates, schedulers, and augmentations.
- Comprehensive Evaluation:
  - Regression and classification metrics ensure robust performance insights.
- Interactive Deployment:
  - Real-time lane detection using Gradio.

##🛠️ Tools and Technologies
- Framework: PyTorch
- Preprocessing: Roboflow API
- Deployment: Gradio
- Data Augmentation: torchvision
- Metrics: MSE, MAE, F1 Score, Accuracy
- Model Architectures: DeepLabV3 with ResNet101, Custom MiniLaneDetectionModel

##📈 Future Work
- Integrate additional datasets for improved model generalization.
- Explore more advanced architectures like Transformer-based models.
- Optimize deployment for edge devices.


Copy and save this content in a `README.md` file in your GitHub repository. Let me know if you'd like further adjustments!
