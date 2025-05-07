# Brain Tumor Classification Using Deep Learning
## Project Overview
This project focuses on classifying brain tumors from MRI scans using deep learning techniques. Brain tumors can be difficult to identify due to their subtle differences, making accurate diagnosis essential for effective treatment planning. The goal of this project was to build a deep learning model capable of identifying and classifying brain tumors into four categories: glioma, meningioma, pituitary tumor, and healthy (no tumor).

![brain_tumor_classification_sample](https://github.com/user-attachments/assets/847f4d71-c654-4d40-800e-512f1d701863)

## Problem Statement
Brain tumors are among the leading causes of death worldwide. Early and accurate detection is crucial for improving survival rates and treatment outcomes. Traditional methods of diagnosis often rely heavily on expert radiologists who may miss early-stage tumors. This project addresses this challenge by leveraging computer vision techniques, specifically Convolutional Neural Networks (CNNs), to automate the process of classifying brain tumors from MRI images.

## Dataset
The dataset used for this project was sourced from Kaggle, specifically the "Brain Tumor MRI Dataset" (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It contains a collection of MRI scans that are labeled into four distinct classes:

- Glioma
- Meningioma
- Pituitary Tumor
- Healthy (No Tumor)

The dataset is split into training and testing sets, allowing for effective model evaluation and validation.

## Approach
### Data Preprocessing:
- All images were resized to a consistent resolution of 256x256 pixels.
- Images were normalized to a [0,1] range to ensure consistency in pixel values.
- The dataset was split into training and validation subsets with a 15% validation split.

### Data Augmentation:
- Various augmentations were applied (e.g., rotation, flipping, brightness adjustment) to simulate different variations in MRI scans and reduce overfitting.

### Model Architecture:
- The project utilized ResNet50V2, a pre-trained deep learning model, leveraging its feature extraction capabilities.
- Additional layers, including Global Average Pooling, Dense layers, and Softmax activation, were added for multi-class classification.

### Training:
- The model was trained with techniques like early stopping, learning rate scheduling, and model checkpointing to ensure efficient training and prevent overfitting.
- L1/L2 regularization and dropout layers were incorporated for better generalization.

### Evaluation:
- The model achieved 97.3% test accuracy.
- The classification report showed impressive precision, recall, and F1-scores across all tumor categories.

![brain_tumor_classification_confusion_matrix](https://github.com/user-attachments/assets/2051b3bc-41b8-47a7-9045-79a773173e45)

## Conclusion
The model demonstrated outstanding performance in classifying brain tumor types from MRI images, making it a promising tool for supporting medical professionals in the diagnostic process. The project highlights the potential of deep learning in healthcare and can be adapted for use in clinical environments to provide fast and reliable tumor classification.

### Future Improvements
- Multi-modal Data: Incorporating additional modalities like CT scans or functional MRIs could further enhance diagnostic accuracy.
- External Validation: Testing the model on external datasets to evaluate its real-world performance and generalization.
- Model Optimization: Experimenting with different architectures or optimizing hyperparameters for even higher accuracy.
