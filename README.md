📌 Project Overview

DeepDiabetic is a deep learning-based framework developed to automatically detect and classify multiple diabetic eye diseases from retinal fundus images. The system focuses on Diabetic Retinopathy (DR), Diabetic Macular Edema (DME), Glaucoma, and Cataract, helping support early detection and faster diagnosis.

🎯 Objectives
Automate the detection of multiple diabetic eye diseases.
Improve classification performance using image preprocessing and data augmentation.
Compare different deep learning models and select the best-performing model.
Support early disease detection and assist ophthalmologists in clinical decision-making.
📊 Dataset

The project uses 1,228 retinal fundus images collected from six public datasets. Image preprocessing and online/offline augmentation techniques were applied to improve model performance.

🧠 Models Used

The following deep learning models were evaluated:

EfficientNetB0
VGG16
ResNet152V2
GRU
Bi-GRU

EfficientNetB0 achieved the best performance among the evaluated models.

⚙️ Methodology
Retinal Fundus Images
        ↓
Dataset Preprocessing
        ↓
Data Augmentation
        ↓
Image Preprocessing
        ↓
Deep Learning Model Training
        ↓
Feature Extraction
        ↓
Disease Classification
        ↓
Model Evaluation
        ↓
Disease Prediction

The images are resized and normalized, followed by augmentation and model training. The trained models classify images into Cataract, DME, DR, and Glaucoma.

📈 Evaluation Metrics

The models are evaluated using:

Accuracy
Precision
Recall
F1-Score
Specificity
AUC
Confusion Matrix
🚀 Key Features
Multi-disease eye disease classification
Deep learning-based automated prediction
Image preprocessing and augmentation
Comparison of multiple deep learning architectures
Suitable for faster and large-scale screening
Supports early detection of diabetic eye diseases
🛠️ Technologies
Python
Deep Learning
CNN
Transfer Learning
Image Processing
Data Augmentation
EfficientNetB0
VGG16
ResNet152V2
GRU / Bi-GRU
👥 Team

Batch: F-15
Department: Computer Science & Engineering
Institution: CMR Institute of Technology, Hyderabad

👨‍💻 Team Members
N. Rajesh Kumar
V. Vivek
Y. Sindhu
📌 Future Scope

The framework can be further improved by using larger and more balanced retinal datasets, optimizing the deep learning models, and developing a practical deployment system for real-world screening.
