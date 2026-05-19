# Joint-Recognition-of-Seismic-Horizon-and-Fault-Based-on-DINOv2-Foundation-Model-Fine-Tuning
Utilizing the DINOv2 model for intelligent interpretation of seismic horizons, faults and structure framework
china university of petroleum (East chian), school of GeoScience, Qingdao, China 


🌍 Seismic Interpretation with Foundation Models

This repository presents a workflow for adapting pre-trained foundation models to the field of seismic interpretation.

🛠️ Data Preparation

To prepare the training dataset using seismic data alongside corresponding horizon and fault interpretations, follow these steps:

Construct Structural Frameworks: Use the data/make_dataset_labels.py script to process the raw data and build the structural frameworks.
Generate Datasets: Execute data/make_datasets.py to generate the actual training and testing splits required for network training.
Data Augmentation: Finally, run data_aug.py to perform data augmentation, enriching the training data to improve model robustness.

🧠 Model Architecture & Fine-tuning

Our approach leverages the power of transfer learning:

Feature Encoder: We utilize pre-trained foundation models (specifically DINOv2 in Small, Base, and Large variants) as the backbone for seismic feature extraction.
Fine-tuning: The models are fine-tuned to adapt to the unique characteristics of seismic horizons, faults, and structural frameworks.
Task-Specific Decoders: Custom decoders are designed to map the encoder features to specific objectives, enabling precise predictions for horizons, faults, and structural frameworks.

🚀 Downstream Applications

Once trained and fine-tuned, the model is deployed for various downstream seismic interpretation tasks, including:

Seismic Horizon Picking
Fault Detection
Structural Framework Modeling



🚀 Quick Start Guide

This repository provides a demonstration corresponding to the data mentioned in the paper, including synthetic seismic records and the F3 dataset. Follow the steps below to get started:

Clone the Repository

First, clone this repository to your local machine:

git clone git@github.com:xiaochou12138/Joint-Recognition-of-Seismic-Horizon-and-Fault-Based-on-DINOv2-Foundation-Model-Fine-Tuning.git

cd Joint-Recognition-of-Seismic-Horizon-and-Fault-Based-on-DINOv2-Foundation-Model-Fine-Tuning

Install Dependencies

Install the required Python packages using pip:

pip install -r requirements.txt



Download Datasets
Before running the code, you need to download the datasets. 
Please download the data from (https://drive.google.com/drive/folders/1gKgE0lJfkmGsF433mj0jMVanPNZjeJKP?usp=drive_link) and place it in the data/ folder.

Run the Code
You can run the following scripts for training, evaluation, and visualization:

Training & Evaluation
Single-task training

python demo_classification.py

Multi-task training

python demo_classification_two_tasks.py

Evaluate single-task model predictions

python evaluate_classification.py

Evaluate multi-task model predictions

python evaluate_classification_two_tasks.py

Post-processing & Analysis
Post-processing (corresponds to the method in the paper)

python manual_judge_horizon.py

💡 Interesting Features & Visualizations

We have included several scripts to help you explore the model's internal mechanics and outputs:

Feature Visualization (plt_tezhengtu.py): 
    Visualizes the feature maps directly predicted by the DINOv2 vision foundation model, helping you understand what the model "sees."
    
Structural Framework Decoupling (separation_fault_and_horizon.py): 
    This script separates the predicted structural framework to obtain clean horizon and fault results. It demonstrates the decoupling process: 
    Faults + Horizons = Structural Framework  ➡️  Structural Framework Decoupling: Faults + Horizons.
