To use the NIH Chest X-ray dataset with this project, ensure the dataset files are placed inside a folder named data, located in the root of the project directory.
This folder should contain the Data_Entry_2017.csv file (the metadata file for the dataset) and a subfolder named images with all the X-ray image files.
Alternatively, you can import the dataset programmatically using the kagglehub library. Simply execute the following code snippet to download the latest version of the dataset:

---

# Disease_Detection_From_ChestXRays

Multi-Label Disease Classification of Chest X-Rays using ResNet50 in Pytorch.

---

# Requirements

- Python >= 3.8
- torch >= 1.9
- torchvision >= 0.10
- opencv-python >= 4.5
- numpy >= 1.21
- matplotlib >= 3.4
- tqdm >= 4.62
- scikit-learn >= 0.24

To install the dependencies, run:

```
pip install -r requirements.txt
```

---

# Dataset

[NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/data#Data_Entry_2017.csv) is used for Multi-Label Disease Classification of Chest X-Rays.  
There are a total of 15 classes (14 diseases and one for 'No findings').  
Images can be classified as "No findings" or one or more disease classes:

- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural_thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia

The dataset contains **112,120 X-ray images** of size 1024x1024 pixels:  
**86,524 images** for training and **25,596 images** for testing.

---

# Sample X-Ray Images

<div class="row">
  <div class="column">
    <img src='/sample_xrays/Fibrosis.png' width='250' alt='Fibrosis' hspace='15'>
  </div>
  <div class="column">
    <img src='/sample_xrays/Atelectasis.png' width='250' alt='Atelectasis' hspace='15'>
  </div>
  <div class="column">
    <img src='/sample_xrays/No Finding.png' width='250' alt='No Finding' hspace='15'>
  </div>
</div>

---

# Model

The **ResNet50** pretrained model is used for Transfer Learning on this dataset. Different layers of the ResNet50 model are trained or fine-tuned in different stages.

---

# Loss Function

Two loss functions are supported:

- **Focal Loss** (default)
- **Binary Cross Entropy Loss (BCE Loss)**

---

# Training

### Training Modes

1. **Training from Scratch**
   Following layers are trainable in **Stage 1**:

   - layer2
   - layer3
   - layer4
   - fc

   Command to start training from scratch:

   ```
   python main.py --data_path data --bs 128 --lr 1e-5 --stage 1
   ```

2. **Resuming from a Saved Checkpoint**
   Resume training from a specific checkpoint by providing the checkpoint path using the `--ckpt` argument and specifying the stage using the `--stage` argument.

   Example command to resume training:

   ```
   python main.py --resume --data_path data --ckpt stage5_1e-06_22.pth --stage 4 --lr 1e-3
   ```

3. **Training in Stage 5**
   Stage 5 is used to fine-tune the entire model with a low learning rate for improved recognition. All layers are trainable.

   Command to start Stage 5 training:

   ```
   python main.py --resume --data_path data --ckpt stage4_checkpoint.pth --stage 5 --lr 1e-6
   ```

Training checkpoints and loss plots will be saved in the `models` directory.

---

# Testing

To test the model using a specific checkpoint, use the `--test` argument along with the checkpoint path.

Command to test the model:

```
python main.py --test --data_path data --ckpt stage5_1e-06_22.pth
```

---

# Results

The model achieved an average **ROC AUC Score** of **0.73241** (excluding the "No findings" class) after training in the following stages:

#### STAGE 1

- Loss Function: FocalLoss
- lr: 1e-5
- Training Layers: layer2, layer3, layer4, fc
- Epochs: 2

#### STAGE 2

- Loss Function: FocalLoss
- lr: 3e-4
- Training Layers: layer3, layer4, fc
- Epochs: 1

#### STAGE 3

- Loss Function: FocalLoss
- lr: 1e-3
- Training Layers: layer4, fc
- Epochs: 3

#### STAGE 4

- Loss Function: FocalLoss
- lr: 1e-3
- Training Layers: fc
- Epochs: 2

#### STAGE 5

- Loss Function: FocalLoss
- lr: 1e-6
- Training Layers: All layers
- Epochs: 3

---

# Summary of Stages

The model is trained in different stages with varying trainable layers and learning rates for optimal performance:

- **Stage 1**: Train `layer2`, `layer3`, `layer4`, and `fc` with `lr=1e-5`
- **Stage 2**: Train `layer3`, `layer4`, and `fc` with `lr=3e-4`
- **Stage 3**: Train `layer4` and `fc` with `lr=1e-3`
- **Stage 4**: Train `fc` with `lr=1e-3`
- **Stage 5**: Fine-tune all layers with `lr=1e-6`

Use the appropriate commands for training or testing from each stage as outlined above.

---

Acknowledgments
This project builds upon the original implementation:

n0obcoder, NIH Chest X-Rays Multi-Label Image Classification in PyTorch. 2023. GitHub Repository.
https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch
Accessed: 20/12/2024.
All credit for the original model design and training pipeline goes to n0obcoder.
