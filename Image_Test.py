import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define disease classes
classes = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "No findings"
]

# Augmentation pipeline (from main file)
main_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


# Function to load the trained model
def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ckpt['model']
    model.eval().to(device)
    print("Model loaded successfully!")
    return model


# Function to preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


# Function to predict diseases
def predict_diseases(model, image_tensor, classes):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    print("\nPredictions:")
    for cls, prob in zip(classes, probabilities):
        status = "Positive" if prob > 0.50 else "Negative"
        print(f"{cls}: Probability: {prob:.4f} ({status})")


# Function to visualize transformations
def visualize_transformations(image_path, transform):
    original_image = Image.open(image_path).convert("RGB")
    transformed_image_tensor = transform(original_image)

    # Convert tensor back to displayable format
    transformed_image_np = transformed_image_tensor.permute(1, 2, 0).numpy() * 255  # Convert to [H, W, C]
    transformed_image_np = transformed_image_np.astype("uint8")

    # Visualize using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image_np)
    plt.title("Transformed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Paths
CHECKPOINT_PATH = "models/stage5_1e-06_22.pth"  # Update with your checkpoint path
IMAGE_PATH = r"D:\Multi_label_NIH\sample_xrays\Atelectasis.png"  # Path to your sample image
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # Load the model
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # Preprocess the image
    print("\nPreprocessing image...")
    image_tensor = preprocess_image(IMAGE_PATH, main_pipeline).to(DEVICE)

    # Predict diseases
    print("\nPredicting diseases...")
    predict_diseases(model, image_tensor, classes)

    # Visualize transformations
    print("\nVisualizing transformations...")
    visualize_transformations(IMAGE_PATH, main_pipeline)
