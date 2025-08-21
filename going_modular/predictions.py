import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predict on a target image with a target model

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: str,
                        image_size: tuple = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device ,
                        model_name : str = "put your model name here"): # Added title parameter

    """Makes a prediction on a target image and plots the image with its prediction.

    Args:
        model (torch.nn.Module): A trained PyTorch model for making predictions.
        image_path (str): Filepath to the target image to make a prediction on.
        class_names (list): A list of class names for the model.
        image_size (tuple, optional): Size to resize the input image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on the input image. Defaults to None.
        device (torch.device, optional): Device to perform the inference on. Defaults to "cuda" if available, otherwise "cpu".
        title (str, optional): Title to add to the plot. Defaults to None.
    """

    # 1. Load in image and convert the DataLoader's transform (if it exists) to be able to work on a single image
    img = Image.open(image_path)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 2. Make sure the model is on the target device
    model.to(device)

    # 3. Turn on inference mode
    model.eval()

    with torch.no_grad():
        # 4. Transform and add an extra dimension to image (model requires batch_size, color_channels, height, width)
        transformed_image = image_transform(img).unsqueeze(dim=0).to(device)

        # 5. Make a forward pass on the transformed image
        target_image_pred = model(transformed_image)

    # 6. Get the predicted class labels
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 7. Plot the image and prediction
    plt.figure()
    plt.imshow(img)
    plt.suptitle(f"{model_name} | Pred: {class_names} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)