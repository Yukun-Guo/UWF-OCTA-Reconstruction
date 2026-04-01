import torch
import numpy as np
import os
import glob
from PIL import Image
import toml
from NetModule import NetModule
import torch.nn.functional as F
from pathlib import Path
from Utils.utils import listFiles

def mat2gray(mat: np.ndarray, min_max = None, autoAjust: bool = False):
    """Normalize a matrix to the range [0, 1].
    Args:
        mat (np.ndarray): Input matrix.
        min_max (List[float], optional): Min and max values for normalization. Defaults to None.
        autoAjust (bool, optional): Whether to automatically adjust min and max. Defaults to False.
    Returns:
        Tuple[np.ndarray, List[float]]: Normalized matrix and [min, max] values used.
    """
    if min_max is not None and len(min_max) == 2:
        meanval = [float(min_max[0]), float(min_max[1])]
    elif autoAjust:
        meanval = [0.0, 1.0]
        tmpMat = mat[::60].ravel()  # Use ravel() instead of reshape(-1) - faster for contiguous arrays
        meanval[0] = float(np.percentile(tmpMat, 27.5))
        meanval[1] = float(np.percentile(tmpMat, 99.5))
    else:
        meanval = [float(mat.min()), float(mat.max())]
    
    # Avoid division by zero
    range_val = meanval[1] - meanval[0]
    if range_val < 1e-10:
        # Constant image - return zeros
        return np.zeros(mat.shape, dtype=np.float32)
    
    # Combine clipping and normalization in fewer operations
    # This is faster than separate clip + subtract + divide
    outMat = np.clip((mat - meanval[0]) / range_val, 0.0, 1.0)
    return outMat.astype(np.float32)

def remove_motion_artifact(img,axis=1):
    """Remove motion artifacts from an image by subtracting the mean across a specified axis.
        This is useful for OCT/OCTA images where motion artifacts can cause intensity variations along certain dimensions.
    Args:
        img (np.ndarray): Input image. range should be [0,255]
        axis (int, optional): Axis along which to compute the mean. Defaults to 1.
    """
    meanv = np.mean(img, axis=axis)  # Average across the specified dimension to reduce motion artifacts
    stdv = np.std(img, axis=axis)  # Standard deviation across the specified dimension
    meanv = np.expand_dims(meanv, axis=axis)  # Expand dimensions to match the original image shape
    stdv = np.expand_dims(stdv, axis=axis)  # Expand dimensions to match the original image shape
    tmp = (img - meanv) / (stdv + 1e-8)  # Subtract the average and normalize by standard deviation
    result = mat2gray(tmp, autoAjust=True) * 255  # Normalize to [0, 255]
    return result

def octa_composite(octa_enface, ilm_enface, alpha=1.0):
    """Composite OCTA enface image with ILM enface image using alpha Composite.
        This is useful for visualizing both vascular and structural information in a single image.
    Args:
        octa_enface (np.ndarray): OCTA enface image (grayscale) range should be [0,255]
        ilm_enface (np.ndarray): ILM enface image (grayscale).range should be [0, 1]
        alpha (float, optional): composite factor for the OCTA image. Value>0 . Defaults to 1.0.    
    Returns:
        np.ndarray: Composite RGB image combining OCTA and ILM information.
    """
    # Ensure both images are in the same shape
    if octa_enface.shape != ilm_enface.shape:
        raise ValueError("OCTA and ILM enface images must have the same shape.")
    
    im_ = (1+alpha)*octa_enface-octa_enface*ilm_enface
    composite = mat2gray(im_, autoAjust=True) * 255
    
    return composite

def load_config(config_path="config.toml"):
    """Load configuration from TOML file"""
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


def preprocess_image(image_path, ilm_path, target_size):
    """Load and preprocess a single image for model inference.

    Args:
        image_path (str): Path to the input image.
        ilm_path (str): Path to the ILM image.
        target_size (tuple): Target (H, W) taken from config image_shape.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, H, W).
    """
    img = Image.open(image_path).convert('L')
    ilm = Image.open(ilm_path).convert('L')
    img_array = np.array(img, dtype=np.float32) 
    ilm_array = np.array(ilm, dtype=np.float32) / 255.0
    img_array= octa_composite(remove_motion_artifact(img_array), ilm_array, alpha=1.0)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)/ 255.0
    return img_tensor


def load_model_from_checkpoint(checkpoint_path, config):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model from checkpoint
    model = NetModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, device


def run_prediction(model, image_tensor, device):
    """Run model inference and return the reconstructed image.

    Args:
        model: Loaded NetModule in eval mode.
        image_tensor (torch.Tensor): Preprocessed input of shape (1, 1, H, W).
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (1, 1, H, W) in [0, 1].
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        reconstructed = torch.sigmoid(logits)
        return reconstructed.cpu()


def save_prediction(reconstructed, output_folder, filename):
    """Save the reconstructed image as a grayscale PNG.

    Args:
        reconstructed (torch.Tensor): Tensor of shape (1, 1, H, W) in [0, 1].
        output_folder (str): Directory to save output images.
        filename (str): Original input file path used to derive the output name.
    """
    os.makedirs(output_folder, exist_ok=True)
    base_name = Path(filename).stem
    img_array = (reconstructed.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
    out_path = os.path.join(output_folder, f"{base_name}_reconstructed.png")
    Image.fromarray(img_array, mode='L').save(out_path)
    print(f"Saved reconstruction: {out_path}")


def main():
    """Main function to run the prediction pipeline."""
    config = load_config("config_outerdata.toml")

    image_path = config['DataModule']['image_path']
    image_suffix = config['DataModule']['image_suffix']
    ilm_suffix = config['DataModule']['ilm_suffix']
    target_size = config['DataModule']['image_shape'][:2]  # (H, W)

    log_dir = config['NetModule']['log_dir']
    model_name = config['NetModule']['model_name']
    checkpoint_dir = os.path.join(log_dir, model_name)
    output_folder = os.path.join(log_dir, "predicted_out")

    # Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        print(f"No checkpoint files found in: {checkpoint_dir}")
        return
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    checkpoint_path = checkpoint_files[0]

    # Step 1: Load images
    print("Step 1: Loading images from folder...")
    image_files = listFiles(image_path, f"*{image_suffix}")
    ilm_files = listFiles(image_path, f"*{ilm_suffix}")
    if not image_files:
        print(f"No images found matching '*{image_suffix}' in {image_path}")
        return
    print(f"Found {len(image_files)} images.")

    # Step 2: Load model
    print("Step 2: Loading model from checkpoint...")
    model, device = load_model_from_checkpoint(checkpoint_path, config)

    # Step 3: Run predictions
    print("Step 3: Running predictions on images...")
    for i, (img_path, ilm_path) in enumerate(zip(image_files, ilm_files)):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        image_tensor = preprocess_image(img_path, ilm_path, target_size)
        reconstructed = run_prediction(model, image_tensor, device)
        save_prediction(reconstructed, output_folder, img_path)

    print(f"\nPrediction completed! Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
