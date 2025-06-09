"""
clip_utils.py
Utility functions for loading and using CLIP models.
"""
import torch
import clip
from PIL import Image

def load_clip_model(version="ViT-L/14", device="cuda"):
    """Load the CLIP model and preprocessing pipeline."""
    model, preprocess = clip.load(version, device=device)
    return model, preprocess

def encode_image(image: Image.Image, detection_2d, clip_preprocess, clip_model, device="cuda"):
    """Encode the cropped image using the CLIP model."""
    x1, y1, x2, y2 = detection_2d[0]
    cropped_image = image.crop((x1, y1, x2, y2))
    with torch.no_grad():
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        image_embedding = clip_model.encode_image(preprocessed_image)
    return image_embedding

def encode_full_image(image: Image.Image, clip_preprocess, clip_model, device="cuda"):
    """Encode the full image using the CLIP model."""
    preprocessed_image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        full_image_embedding = clip_model.encode_image(preprocessed_image)
    return full_image_embedding

def encode_past_frames(image_paths, detection_2ds, clip_preprocess, clip_model, device="cuda"):
    """Encode past frames (cropped) using the CLIP model."""
    past_embeddings = []
    for image_path, detection_2d in zip(image_paths, detection_2ds):
        embedding = encode_image(image_path, detection_2d, clip_preprocess, clip_model, device)
        past_embeddings.append(embedding)
    return torch.stack(past_embeddings)

def clip_encode_text(text_prompt, clip_model, device="cuda"):
    """Encode text using the CLIP model."""
    with torch.no_grad():
        text_embedding = clip_model.encode_text(clip.tokenize([text_prompt]).to(device))
    return text_embedding
