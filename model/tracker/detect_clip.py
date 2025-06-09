import time
import os
import json
import torch
from PIL import Image
from .box_op import *
from .trajectory import Trajectory
import numpy as np
import sys
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import cv2
import re

def encode_text_sbert(text_prompt, sbert_model, device="cuda"):
    """Encode text using Sentence Transformers (SBERT)."""
    text_embedding = sbert_model.encode([text_prompt])
    return text_embedding

