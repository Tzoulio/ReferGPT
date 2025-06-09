# Installation Guide

## Prerequisites

- Python 3.12+ (tested with Python 3.12.9)
- CUDA 11.0+ or CUDA 12.0+ (for GPU acceleration)
- Linux (Ubuntu 18.04+ recommended)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tzoulio/ReferGPT.git
cd ReferGPT
```

### 2. Create Conda Environment

```bash
conda create -n refergpt python=3.12
conda activate refergpt
```

### 3. Install Dependencies

#### Option A: Install from requirements file (recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Install major dependencies manually
```bash
# Core PyTorch and ML frameworks
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install transformers==4.49.0 accelerate==1.5.0

# NLP and Vision-Language Models
pip install sentence-transformers==3.4.1
pip install clip==1.0

# Computer Vision and Scientific Computing
pip install opencv-python==4.11.0.86
pip install numpy==2.0.1 PyYAML==6.0.2

# Tracking and evaluation utilities
pip install filterpy==1.4.5 motmetrics==1.4.0
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers installed successfully')"
python -c "import clip; print('CLIP installed successfully')"
```

## Data Preparation

### Dataset Structure

After downloading all required data, your dataset directory should look like this:

```
dataset/
├── data/
│   ├── KITTI/                              # KITTI tracking dataset (symlink)
│   │   ├── training/
│   │   │   ├── calib/
│   │   │   ├── image_02/
│   │   │   ├── pose/
│   │   │   ├── label_02/
│   │   │   └── velodyne/
│   │   └── testing/
│   │       ├── calib/
│   │       ├── image_02/
│   │       ├── pose/
│   │       ├── label_02/
│   │       └── velodyne/
│   ├── detections_casa/                    # CasA detections (recommended)
│   │   ├── 0005/
│   │   ├── 0011/
│   │   └── 0013/
│   ├── refer-kitti-v1/                     # Refer-KITTI v1 annotations
│   │   ├── expression/
│   │   │   ├── 0001/
│   │   │   ├── 0002/
│   │   │   ├── ...
│   │   │   └── 0020/
│   │   └── labels_with_ids/
│   │       └── image_02/
│   │           ├── 0000/
│   │           ├── 0001/
│   │           ├── ...
│   │           └── 0020/
│   ├── refer-kitti-v2/                     # Refer-KITTI v2 annotations
│   │   ├── expression/
│   │   │   ├── 0000/
│   │   │   ├── 0001/
│   │   │   ├── ...
│   │   │   └── 0020/
│   │   └── labels_with_ids/
│   │       └── image_02/
│   │           ├── 0000/
│   │           ├── 0001/
│   │           ├── ...
│   │           └── 0020/
│   ├── refer-kitti-v1++/                   # Refer-KITTI v1++ annotations (optional)
│   │   └── expression/
│   │       ├── 0001/
│   │       ├── 0002/
│   │       ├── ...
│   │       └── 0020/
│   ├── updated_casa_llm_output_data/       # Pre-computed LLM outputs for cars
│   │   ├── 5/
│   │   ├── 11/
│   │   └── 13/
│   └── updated_casa_llm_output_data_pedestrian/  # Pre-computed LLM outputs for pedestrians
│       ├── 11/
│       └── 13/
├── data_path/                              # Sequence mapping for v1
│   ├── refer-kitti.train
│   └── seqmap.txt
├── data_path_v2/                           # Sequence mapping for v2
│   ├── refer-kitti-v2.train
│   └── seqmap.txt
└── prompts/                                # System prompts for LLM
    ├── prompt.txt
    ├── pedestrian_prompt.txt
    └── pedestrian_promptv2.txt
```

### Download Required Data

#### 1. KITTI Tracking Dataset

Download the KITTI tracking dataset from the [official website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and create a symlink:

```bash
# Create symlink to your KITTI dataset location
ln -s /path/to/your/KITTI/dataset dataset/data/KITTI
```

#### 2. CasA Detections

Download CasA detections (recommended detector):
```bash
# Download from Google Drive
# https://drive.google.com/drive/folders/1-1HgfdTiYApqfiF7oIJQ5xBIyudTdczy

# Extract to dataset/data/detections_casa/
# Should contain subdirectories: 0005/, 0011/, 0013/
```

#### 3. Refer-KITTI Annotations

Download the referring expressions and labels:
```bash
# Download from Google Drive
# https://drive.google.com/drive/folders/1IpDeFDu9CtrQWyH9Au8K91uk1SbwujWI

# Extract refer-kitti-v1 and refer-kitti-v2 to dataset/data/
```

#### 4. Pre-computed LLM Outputs (Optional)

For faster inference, download pre-computed ChatGPT outputs:
```bash
# Download from Google Drive
# https://drive.google.com/drive/folders/1B-Cqi9forxT_KjWwCLSIVJH6EQ1BK-3v

# Extract to:
# - dataset/data/updated_casa_llm_output_data/ (for cars)
# - dataset/data/updated_casa_llm_output_data_pedestrian/ (for pedestrians)
```

## Configuration

After installation, you'll need to configure the paths in the config files:

```bash
# Edit the main configuration file
nano config/global/cfg_refergpt.yaml
```

Update the following paths:
- `dataset_path`: Path to your KITTI dataset
- `detections_path`: Path to your detection files  
- `expression_path`: Path to referring expressions
- `llm_output_data_file`: Path to LLM outputs (if using pre-computed)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in VLM configuration
   - Use smaller VLM model (e.g., Phi instead of LLaVA)

2. **Missing Dependencies**
   ```bash
   pip install --upgrade transformers accelerate
   ```

3. **Path Configuration**
   - Ensure all paths in config files are absolute paths
   - Verify dataset structure matches expected format

4. **Version Conflicts**
   - Use the exact versions specified in `requirements.txt`
   - Create a fresh conda environment if needed

### Notes

- **KITTI Dataset**: Use symlinks to avoid duplicating large datasets
- **CasA Detections**: Currently supports sequences 0005, 0011, and 0013
- **LLM Outputs**: Pre-computed outputs are available for CasA detections only
- **Refer-KITTI Versions**: v1 and v2 have different annotation formats and sequence coverage

## Next Steps

After completing the installation, refer to the main [README.md](../readme.md) for:
- Quick start guide
- Usage examples
- Configuration options
- Results and evaluation