# ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking
<p align="center">
<img src="https://github.com/Tzoulio/ReferGPT/blob/main/img/in_front_cars.gif"/>
<img src="https://github.com/Tzoulio/ReferGPT/blob/main/img/same_direction.gif"/>
<img src="https://github.com/Tzoulio/ReferGPT/blob/main/img/black_cars.gif"/>
</p>

## Abstract
<p align="justify">
Tracking multiple objects based on textual queries is a challenging task that requires linking language understanding with object association across frames. Previous works typically train the whole process end-to-end or integrate an additional referring text module into a multi-object tracker, but they both require supervised training and potentially struggle with generalization to open-set queries. In this work, we introduce ReferGPT, a novel zero-shot referring multi-object tracking framework. We provide a multi-modal large language model (MLLM) with spatial knowledge enabling it to generate 3D-aware captions. This enhances its descriptive capabilities and supports a more flexible referring vocabulary without training. We also propose a robust query-matching strategy, leveraging CLIP-based semantic encoding and fuzzy matching to associate MLLM generated captions with user queries. Extensive experiments on Refer-KITTI, Refer-KITTIv2 and Refer-KITTI+ demonstrate that ReferGPT achieves competitive performance against trained methods, showcasing its robustness and zero-shot capabilities in autonomous driving.</p>

## Methodology 
<div align="center">
  <img src="./img/main_architecture.png">
</div>

## Instalation
TBD

## Results
| **Dataset**   | **HOTA** | **DetA** | **DetRe** | **DetPR**| **AssA** | **AssRe** | **AssPr**| **LocA** |
|---------------|----------|----------|-----------|----------|----------|-----------|----------|----------|
|**Refer-KITTI**      |  49.46   |  39.43   |   50.21   |   58.91  |   62.57  |   73.74   |   72.78  |   81.85  |
|**Refer-KITTIv2**     |  30.12   |  15.69   |   21.55   |   34.41  |   59.02  |   74.59   |   68.20  |   79.76  |
|**Refer-KITTI+**      |  43.44   |  29.89   |   36.59   |   56.98  |   63.60  |   75.20   |   73.27  |   82.23  |

## Acknowledgement 
<p align="justify">
The work is supported from the ”Onderzoeksprogramma Artificiele Intelligentie (AI) Vlaanderen” programme and by Innoviris within the research project TORRES. N. Deligiannis acknowledges support from the Francqui Foundation (2024-2027 Francqui Research Professorship). </p>

## Citation 
TBD 
