





# Proximity QA: Unleashing the Power of Multi-Modal Large Language Models for Spatial Proximity Analysis

Multi-modal large language models (MLLMs)
have demonstrated remarkable vision-language
capabilities, primarily due to the exceptional
in-context understanding and multi-task learning
strengths of large language models (LLMs). The
advent of visual instruction tuning has further enhanced MLLMs’ performance in vision-language
understanding. However, while existing MLLMs
adeptly recognize what objects are in an image,
they still face challenges in effectively discerning
where these objects are, particularly along the
distance (scene depth) axis. To overcome this
limitation in MLLMs, we introduce Proximity
Question Answering (Proximity QA), a novel
framework designed to enable MLLMs to analyse
the proximity relationship between objects in
images. The framework operates in two phases:
the first phase focuses on guiding the models
to understand the relative depth of objects, and
the second phase further encourages the models
to analyse the proximity relationships between
objects based on their depth perceptions. We
also propose a VQA dataset called Proximity-110K, containing additional instructions that
incorporate depth information and the proximity
relationships of objects. We have conducted
extensive experiments to validate Proximity QA’s
superior ability in depth perception and proximity
analysis, outperforming other state-of-the-art
MLLMs. 

![pqa_v2f4961c876370efe6.jpeg](https://img.picgo.net/2024/04/02/pqa_v2f4961c876370efe6.jpeg)

## Traing Data
Our training data images originate from parts of the COCO and VG datasets. Regarding instruction data, we combine LLaVA-665K and Proximity-110K datasets for training, which can be accessed via [this link](https://huggingface.co/Electronics/ProximityQA/blob/main/llava_proximity-mix.json). The training pipline is the same with [LLaVA-1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning). Moreover, we are optimizing our training data, which includes incorporating more conversation templates, QA-types, etc. The optimized data will be released soon.



## Model Zoo
LoRA weights will be released soon.

## Evaluation

### GQA

Download the data following the official instructions [here](https://cs.stanford.edu/people/dorarad/gqa/download.html) and put under `./playground/eval/val_image`.

#### Evaluation on Depth Perception
Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa_depth_percp.sh

python ./playground/eval/eval_depth_percp.py
```
#### Evaluation on Proximity Analysis
Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa_depth_direct.sh

python ./playground/eval/eval_depth_direct.py
```

### Make3D

#### Evaluate proximity analysis
Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/make3d_depth_direct.sh
```
