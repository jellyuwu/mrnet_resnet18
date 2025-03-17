# Advancing Meniscus Tear Diagnosis in Knee MRI: Performance Improvements Over MRNet

## Dataset

The MRNet dataset consists of knee MRI exams performed at Stanford University Medical Center. Further details can be found at https://stanfordmlgroup.github.io/competitions/mrnet/

* 1,370 knee MRI exams performed at Stanford University Medical Center
* 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears
* Labels were obtained through manual extraction from clinical reports

## Setup
1. Setting up a virtual environment
```
conda create --name mrnet python=3.9
```
2. Activating the virtual environment
```
conda activate mrnet
```
3. Installing required dependencies and packages
```
pip install -r requirements.txt
```
4. Make sure you have the correct PyTorch version with CUDA support installed. For example, for CUDA 12.1, use:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If you are using a different CUDA version, visit the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to find the appropriate command.

## Deployment

### Using the Best Configurations from the [Reference Paper](https://arxiv.org/pdf/2010.01947) (Refer to commands.sh for additional details)

#### Step 1: Select the Approach
Edit the [config.py](src/config.py) to specify the approach:
```
APPROACH = 'pretrained'
```

- The `pretrained` approach uses ImageNet pre-trained weights.

#### Step 2: Train a Model for Each Plane
Run the following commands to train models for the `meniscus` detection task across `axial`, `coronal`, and `sagittal` planes:
```
python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'axial' \
    --epochs 200 \
    --augment_prob 0.40

python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'coronal' \
    --epochs 200 \
    --augment_prob 0.40

python 'src/train_baseline.py' \
    --prefix_name 'base' \
    -t 'meniscus' \
    -p 'sagittal' \
    --epochs 100 \
    --augment_prob 0.90
```

- The `pretrained` approach uses train_baseline.py for training.
- The `data augmentation probability` (--augment_prob) is set as per the best configurations.

#### Step 3: Combine Predictions Across Planes
For each task, combine predictions from different MRI planes by training a `Logistic Regression` model:
```
python src/combine.py -t 'meniscus'
```

- The models with the `highest validation AUC` are selected per plane.

## Results

The training log is saved as `training_log.txt`.  

- Validation AUC for the task "meniscus": **0.8654**  
- This value can be found on **line 11083** of `training_log.txt`.  

The trained models are stored in:  
📂 `my-data/models/training/pretrained`


## Acknowledgment
This repository is downloaded from https://github.com/dazcona/mrnet