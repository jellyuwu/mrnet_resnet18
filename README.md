# A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries using the MRNet Dataset

Paper presented at The Third International Workshop on Deep and Transfer Learning ([DTL2020](http://intelligenttech.org/DTL2020/)) as part of International Conference on Intelligent Data Science Technologies and Applications (IDSTA2020).

Please consider citing the following paper if you use any of the work:
```
@article{azcona2020comparative,
  title={A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries using the MRNet Dataset},
  author={Azcona, David and McGuinness, Kevin and Smeaton, Alan F},
  journal={arXiv preprint arXiv:2010.01947},
  year={2020}
}
```

## Dataset

The MRNet dataset consists of knee MRI exams performed at Stanford University Medical Center. Further details can be found at https://stanfordmlgroup.github.io/competitions/mrnet/

* 1,370 knee MRI exams performed at Stanford University Medical Center
* 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears
* Labels were obtained through manual extraction from clinical reports



## Deployment

In our paper we propose and evaluate the performance of the following architectures to train networks and output the probabilities for a patient to have an ACL tear, meniscal tear, or some abnormality on their knee:

1. Training a Deep Residual Network with Transfer Learning
2. Training a Deep Residual Network from Scratch & Use a Fixed Number of Slices
3. Training a Multi-Plane Deep Residual Network
4. Training a Multi-Plane Multi-Objective Deep Residual Network

### 1. Training a Deep Residual Network with Transfer Learning

1. Select the approach by editing [config.py](src/config.py):
```
APPROACH = 'pretrained'
```

`pretrained` uses ImageNet pre-trained weights.

2. Train a model for each task and for each plane:
```
$ python src/train_baseline.py -t '<task>' -p '<plane>'
```

For the `pretrained` approach we use `train_baseline.py`.

For instance, for task ```acl```:
```
$ python src/train_baseline.py -t 'acl' -p 'axial'
$ python src/train_baseline.py -t 'acl' -p 'coronal'
$ python src/train_baseline.py -t 'acl' -p 'sagittal'
```
and then repeat for tasks ```meniscus``` and ```abnormal```. 

3. For each task, combine predictions per plane by training a Logistic Regression model:
```
$ python src/combine.py -t '<task>'
```

For instance, for task ```acl```:
```
$ python src/combine.py -t 'acl'
```
and then repeat for tasks ```meniscus``` and ```abnormal```.

The models with the greatest validation AUC are picked per plane

4. Generate predictions for each patient in the sample test set for all tasks: ```acl```, ```meniscus``` and ```abnormal```:
```
$ python src/predict.py
```

### 2. Training a Deep Residual Network from Scratch & Use a Fixed Number of Slices

1. Select the approach by editing [config.py](src/config.py):
```
APPROACH = 'slices'
```

`slices` uses a fixed number of slices to train a network from scratch with random initialization of the weights

2. Train a model for each task and for each plane:
```
$ python src/train_slices.py -t '<task>' -p '<plane>'
```

3. For each task, combine predictions per plane by training a Logistic Regression model:
```
$ python src/combine.py -t '<task>'
```

4. Generate predictions for each patient in the sample test set for all tasks:
```
$ python src/predict.py
```

### 3. Training a Multi-Plane Deep Residual Network

1. Select the approach by editing [config.py](src/config.py):
```
APPROACH = 'slices'
```

2. Train a model for each task but all the planes together:
```
$ python src/train_slices_planes.py -t '<task>'
```

4. Generate predictions for each patient in the sample test set for all tasks:
```
$ python src/predict_planes.py
```

### 4. Training a Multi-Plane Multi-Objective Deep Residual Network

1. Select the approach by editing [config.py](src/config.py):
```
APPROACH = 'slices'
```

2. Train a model for all task and all planes together:
```
$ python src/train_slices_planes_tasks.py -t '<task>'
```

4. Generate predictions for each patient in the sample test set for all tasks:
```
$ python src/predict_planes.py
```

## Further work

The following notebooks show how to **augment the MR images** by using: 

* [Pytorch transformations](src/notebooks/Augmentation%20I.ipynb), 
* [imgaug](src/notebooks/Augmentation%20II.ipynb), 
* [albumentations](src/notebooks/Augmentation%20III.ipynb#) and 
* [augmentor](src/notebooks/Augmentation%20IV.ipynb)