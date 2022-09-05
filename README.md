# SimCLR-in-TensorFlow-2
(Minimally) implements SimCLR ([A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Chen et al.) in TensorFlow 2. Uses many delicious pieces of `tf.keras` and TensorFlow's core APIs. A report is available [here](https://app.wandb.ai/sayakpaul/simclr/reports/Towards-self-supervised-image-understanding-with-SimCLR--VmlldzoxMDI5NDM).

## Acknowledgements
I did not code everything from scratch. This particular research paper felt super amazing to read and often felt natural to understand, that's why I wanted to try it out myself and come up with a minimal implementation. I reused the works of the following for different purposes -
- Data augmentation policies comes from here: https://github.com/google-research/simclr/blob/master/data_util.py.
- Loss function comes from here: https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py.

Following are the articles I studied for understanding SimCLR other than the paper:
- [Understanding SimCLR — A Simple Framework for Contrastive Learning of Visual Representations with Code](https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c)
- [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)
- [Illustrated SimCLR](https://amitness.com/2020/03/illustrated-simclr/) (This one does an amazing job at explaining the loss function" NT-XEnt Loss)


## Dataset
- Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- data_processing_102_flowers.ipynb can be referred for raw data split codes.
- All folders which had greater than 100 images were taken for experiment.
- Unlabeled data: 70 per class, labeled train: 20 per class, test and val (same set): 10 per class

## Architecture
- Training This simclr architecture with lower resolution image gave loss as 'NAN' and hence had to experiment with 224*224*3
- This model was trained only for 50 epochs due to resource constraint. 
- It took 1hr 45min to complete just 10 epochs
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
resnet50 (Model)             (None, 7, 7, 2048)        23587712
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 256)               524544
_________________________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                6450
=================================================================
Total params: 24,151,602
Trainable params: 24,098,482
Non-trainable params: 53,120
```

## Contrastive learning progress
- The results of simclr training curve can be seen in Simclr_imagenet_subset.ipynb

## Training with 10% training data using the learned representations (linear evaluation)
- In linear_Evaluation_Imagenet_subset.ipynb, various experiments with and without project head was conducted.
- The results of the respective experiments can be seen in the cells.

## Supervised training with the full training dataset

Here's the architecture that was used:

```

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_4 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
resnet50 (Model)             (None, 7, 7, 2048)        23587712
_________________________________________________________________
global_average_pooling2d_1 ( (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               524544
_________________________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 23)                 1285
=================================================================
Total params: 24,113,541
Trainable params: 24,060,421
Non-trainable params: 53,120
```
## Observation and conclusions
- The val_accuracy appears to be 1 in both ssl--> downstream task and supervised classification task.
- Also tried RotNet which gave 34% accuracy with self-supervised trained model used for downstream task with same dataset.
- Need more time to check results in depth as there is resource constraint.

## Pre-trained weights
Available here - `Pretrained_Weights`.

## Trained weights
Available here - `SSL_trained_weights`
