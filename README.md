# Equivariant_ISIC
## The SIIM-ISIC Melanoma Classification Challenge
The goal of the SIIM-ISIC Challenge is identifying Melanoma in lesion images. It is a binary classification problem and the model needs to output the probability between 0.0 and 1.0. The value 0.0 accounts for benign and 1.0 for malignant. The training dataset consists of 33125 images, of which 32542 are benign and 584 malignant. In addition to the images, meta-data is provided in .csv-files, consisting of:
* image_name - unique identifier, points to filename of related DICOM image
* patient_id - unique patient identifier
* sex - the sex of the patient (when unknown, will be blank)   
* age_approx - approximate patient age at time of imaging
* anatom_site_general_challenge - location of imaged site
* diagnosis - detailed diagnosis information (train only)
* benign_malignant - indicator of malignancy of imaged lesion
* target - binarized version of the target variable

It is possible to participate in two ways in this challenge. You can either participate with models using meta-data and with models which classify the data solely based on the images, without the usage of meta-data. The evaluation metric is the area under the ROC curve between the predicted probability and the observed target.

![SAMPLE1](./ReadmeFiles/DataStruct.png)
> Figure 1. Example Image of the Dataset

## Goal
The aim of the project is to show that using General E(2)-Equivariant Steerable CNNs for skin lesion analysis is beneficial. Li, Yu et al. already demonstrated the effectiveness of using rotation equivariant networks for skin lesion segmentation in their paper: "Deeply Supervised Rotation Equivariant Network for Lesion Segmentation in Dermoscopy Images". This leads to the conclusion, that using rotation equivariant networks for the SIIM-ISIC 2020 challenge would be also beneficial.

Paper: https://arxiv.org/pdf/1807.02804.pdf

This leads to the conclusion, that using rotation equivariant networks for the SIIM-ISIC 2020 challenge would be also beneficial. For this purpose, the e2cnn-Pytorch extension created by Weiler and Cesa is used.

Paper: https://arxiv.org/pdf/1911.08251.pdf

## Experiments
### Data Augmentation
In their paper "Data Augmentation for Skin Lesiona Analysis", Perez, Vasconcelos et al. carried out different experiments looking for the best augmentations for skin lesion analysis. Results show, the best augmentation are in the following order:
* Random Crops
* Affine Transformations (rotating, sheering and scaling images)
* Flips
* Saturation, Contrast, Brightness and Hue changes (saturation, contrast and brightness modification sampled from an uniform distribution of (0.7, 1.3), hue shifted by a value sampled from an uniform distribution between (-0.1,0.1))

This experiment focuses on usage of rotation equivariant networks. Flips and rotations of input images can be neglected, since the output would be the same. Furthermore, hair augmentation is experimented with as well, since a huge amount of input images have lots of hairs in the image.

### Architecture
Last years challenge was dominated mostly by ensembles of pretrained DenseNets, EfficientNets, ResNets and SeResNexts. Experiments were carried out on DenseNet121, DenseNet161, ResNet50 and Resnet101, since the other nets include operations which cannot be carried out with the e2cnn-library (e.g. swish-function: output = x * sigmoid(x)). 

### External Data
It was also experimented with the usage of external data, namely the data from the ISIC-2019 challenge to reduce the class imbalance. ISIC-2019 data had about 4000 malignant samples, which should be beneficial for learning the the malignant class.

### Post-processing
For getting the final predictions, heavy test time augmentation is used, as proposed in "Data Augmentation for Skin Lesiona Analysis".
