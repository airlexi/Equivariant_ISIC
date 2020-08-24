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

The dataset with input sizes 1024x1024 can be downloaded at https://www.kaggle.com/c/siim-isic-melanoma-classification/data. Chris Deotte provides resized datasets here: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092.

## Goal
The aim of the project is to show that using General E(2)-Equivariant Steerable CNNs for skin lesion analysis is beneficial. Li, Yu et al. already demonstrated the effectiveness of using rotation equivariant networks for skin lesion segmentation in their paper: "Deeply Supervised Rotation Equivariant Network for Lesion Segmentation in Dermoscopy Images". This leads to the conclusion, that using rotation equivariant networks for the SIIM-ISIC 2020 challenge would be also beneficial.

Paper: https://arxiv.org/pdf/1807.02804.pdf

This leads to the conclusion, that using rotation equivariant networks for the SIIM-ISIC 2020 challenge would be also beneficial. For this purpose, the e2cnn-Pytorch extension created by Weiler and Cesa is used.

Paper: https://arxiv.org/pdf/1911.08251.pdf

## Experiments
### Data Augmentation
I have carried out different experiments, trying out different combinations of augmentations.

In their paper "Data Augmentation for Skin Lesiona Analysis", Perez, Vasconcelos et al. carried out different experiments looking for the best augmentations for skin lesion analysis. Results suggest that these are the best augmentations in the respective order:
* Random Crops
* Affine Transformations (rotating, sheering and scaling images)
* Flips
* Saturation, Contrast, Brightness and Hue changes (saturation, contrast and brightness modification sampled from an uniform distribution of (0.7, 1.3), hue shifted by a value sampled from an uniform distribution between (-0.1,0.1))

This experiment focuses on usage of rotation equivariant networks. Flips and rotations of input images can be neglected, since the output would be the same. Furthermore, hair augmentation is experimented with as well, since a huge amount of input images have lots of hairs in the image.

Furthermore, since lots of images are taken through a microscope lense, I have tested microscope augmentation.

I have also carried out experiments with the exact augmentations of the top-performing team in the competition.

### Architecture
Last years challenge was dominated mostly by ensembles of pretrained DenseNets, EfficientNets, ResNets and SeResNexts. Experiments were carried out on DenseNet121, DenseNet161 (https://arxiv.org/abs/1608.06993), ResNet50 and Resnet101 (https://arxiv.org/abs/1512.03385), since the other nets include operations which cannot be carried out with the e2cnn-library (e.g. swish-function: output = x * sigmoid(x)). I experimented with using meta-data and not using meta-data.

### External Data
It was also experimented with the usage of external data, namely the data from the ISIC-2019 challenge to reduce the class imbalance. ISIC-2019 data had about 4000 malignant samples, which should be beneficial for learning the the malignant class.

### Train-Test-Split
Since we have no validation set, we need to split the data into train and validation sets. To achieve a 80/20 split, I have split the data into 5 folds. I have experimented with stratified folds, group folds and random folds.

### Data imbalance
To fight data imbalance, I have tried using weighted loss, focal loss (https://arxiv.org/pdf/1708.02002.pdf) and oversampling, but experiments showed there were no benefits in using these techniques.

### Post-processing
For getting the final predictions, heavy test time augmentation is used, as proposed in "Data Augmentation for Skin Lesiona Analysis".

### Other hyperparameters
As a starting learning rate, I used a learning rate of 0.001. Furthermore, the learning rate reduces on plateaus by a factor of 0.2, with a patience of 1. I have also implemented early stopping, to prevent overfitting.

## Results
The leading submissions on the final leaderboard were ensembles of mostly pretrained EfficientNets. The best AUC score on the final leaderboard was 0.9490. My best performing equivariant model scored an AUC of 0.8659 on the private leaderboard. It used the augmentations proposed by Perez, Vasconcelos et al., the architecture is Densenet121 using meta-data, no external data, random folds.

This is not a great score, all of the models I tested were able to achieve a score of ca. 0.86 on the leaderboard, but none was able to perform better. There was no significant difference between deeper models like Resnet101 or Densenet161 and Resnet50 and Densenet121. It is yet to find out why there is such a huge difference between these equivariant models and the pretrained models used by others which were performing much better on the leaderboard. Furthermore, there is the question, why does making models more complex not improve their performance in this case? In theory, they should perform definitely better. One person achieved a score of 0.90 on public leaderboard with just a Resnet18, which is a much smaller network than for example Resnet50 (https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155668). One user, who used on Resnet50, achieved a score of 0.9203 training without external data (https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/171745). Still, the equivariant Resnet50 does not get a better score than 0.86.

## Status
Since the challenge ended on August 17th and the semester is over, I stopped working on the project.

## To-dos
Depending on the interest in the challenge, it could be evaluated, why the rotation equivariant networks do not yield better results in this challenge, although Li, Yu et al. showed benefits of using them in the 2017 ISIC challenge.

## Cite
