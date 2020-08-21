#Equivariant_ISIC
## The SIIM-ISIC Melanoma Classification Challenge
The goal of the SIIM-ISIC Challenge is identifying Melanoma in lesion images. It is a binary classification problem and the model needs to output the probability between 0.0 and 1.0. 
The value 0.0 accounts for benign and 1.0 for malignant. The training dataset consists of 33125 images, of which 32542 are benign and 584 malignant. 
In addition to the images, meta-data is provided in .csv-files, consisting of:
* image_name - unique identifier, points to filename of related DICOM image
* patient_id - unique patient identifier
* sex - the sex of the patient (when unknown, will be blank)   
* age_approx - approximate patient age at time of imaging
* anatom_site_general_challenge - location of imaged site
* diagnosis - detailed diagnosis information (train only)
* benign_malignant - indicator of malignancy of imaged lesion
* target - binarized version of the target variable

It is possible to participate in two ways in this challenge. You can either participate with models using meta-data and with models which classify the data solely based on the images, without 
the usage of meta-data.

