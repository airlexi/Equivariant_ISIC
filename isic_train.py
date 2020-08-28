from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torchvision
import math
import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces
from PIL import Image
import os
import json
import pickle
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import cv2
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchtoolbox.transform as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import sklearn
from tqdm.autonotebook import tqdm

class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)
        
        if self.transforms:
            x = self.transforms(x)
            
        if self.train:
            y = self.df.loc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
        
        
    def __len__(self):
        return len(self.df)
    
 
        
        
class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    
    taken from: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'
        
    
class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
        
    taken from: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'
    
    
    
    
def run_epoch(model, optimizer, dataloader, train, loss_fnc, length_dataset, batch_size):
    """running one epoch of training or validation"""
    
    if train==True:
        #set model to training mode
        model.train() 
        
        #variables for predictions, labels and loss
        train_preds = torch.zeros(size = (length_dataset, 1), device="cpu", dtype=torch.float32) 
        train_labels = torch.zeros(size = (length_dataset, 1), device="cpu", dtype=torch.float32) 
        epoch_loss = 0.0
        
        for k, (images, labels) in tqdm(enumerate(dataloader), total=int(torch.ceil(torch.tensor(length_dataset / batch_size)).item())):
            
            images[0] = images[0].to(device)
            images[1] = images[1].to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            out = model(images)
            loss = loss_fnc(out, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = torch.sigmoid(out)
            
            #write predictions and labels to respective variables
            train_preds[k*batch_size : k*batch_size + images[0].shape[0]] = pred.detach()
            train_labels[k*batch_size : k*batch_size + images[0].shape[0]] = labels.detach().unsqueeze(1)
            
        #compute accuracy
        train_acc = accuracy_score(train_labels.cpu(), 
                                       torch.round(train_preds).cpu())
        
        #compute Area Under Receiving Operator Characteristic
        train_roc = roc_auc_score(train_labels.cpu(), 
                                      train_preds.cpu())
    
        return epoch_loss, train_acc, train_roc
        
        
        
    else:
        #set model to evaluation mode
        model.eval() 
        
        #variables for predictions, labels and loss
        valid_preds = torch.zeros(size = (length_dataset, 1), device="cpu", dtype=torch.float32)
        valid_labels = torch.zeros(size = (length_dataset, 1), device="cpu", dtype=torch.float32)
        epoch_loss = 0.0
        
        with torch.no_grad():
            for k, (images, labels) in tqdm(enumerate(dataloader), total=int(torch.ceil(torch.tensor(length_dataset / batch_size)).item())):
                
                images[0] = images[0].to(device)
                images[1] = images[1].to(device)
                labels = labels.float().to(device)

                out = model(images)
                loss = loss_fnc(out,labels.unsqueeze(1))
                
                epoch_loss +=loss.item()
                pred = torch.sigmoid(out)
                
                #write predictions and labels to respective variables
                valid_preds[k*batch_size : k*batch_size + images[0].shape[0]] = pred.cpu()
                valid_labels[k*batch_size : k*batch_size + images[0].shape[0]] = labels.cpu().unsqueeze(1)

            #compute accuracy
            valid_acc = accuracy_score(valid_labels.cpu(), 
                                       torch.round(valid_preds.cpu()))
            #compute Area Under Receiving Operator Characteristic
            valid_roc = roc_auc_score(valid_labels.cpu(), 
                                      valid_preds.cpu())

        
        return epoch_loss, valid_acc, valid_roc


    
def fit(max_epochs, patience, batch_size):
    """fitting the model"""
    
    #defining the split for cross-validation
    skf = KFold(n_splits=5, shuffle=True, random_state=47) #random_state set, such that the split is reproducible
    
    #list for saving the output logs
    logs = []
    
    #training for each of the above defined folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
        print("FOLD: "+str(fold))
        logs.append("FOLD: "+str(fold))
        
        #creating the chosen model
        model = DenseNet(32, [6,12,24,16], len(meta_features)).to(device)
        print("model initialized!")
        
        #setting the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1, verbose=True, factor=0.1)
        
        #getting the indices for the respective train-validation split
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        valid_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        #creating the dataset with the respective train-validation splits
        train_set = MelanomaDataset(df=train_data, 
                            imfolder='train', 
                            train=True, 
                            transforms=train_transform,
                            meta_features=meta_features)
        
        val_set = MelanomaDataset(df=valid_data, 
                            imfolder='train', 
                            train=True, 
                            transforms=test_transform,
                            meta_features=meta_features)
        
        #initializing the dataloaders
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        
        #variables to save the best loss and best ROC (Area Under Receiving Operator Characteristic)
        best_loss = 100000000
        best_roc = 0.0
        
        #setting the loss functios, BCEWithLogitsLoss includes a sigmoid layer into the Binary Cross-Entropy loss
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion_val = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(max_epochs):
            
            print(f"Epoch {epoch + 1}/{max_epochs}")
            # Training one epoch
            print("")
            print("training...")
            epoch_loss, train_acc, train_roc = run_epoch(model, optimizer, train_loader, train=True, 
                                                         loss_fnc =criterion,
                                                               length_dataset =len(train_set), batch_size = batch_size)
            

            print(f"Train loss: {epoch_loss}")
            print(f"ACC on training set: {train_acc}")
            print(f"ROC on training set: {train_roc}")
            print("")
            
            #saving the logs to the list
            logs.append(("Fold: "+str(fold) +", epoch: "+str(epoch)+
                        ": train loss: "+str(epoch_loss)+
                        ", train acc: "+str(train_acc)+", train roc: "+str(train_roc)))
            
            # Validating one epoch
            print("validating...")
            
            val_loss, val_acc, val_roc = run_epoch(model, None, val_loader, train=False,
                                                        loss_fnc = criterion_val,length_dataset =len(val_set),
                                                                   batch_size = batch_size)
                                                                   
            print(f"Test loss: {val_loss}")
            print(f"Val Acc: {val_acc}")
            print(f"Val Roc: {val_roc}")
            print("")
            
            #saving the logs to the list
            logs.append(("Fold: "+str(fold) +", epoch: "+str(epoch)+
                        ": val loss: "+str(val_loss)+
                        ", val acc: "+str(val_acc)+", val roc: "+str(val_roc)))

            #learning rate scheduler step, minimizing the learning rate when the train loss does not decrease for a epoch
            lr_scheduler.step(epoch_loss)
            
            # saving best weights
            if val_roc >= best_roc:
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
                best_roc = val_roc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model"+str(fold))

            # Early stopping
            if epoch - best_epoch >= patience:
                break
            
            #saving the log to a .json file
            with open("results"+str(fold)+".json", "w") as file:
                json.dump(logs, file)




if __name__ == "__main__":
       
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #loading the csv file including meta data and file names
    train_df = pd.read_csv('train.csv')

    # data augmentation for training
    train_transform = transforms.Compose([
        AdvancedHairAugmentation(hairs_folder='mel_hairs'),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, scale=(0.8,1.2), shear=(-20,20)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        Microscope(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    # data augmentation for validating
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    # preprocessing of meta data, taken from https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
    # One-hot encoding of location of imaged site
    concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    
    # encoding the sex of patients, -1 if it is missing
    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    train_df['sex'] = train_df['sex'].fillna(-1)

    # normalizing the age of the patients between 0 and 1 to use it for classification as well, 0 if the age is missing
    train_df['age_approx'] /= train_df['age_approx'].max()
    train_df['age_approx'] = train_df['age_approx'].fillna(0)
    
    # filling missing values for patient_ids
    train_df['patient_id'] = train_df['patient_id'].fillna(0)
    meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]
    meta_features.remove('anatom_site_general_challenge')


    # Fitting the model model
    fit( max_epochs=30, patience=60, batch_size=64)

