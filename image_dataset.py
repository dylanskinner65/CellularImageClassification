import torch
import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from skimage import exposure

# make the train_with_target_id if it doesn't exist


def update_target_ids(filepath='data'):
    if not os.path.exists(f'{filepath}/train_with_target_id.csv'):
        controls = pd.read_csv(f'{filepath}/train_controls.csv')
        train = pd.read_csv(f'{filepath}/train.csv')
        df = pd.concat([train, controls], axis=0).reset_index(drop=True)
        df['target_id'] = LabelEncoder().fit_transform(df['sirna'])
        df.to_csv(f'{filepath}/train_with_target_id.csv', index=False)


class ImageDataset(Dataset):
    def __init__(self, train=True, transform=None, apply_equalize=False, clip_limit=0.01, apply_transform_train=False, apply_transform_test=False, filepath='data'):
        update_target_ids(filepath)
        postfix = 'train_fixed' if train else 'test'
        self.folder_path = f'{filepath}/{postfix}'
        self.df = pd.read_csv(
            f'{filepath}/train_with_target_id.csv') if train else pd.read_csv(f'{filepath}/{postfix}.csv')
        self.train = train
        self.transform = transform
        self.apply_equalize = apply_equalize
        self.apply_transform_train = apply_transform_train
        self.apply_transform_test = apply_transform_test
        self.clip_limit = clip_limit
        self.file_list = self._get_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path)
        
        if self.apply_equalize:
            image = np.asarray(image)/255.0
            image = exposure.equalize_adapthist(image, clip_limit=self.clip_limit)
        
        if self.transform:
            image = self.transform(image)
            image = image.to(dtype=torch.float32)
        
        if self.apply_transform_train:
            train_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(), 
                transforms.RandomRotation(90),
                transforms.Normalize(mean=[0.0], std=[1.0]),
                transforms.Lambda(lambda x: x*torch.normal(1.0, 0.1, size=x.size()) + torch.normal(0, 0.1, size=x.size()))
                ])

            image = train_transform(image)
        
        if self.apply_transform_test:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomVerticalFlip(1.0),
                transforms.RandomRotation((90,90)),
            ])

            image = test_transform(image)

        if self.train:
            label = self._extract_label(img_name)
            return image, label

        return image

    def _get_file_list(self):
        try:
            files = [file for file in os.listdir(
                self.folder_path) if file.endswith('.png')]
            return files
        except FileNotFoundError:
            print(f"The folder '{self.folder_path}' does not exist.")
            return []

    def _extract_label(self, filename):
        # Implement logic to extract label from filename or path
        # For example, if filenames are in the format "class_label_image.png"
        file_name = filename.split('_')
        label = file_name[0] + '_' + file_name[1] + '_' + file_name[2]
        # selected_row = self.df.loc[self.df['experiment'] == label]
        # return selected_row['sirna'].values[0]
        return self.df.loc[self.df['id_code'] == label]['target_id'].values[0]
