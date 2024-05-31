import torch
import cv2
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    """
    A custom dataset class that loads a paired list of images and labels for training.

    Attributes:
        imList (list of str): A list containing paths to the images.
        labelList (list of str): A list containing paths to the corresponding label images.
        transform (callable, optional): An optional transform to be applied on a sample.
    """
    def __init__(self, imList, labelList, transform=None):
        """
        Initializes the dataset with image paths and labels.

        Parameters:
            imList (list of str): List of image file paths.
            labelList (list of str): List of label image file paths corresponding to the images.
            transform (callable, optional): A function/transform that takes in an image
                and label and returns a transformed version.
        """
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: the total number of samples in the dataset.
        """
        return len(self.imList)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index, applies a transform if any.

        Parameters:
            idx (int): Index

        Returns:
            tuple: (image, label) where both are transformed images and labels.
        """
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        
        # Load images using cv2
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)  # explicitly define the intention to load as grayscale
        
        # Apply the transform to both the image and the label if any transform is specified
        if self.transform:
            image, label = self.transform(image, label)

        return image, label
