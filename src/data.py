import glob
import os
import lightning.pytorch as pl
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)


class SpleenDataModule(pl.LightningDataModule):
    """
    Class that represents train, validation and test dataloading functionality.

    Attributes
    ----------
    data_dir : str
        a chained series of transforms for train data
    train_transforms : Compose
        a chained series of transforms for train data
    val_transforms : Compose
        a chained series of transforms for validation data
    test_transforms : Compose
        a chained series of transforms for test data
    train_ds : Dataset
        training dataset
    val_ds :Dataset
        validation dataset
    test_ds : Dataset
        testing dataset

    Methods
    -------
    setup(stage):
        performs data setup operations and defines datasets
    train_dataloader():
        generates and returns training dataloader
    val_dataloader():
        generates and returns validation dataloader
    test_dataloader():
        generates and returns test dataloader
    """
    def __init__(self, data_dir: str = "./"):
        """Initialize attributes

        Parameters
        ----------
        data_dir: str
            path to data directory
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.train_transforms = Compose(
            [
                # loads the images from NifTi files passed as keys
                LoadImaged(keys=["image", "label"]),
                # ensure channel dimension to be the first dimension
                EnsureChannelFirstd(keys=["image", "label"]),
                # apply specific intensity scaling (a-original range, b-target range)
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # crop foreground using bounding box determined by source key
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # change the orientation of passed images to 3D
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # resample into specified voxel spacing
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                # 'RandCropByPosNegLabeld' crops random fixed sized regions with the center being
                # a foreground or background voxel based on the Pos Neg ratio
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
            ]
        )
        self.val_transforms = Compose(
            [
                # loads the images from NifTi files passed as keys
                LoadImaged(keys=["image", "label"]),
                # ensure channel dimension to be the first dimension
                EnsureChannelFirstd(keys=["image", "label"]),
                # apply specific intensity scaling (a-original range, b-target range)
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # crop foreground using bounding box determined by source key
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # change the orientation of passed images to 3D
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # resample into specified voxel spacing
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ]
        )
        self.test_transforms = Compose(
            [
                # loads the images from NifTi files passed as keys
                LoadImaged(keys="image"),
                # ensure channel dimension to be the first dimension
                EnsureChannelFirstd(keys="image"),
                # change the orientation of passed images to 3D
                Orientationd(keys=["image"], axcodes="RAS"),
                # resample into specified voxel spacing
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                # apply specific intensity scaling (a-original range, b-target range)
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # crop foreground using bounding box determined by source key
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    def setup(self, stage: str):
        """Performs data setup operations such as data splitting, applying transforms
        and defining the datasets

        Parameters
        ----------
        stage: str
            either 'fit', 'validate', 'test', or 'predict'
        """
        # assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_images = sorted(glob.glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")))
            train_labels = sorted(glob.glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))
            # arrange data as a list of dictionaries
            data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                          zip(train_images, train_labels)]
            # split to training and validation sets (training set split ration 0.8)
            train_files, val_files = data_dicts[:-9], data_dicts[-9:]

            # 'CacheDataset' used to accelerate training process
            self.train_ds = CacheDataset(data=train_files, transform=self.train_transforms, cache_rate=1.0, num_workers=4)
            # self.train_ds = Dataset(data=train_files, transform=self.train_transforms)
            # 'CacheDataset' used to accelerate validation process
            self.val_ds = CacheDataset(data=val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=4)
            # self.val_ds = Dataset(data=val_files, transform=self.val_transforms)

        # assign test dataset for use in dataloader
        if stage == "test":
            test_images = sorted(glob.glob(os.path.join(self.data_dir, "imagesTs", "*.nii.gz")))
            test_files = [{"image": image} for image in test_images]
            self.test_ds = Dataset(data=test_files, transform=self.test_transforms)

    def train_dataloader(self):
        """Returns training dataloader, an iterable over training dataset with
        cache deterministic transforms' result during training

        Returns
        -------
        DataLoader
            MONAI dataloader for training data loading
        """
        return DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Returns validation dataloader, an iterable over validation dataset with
        cache deterministic transforms' result during validation

        Returns
        -------
        DataLoader
            MONAI dataloader for validation data loading
        """
        return DataLoader(self.val_ds, batch_size=1, num_workers=4)

    def test_dataloader(self):
        """Returns testing dataloader, an iterable over testing dataset

        Returns
        -------
        DataLoader
            MONAI dataloader for testing data loading
        """
        return DataLoader(self.test_ds, batch_size=1, num_workers=4)
