from monai.data import CacheDataset, DataLoader
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


class Loader:
    """
    Class that represents train and validation data loading functionality.

    Attributes
    ----------
    train_files : list
        training set organised as a list of dictionaries with image label pairs
    val_files : list
        validation set organised as a list of dictionaries with image label pairs
    train_transforms : Compose
        a chained series of transforms for train data
    val_transforms : Compose
        a chained series of transforms for validation data

    Methods
    -------
    get_train_loader():
        returns training dataloader
    get_val_loader():
        returns validation dataloader
    """

    def __init__(self, train_files, val_files):
        """Initialize attributes

        Parameters
        ----------
        train_files: list
            training set organised as a list of dictionaries with image label pairs
        val_files: list
            validation set organised as a list of dictionaries with image label pairs
        """
        self.train_files, self.val_files = train_files, val_files
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

    def get_train_loader(self):
        """Returns training dataloader, an iterable over training dataset with
        cache deterministic transforms' result during training

        Returns
        -------
        DataLoader
            MONAI dataloader for training data loading
        """
        # 'CacheDataset' used to accelerate training process
        train_ds = CacheDataset(data=self.train_files, transform=self.train_transforms, cache_rate=1.0, num_workers=4)
        # train_ds = Dataset(data=train_files, transform=train_transforms)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
        return train_loader

    def get_val_loader(self):
        """Returns validation dataloader, an iterable over validation dataset with
        cache deterministic transforms' result during validation

        Returns
        -------
        DataLoader
            MONAI dataloader for validation data loading
        """
        # 'CacheDataset' used to accelerate validation process
        val_ds = CacheDataset(data=self.val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=4)
        # val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
        return val_loader
