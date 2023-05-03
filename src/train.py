import glob
import os
from src.data_loader import Loader
from src.model import LitUNet
import lightning.pytorch as pl


def main():
    # path to data directory
    root_dir = "/Users/sandunipinnawala/Documents/Git_Repos/monai-3d-spleen-seg-testing"
    data_dir = os.path.join(root_dir, "data/Task09_Spleen")

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    # arrange data as a list of dictionaries
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(train_images, train_labels)]
    # split to training and validation sets (training set split ration 0.8)
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    loader = Loader(train_files, val_files)
    # training and validation dataloaders
    train_dataloader = loader.get_train_loader()
    val_dataloader = loader.get_val_loader()

    # model
    model = LitUNet()

    # train model
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
