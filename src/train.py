import glob
import os
from argparse import ArgumentParser
from src.loader import Loader
from src.model import LitUNet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    parser = ArgumentParser()

    # root directory argument
    parser.add_argument("--root_dir", type=str, default="/Users/sandunipinnawala/Documents/Git_Repos/monai-3d-spleen"
                                                        "-seg-testing")
    parser.add_argument('--seed', type=int, default=0)

    # model specific arguments
    parser.add_argument('--spatial_dims', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--channels', type=tuple, default=(16, 32, 64, 128, 256))
    parser.add_argument('--strides', type=tuple, default=(2, 2, 2, 2))
    parser.add_argument('--num_res_units', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--roi_size', type=tuple, default=(160, 160, 160))
    parser.add_argument('--sw_batch_size', type=int, default=4)

    # parse the user inputs and defaults
    args = parser.parse_args([])
    # path to data directory
    data_dir = os.path.join(args.root_dir, "data/Task09_Spleen")
    # for reproducibility
    pl.seed_everything(args.seed)

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    # arrange data as a list of dictionaries
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                  zip(train_images, train_labels)]
    # split to training and validation sets (training set split ration 0.8)
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    loader = Loader()
    # training and validation dataloaders
    train_dataloader = loader.get_train_loader(train_files)
    val_dataloader = loader.get_val_loader(val_files)

    # model
    model = LitUNet(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=args.root_dir+"/models/",
        filename="best_model",
    )

    # train model
    trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
