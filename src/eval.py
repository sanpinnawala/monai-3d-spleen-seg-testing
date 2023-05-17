import glob
import os
from argparse import ArgumentParser
from src.loader import Loader
from src.model import LitUNet
import lightning.pytorch as pl


def main():
    parser = ArgumentParser()

    # root directory argument
    parser.add_argument("--root_dir", type=str, default="/{path}/monai-3d-spleen-seg-testing")
    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    # path to data directory
    data_dir = os.path.join(args.root_dir, "data/Task09_Spleen")

    test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
    # testing set
    test_files = [{"image": image} for image in test_images]

    loader = Loader()
    # testing dataloader
    test_dataloader = loader.get_test_loader(test_files)

    # model
    model = LitUNet.load_from_checkpoint(args.root_dir+"/models/best_model.ckpt")

    # test model
    pl.Trainer().test(model, test_dataloader)


if __name__ == "__main__":
    main()
