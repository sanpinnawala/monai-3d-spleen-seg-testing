import glob
import os
from argparse import ArgumentParser
from src.data import SpleenDataModule
from src.model import LitUNet
import lightning.pytorch as pl


def main():
    parser = ArgumentParser()

    # root directory argument
    parser.add_argument("--root_dir", type=str, default="/Users/sandunipinnawala/Documents/Git_Repos/monai-3d-spleen"
                                                        "-seg-testing")
    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    # path to data directory
    data_dir = os.path.join(args.root_dir, "data/Task09_Spleen")

    # data module
    spleen = SpleenDataModule(data_dir)
    spleen.setup(stage="test")
    # model
    model = LitUNet.load_from_checkpoint(args.root_dir+"/models/best_model.ckpt")

    # test model
    pl.Trainer().test(model, datamodule=spleen)


if __name__ == "__main__":
    main()
