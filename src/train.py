import os
from argparse import ArgumentParser
from src.data import SpleenDataModule
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

    # data module
    spleen = SpleenDataModule(data_dir)
    spleen.setup(stage="fit")
    # model
    model = LitUNet(args)
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=args.root_dir+"/models/",
        filename="best_model",
    )

    # train model
    trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=spleen)


if __name__ == "__main__":
    main()
