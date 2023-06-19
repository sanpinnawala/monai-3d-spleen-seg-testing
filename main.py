from lightning.pytorch.cli import LightningCLI

from src.data import SpleenDataModule
from src.model import LitUNet


def main():
    cli = LightningCLI(LitUNet, SpleenDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
