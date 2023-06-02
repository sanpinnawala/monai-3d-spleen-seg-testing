import lightning.pytorch as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, AsDiscrete


class LitUNet(pl.LightningModule):
    """
    Class that defines LighteningModule

    Attributes
    ----------
    model : UNet
        MONAI UNet as base model architecture
    loss_function : DiceLoss
        MONAI DiceLoss with softmax as prediction function
    dice_metric : DiceMetric
        MONAI DiceMetric to compute mean dice
    post_pred : Compose
        callable to transform model outputs to discrete values
        in validation loop
    post_label : Compose
        callable to transform labels to discrete values
        in validation loop

    Methods
    -------
    forward(x):
        return model outputs following forward pass
    configure_optimizers():
        defines and returns the optimizer object
    training_step(batch, batch_idx):
        defines training loop
    validation_step(batch, batch_idx):
        defines validation loop
    test_step(batch, batch_idx):
        defines test loop
    """
    def __init__(self, hparams):
        """Initialize attributes

        Parameters
        ----------
        hparams : Any
            user inputs and defaults
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def forward(self, x):
        """ Forward pass

        Parameters
        ----------
        x : Any
            model inputs

        Returns
        -------
        Tensor
            model outputs
        """
        return self.model(x)

    def configure_optimizers(self):
        """ Define the optimizer

        Returns
        -------
        Adam
            pytorch Adam optimizer object
        """
        # Create Adam optimizer object with learning rate of 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """ Training loop

        Parameters
        ----------
        batch : Any
            single batch from training set
        batch_idx : int
            batch index

        Returns
        -------
        Any
            computed dice loss
        """
        # get training inputs and labels
        inputs, labels = (
            batch["image"],
            batch["label"],
        )
        # forward pass
        outputs = self.model(inputs)
        # calculate loss
        loss = self.loss_function(outputs, labels)
        # log statistics
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation loop

        Parameters
        ----------
        batch : Any
            single batch from validation set
        batch_idx : int
            batch index

        Returns
        -------
        Any
            computed dice metric
        """
        # get validation inputs and labels
        val_inputs, val_labels = (
            batch["image"],
            batch["label"],
        )
        # define spatial window size for inference
        roi_size = self.hparams.roi_size
        # define batch size to run window slices
        sw_batch_size = self.hparams.sw_batch_size

        # perform MONAI sliding window inference
        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, self.model)

        # decollate output data and apply post processing transforms
        val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [self.post_label(i) for i in decollate_batch(val_labels)]

        # compute mean dice
        val_loss = self.dice_metric(y_pred=val_outputs, y=val_labels)
        # log statistics
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """ Test loop

        Parameters
        ----------
        batch : Any
            single batch from test set
        batch_idx : int
            batch index
        """
        # get test inputs
        test_inputs = batch["image"]
        # define spatial window size for inference
        roi_size = self.hparams.roi_size
        # define batch size to run window slices
        sw_batch_size = self.hparams.sw_batch_size

        # perform MONAI sliding window inference
        test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, self.model)
        # decollate output data and apply post processing transforms
        test_outputs = [self.post_pred(i) for i in decollate_batch(test_outputs)]

