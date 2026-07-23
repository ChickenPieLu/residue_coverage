import segmentation_models_pytorch as smp

MODEL_NAME = "smp_unet_resnet34_imagenet"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"


def make_model(encoder_weights=ENCODER_WEIGHTS):
    """Build the model; use no encoder weights before loading a checkpoint."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
