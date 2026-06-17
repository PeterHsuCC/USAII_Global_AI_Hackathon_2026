from torch import nn


def freeze(module: nn.Module) -> None:
    """Set requires_grad=False on every parameter in `module`.

    Used to keep the GoEmotions classifier frozen during Stage 1/2
    training (Section 12.1: "this loss only updates the learned logistic
    weights theta and bias b_m, not any emotion feature extractor"), and
    to freeze the shared encoder and task heads before Stage 3 calibration
    (Section 12.3: "Freeze the shared encoder and task heads").
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)
