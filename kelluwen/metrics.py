import torch
from torch.nn import functional
from .misc import gaussian_kernel

import sys


def ssim(
    target,
    prediction,
    kernel=None,
    size=5,
    sigma=3,
    dynamic_range=1,
    k=[0.01, 0.03],
    reduction=None,
):
    """ Returns the SSIM index, luminance, contrast, and structure between target and prediction

    Parameters
    ----------
    target : torch.Tensor
        target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
    prediction : torch.Tensor
        prediction tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
    kernel : torch.Tensor, optional
        kernel tensor for calculating moving box, by default None
    size : (int, list, or tuple), optional
        size or list of sizes for calculating moving window/box, by default 5
    sigma : (int, float, or tuple), optional
        sigma or list of sigmas for calculating moving window/box kernel, by default 3
    dynamic_range : int, optional
        intensity dynamic range of tensor and prediction tensors, by default 1
    k : list, optional
        Stabilizer constants for SSIM, by default [0.01, 0.03]
    reduction : (None, str), optional
        reduction mode used , "sum" or "mean", or None for no reduction, by default None

    Returns
    -------
    torch.Tensor
        ssim measurement between target and prediction
    torch.Tensor
        luminance measurement between target and prediction
    torch.Tensor
        contrast measurement between target and prediction
    torch.Tensor
        structure measurement between target and prediction
    """
    # Check target parameter
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor")
    elif len(target.shape) > 5:
        raise ValueError("target must have a max. of 3 spatial dimensions")
    elif torch.min(target) < 0:
        raise ValueError("target values must be non-negative")
    # Check prediction parameter
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a tensor")
    elif prediction.shape != target.shape:
        raise ValueError("prediction must same shape as target")
    elif torch.min(prediction) < 0:
        raise ValueError("prediction values must be non-negative")
    elif type(prediction) != type(target):
        raise TypeError("prediction must be the same type as target")
    # Check kernel parameter if provided
    if kernel is not None:
        if type(kernel) != type(target):
            raise TypeError("kernel must be the same type as target")
        elif len(kernel.shape) != len(target.shape):
            raise ValueError("kernel must have the same number of dimensions as target")
        elif kernel.shape[0] != target.shape[1]:
            raise ValueError("kernel group size must match target channel size")
    # Check size and sigma parameters if no kernel was provided
    else:
        # Check size parameter
        if not isinstance(size, (int, list, tuple)):
            raise TypeError("size must be an integer or a list (or tuple) of integers")
        elif isinstance(size, int):
            size = [size] * len(target.shape[2:])  # Convert size to list if it's an int
        if any((not isinstance(elem, int) for elem in size)):
            raise TypeError("size elements must be integers")
        elif any((elem % 2 == 0 or elem <= 0 for elem in size)):
            raise ValueError("size elements must be odd and positive")
        elif len(size) != len(target.shape[2:]):
            raise ValueError(
                "size must have the same number of elements as spatial dimensions of target"
            )
        # Check sigma parameter
        if not isinstance(sigma, (int, float, list, tuple)):
            raise TypeError("sigma must be an number or a list (or tuple) of numbers")
        elif isinstance(sigma, (int, float)):
            sigma = [sigma] * len(
                target.shape[2:]
            )  # Convert sigma to list if it's an num.
        if any((not isinstance(elem, (int, float)) for elem in sigma)):
            raise TypeError("sigma elements must be numbers")
        elif any((elem <= 0 for elem in sigma)):
            raise ValueError("sigma elements must be positive")
        elif len(sigma) != len(target.shape[2:]):
            raise ValueError(
                "sigma must have the same number of elements as spatial dimensions of target"
            )
    # Check dynamic_range parameter
    if not isinstance(dynamic_range, (int, float)):
        raise TypeError("dynamic_range must be a number")
    elif dynamic_range <= 0:
        raise ValueError("dynamic_range must be positive")
    # Check k parameter
    if not isinstance(k, list):
        raise TypeError("k must be a list")
    elif len(k) != 2 or any(not isinstance(elem, (int, float)) for elem in k):
        raise ValueError("k must be a list containing two numbers")
    # Check reduction parameter
    if reduction is not None and reduction not in ["sum", "mean"]:
        raise ValueError('reduction must be either None, "mean", or "sum"')

    # Calculate constants
    k1, k2 = k
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    c3 = c2 / 2
    # Generate kernel
    if kernel is None:
        kernel = gaussian_kernel(size, sigma, dtype=type(target))  # Get kernel
        kernel /= torch.sum(kernel)  # Normalise kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        # Match kernel group size to target channel size
        if kernel.shape[0] != target.shape[1]:
            kernel_repeat_shape = [1] * len(target.shape)
            kernel_repeat_shape[0] = target.shape[1]
            kernel = kernel.repeat(kernel_repeat_shape)
    # Obtain convolution function for current spatial dimensions
    if len(target.shape[2:]) == 1:
        conv = functional.conv1d
    elif len(target.shape[2:]) == 2:
        conv = functional.conv2d
    else:
        conv = functional.conv3d
    # Calcualte means and sigmas
    groups = target.shape[1]
    mu_x = conv(target, kernel, groups=groups)
    mu_y = conv(prediction, kernel, groups=groups)
    sigma_x = torch.sqrt(
        functional.relu(conv(target ** 2, kernel, groups=groups) - mu_x ** 2)
    )  # ReLU to avoid small negatives from approximations
    sigma_y = torch.sqrt(
        functional.relu(conv(prediction ** 2, kernel, groups=groups) - mu_y ** 2)
    )  # ReLU to avoid small negatives from approximations
    sigma_xy = conv(target * prediction, kernel, groups=groups) - mu_x * mu_y

    # Calculate luminance, contrast, and structure
    luminance = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x ** 2 + sigma_y ** 2 + c2)
    structure = (sigma_xy + c3) / (sigma_x * sigma_y + c3)
    # Calculate SSIM
    ssim_index = luminance * contrast * structure
    # Reduce spatial dimensions if required
    if reduction == "sum":
        luminance = torch.flatten(luminance, 2).sum(-1)
        contrast = torch.flatten(contrast, 2).sum(-1)
        structure = torch.flatten(structure, 2).sum(-1)
        ssim_index = torch.flatten(ssim_index, 2).sum(-1)
    elif reduction == "mean":
        luminance = torch.flatten(luminance, 2).mean(-1)
        contrast = torch.flatten(contrast, 2).mean(-1)
        structure = torch.flatten(structure, 2).mean(-1)
        ssim_index = torch.flatten(ssim_index, 2).mean(-1)
    # Return channel-wise ssim, luminance, contrast, and structure
    return ssim_index, luminance, contrast, structure


def pearson_correlation_coefficient(
    target, prediction, mask=None, batch_reduction="mean", channel_reduction="mean"
):
    """ Returns the Pearson's Correlation Coefficient between target and prediction

    Parameters
    ----------
    target : torch.Tensor
        target tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
    prediction : torch.Tensor
        prediction tensor of shape (B, C, *) where * is one, two, or three spatial dimensions
    mask: torch.bool, optional
            mask tensor of shape (B, C, *) where * is one, two, or three spatial dimensions, used to select the parts of the target/prediction that will be generate the loss, by default None
    batch_reduction : (None, "mean", "sum"), optional
        reduction mode used along batch dimension, "sum" or "mean", or None for no reduction, by default "mean"
    channel_reduction : (None, "mean", "sum"), optional
        reduction mode used along channel dimension, "sum" or "mean", or None for no reduction, by default "mean"

    Returns
    -------
    torch.Tensor
        Pearson's Correlation Coefficient between target and prediction
    """
    # Check target parameter
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor")
    elif len(target.shape) < 3 or len(target.shape) > 5:
        raise ValueError(
            "target must be of shape (B, C, *) where * is one, two, or three spatial dimensions"
        )
    # Check prediction parameter
    if type(prediction) != type(target):
        raise TypeError("prediction must be the same type as target")
    elif prediction.shape != target.shape:
        raise ValueError("prediction must same shape as target")
    # Check batch_reduction parameter
    if batch_reduction not in [None, "sum", "mean"]:
        raise ValueError('batch_reduction must be either None, "mean", or "sum"')
    # Check channel_reduction parameter
    if channel_reduction not in [None, "sum", "mean"]:
        raise ValueError('channel_reduction must be either None, "mean", or "sum"')
    # Check mask parameter
    if mask is not None:
        if not isinstance(mask, (torch.cuda.BoolTensor, torch.BoolTensor)):
            raise TypeError("mask must be the of type torch.BoolTensor")
        elif mask.shape != target.shape:
            raise ValueError("mask must have the same shape as target")

    # Flatten the inputs
    target = target.flatten(2)
    prediction = prediction.flatten(2)

    # Calculate masked Pearson's Correlation Coefficient if required
    if mask is not None:
        mask = mask.flatten(2)
        pcc = torch.zeros((target.shape[0], target.shape[1], 1)).to(target)
        for batch in range(target.shape[0]):
            for channel in range(target.shape[1]):
                # Apply mask
                masked_tar = target[batch, channel][mask[batch, channel]]
                masked_pred = prediction[batch, channel][mask[batch, channel]]

                # Calculate the means of the masked target and prediction
                masked_mu_tar = masked_tar.mean()
                masked_mu_pred = masked_pred.mean()

                # Calculate the standard deviations of the masked target and prediction
                masked_std_tar = masked_tar.std()
                masked_std_pred = masked_pred.std()

                # Calculate the covariance of the masked target and prediction
                masked_cov_tar_pred = torch.sum(
                    (masked_tar - masked_mu_tar) * (masked_pred - masked_mu_pred)
                ) / (masked_tar.shape[-1] - 1)

                # Calculate Pearson's Correlation Coefficient of the masked target and prediction
                masked_pcc = masked_cov_tar_pred / (masked_std_tar * masked_std_pred)

                # Store the masked Pearson's Correlation Coefficient
                pcc[batch, channel] = masked_pcc

    else:
        # Calculate the means
        mu_tar = target.mean(2, keepdim=True)
        mu_pred = prediction.mean(2, keepdim=True)

        # Calcualte the standard deviations
        std_tar = target.std(2, keepdim=True)
        std_pred = prediction.std(2, keepdim=True)

        # Calculate the covariances
        cov_tar_pred = torch.sum(
            (target - mu_tar) * (prediction - mu_pred), 2, keepdim=True
        ) / (target.shape[-1] - 1)

        # Calculate Pearson's Correlation Coefficient
        pcc = cov_tar_pred / (std_tar * std_pred)

    # Reduce channels if required
    if channel_reduction == "sum":
        pcc = pcc.sum(1)
    elif channel_reduction == "mean":
        pcc = pcc.mean(1)

    # Reduce batch if required
    if batch_reduction == "sum":
        pcc = pcc.sum(0)
    elif batch_reduction == "mean":
        pcc = pcc.mean(0)
    return pcc
