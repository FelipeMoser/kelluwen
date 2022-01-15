from torch.nn.functional import conv1d, conv2d, conv3d, relu
from torch import sqrt, sum, cat, flatten
from .transforms import generate_kernel


def dsc(
    image,
    reference,
    smoothing_constant=0.01,
    reduction_channel="mean",
    type_output="dict",
):
    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction '{}'. Supported reductions: {}.".format(
                function, supported_reductions
            )
        )

    # Check that the smoothing constant is a number
    if not isinstance(smoothing_constant, (int, float)):
        raise TypeError("Smoothing constant must be a number.")

    # Calculate intersection
    dims = tuple(range(2, image.dim()))
    intersection = (image * reference).float().sum(dim=dims)
    union = image.float().sum(dim=dims) + reference.float().sum(dims)

    # Calculate dsc
    dsc = (2 * intersection + smoothing_constant) / (union + smoothing_constant)

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        dsc = dsc.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        dsc = dsc.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            f"Reduction '{reduction_channel}' not implemented! Please contact the developers."
        )

    # Return results
    if type_output == "raw":
        return dsc
    else:
        return {"dsc": dsc}


def iou(
    image,
    reference,
    smoothing_constant=0.01,
    reduction_channel="mean",
    type_output="dict",
):

    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction '{}'. Supported reductions: {}.".format(
                function, supported_reductions
            )
        )


    # Check that the smoothing constant is a number
    if not isinstance(smoothing_constant, (int, float)):
        raise TypeError("Smoothing constant must be a number.")

    # Calculate intersection
    dims = tuple(range(2, image.dim()))
    intersection = (image * reference).sum(dim=dims)
    union = image.sum(dim=dims) + reference.sum(dim=dims) - intersection

    # Calculate iou
    iou = (intersection + smoothing_constant) / (union + smoothing_constant)

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        iou = iou.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        iou = iou.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            "Reduction '{}' not implemented! Please contact the developers."
        )

    # Return results
    if type_output == "raw":
        return iou
    else:
        return {"iou": iou}


def mae(image, reference, reduction_channel="mean", type_output="dict"):
    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction '{}'. Supported reductions: {}.".format(
                function, supported_reductions
            )
        )

    # Calculate mean absolute error
    mae = (image - reference).abs().mean(dim=tuple(range(2, image.dim())))

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        mae = mae.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        mae = mae.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            "Reduction '{}' not implemented! Please contact the developers."
        )

    # Return results
    if type_output == "raw":
        return mae
    else:
        return {"mae": mae}


def mse(image, reference, reduction_channel="mean", type_output="dict"):
    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction '{}'. Supported reductions: {}.".format(
                function, supported_reductions
            )
        )

    # Calculate mean squared error
    mse = (image - reference).square().mean(dim=tuple(range(2, image.dim())))

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        mse = mse.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        mse = mse.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            "Reduction '{}' not implemented! Please contact the developers."
        )

    # Return results
    if type_output == "raw":
        return mse
    else:
        return {"mse": mse}


def pcc(
    image,
    reference,
    smoothing_constant=0.01,
    reduction_channel="mean",
    reduction_spatial="mean",
    kernel=None,
    type_kernel="gaussian",
    shape_kernel=None,
    sigma_kernel=None,
    type_output="dict",
):
    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()
    if kernel == None:
        if shape_kernel == None:
            shape_kernel = [5] * (image.dim() - 2)
        if sigma_kernel == None:
            sigma_kernel = [3] * (image.dim() - 2)

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that channel reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction type '{}'. Supported types: {}.".format(
                function, supported_reductions
            )
        )

    # Check that spatial reduction functions are supported
    if reduction_spatial not in supported_reductions:
        raise ValueError(
            "Unsupported spatial type '{}'. Supported types: {}".format(
                function, supported_reductions
            )
        )

    # Check that the smoothing constant is a number
    if not isinstance(smoothing_constant, (int, float)):
        raise TypeError("Smoothing constant must be a number.")

    # Generate kernel if required
    if kernel == None:
        kernel = generate_kernel(
            type_kernel=type_kernel,
            shape_kernel=shape_kernel,
            sigma_kernel=sigma_kernel,
        )["kernel"]
        kernel = cat(image.shape[1] * [kernel[None, None, :]]).to(image.device)

    # Select convolution type depending on dimensionality
    if image.dim() == 3:
        conv = conv1d
    elif image.dim() == 4:
        conv = conv2d
    else:
        conv = conv3d

    # Calculate means
    mean_x = conv(image, kernel, groups=image.shape[1])
    mean_y = conv(reference, kernel, groups=reference.shape[1])

    # Calculate standard deviations. Note that we use ReLU to remove small negatives from approximations
    std_x = sqrt(relu(conv(image ** 2, kernel, groups=image.shape[1]) - mean_x ** 2))
    std_y = sqrt(
        relu(conv(reference ** 2, kernel, groups=reference.shape[1]) - mean_y ** 2)
    )

    # Calculate covariance
    cov_xy = conv(image * reference, kernel, groups=image.shape[1]) - mean_x * mean_y

    # Calculate luminance, contrast, and structure
    pcc = (cov_xy + smoothing_constant) / (std_x * std_y + smoothing_constant)

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        pcc = pcc.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        pcc = pcc.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            "Reduction '{}' not implemented! Please contact the developers."
        )

    # Average over spatial dimensions if required
    if reduction_spatial == "none":
        pass
    else:
        dims = tuple(range(1 + (reduction_channel == "none"), pcc.dim()))
        if reduction_spatial == "mean":
            pcc = pcc.mean(dim=dims)

        elif reduction_spatial == "sum":
            pcc = pcc.sum(dim=dims)

        else:
            raise Exception(
                "Reduction '{}' not implemented! Please contact the developers."
            )

    # Return results
    if type_output == "raw":
        return pcc
    else:
        return {"pcc": pcc}


def ssim(
    image,
    reference,
    smoothing_constant=0.01,
    k1=0.01,
    k2=0.03,
    dynamic_range=1,
    reduction_channel="mean",
    reduction_spatial="mean",
    kernel=None,
    type_kernel="gaussian",
    shape_kernel=None,
    sigma_kernel=None,
    type_output="dict",
):

    # Retrieve required variables
    reduction_channel = reduction_channel.lower()
    type_output = type_output.lower()
    if kernel == None:
        if shape_kernel == None:
            shape_kernel = [5] * (image.dim() - 2)
        if sigma_kernel == None:
            sigma_kernel = [3] * (image.dim() - 2)

    # Define supported reductions
    supported_reductions = ("none", "mean", "sum")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check that channel reduction function is supported
    if reduction_channel not in supported_reductions:
        raise ValueError(
            "Unsupported channel reduction type '{}'. Supported types: {}.".format(
                function, supported_reductions
            )
        )

    # Check that spatial reduction functions are supported
    if reduction_spatial not in supported_reductions:
        raise ValueError(
            "Unsupported spatial type '{}'. Supported types: {}".format(
                function, supported_reductions
            )
        )
    # Calculate constants
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    # Generate kernel if required
    if kernel == None:
        kernel = generate_kernel(
            type_kernel=type_kernel,
            shape_kernel=shape_kernel,
            sigma_kernel=sigma_kernel,
        )["kernel"]
        kernel = cat(image.shape[1] * [kernel[None, None, :]]).to(image.device)

    # Select convolution type depending on dimensionality
    if image.dim() == 3:
        conv = conv1d
    elif image.dim() == 4:
        conv = conv2d
    else:
        conv = conv3d

    # Calculate means
    mean_x = conv(image, kernel, groups=image.shape[1])
    mean_y = conv(reference, kernel, groups=reference.shape[1])

    # Calculate standard deviations. Note that we use ReLU to remove small negatives from approximations
    std_x = sqrt(relu(conv(image ** 2, kernel, groups=image.shape[1]) - mean_x ** 2))
    std_y = sqrt(
        relu(conv(reference ** 2, kernel, groups=reference.shape[1]) - mean_y ** 2)
    )

    # Calculate covariance
    cov_xy = conv(image * reference, kernel, groups=image.shape[1]) - mean_x * mean_y

    # Calculate SSIM
    ssim = (
        (2 * mean_x * mean_y + c1)
        * (2 * cov_xy + c2)
        / ((mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2))
    )

    # Average over channels if required
    if reduction_channel == "none":
        pass
    elif reduction_channel == "mean":
        ssim = ssim.mean(dim=1, keepdim=True)
    elif reduction_channel == "sum":
        ssim = ssim.sum(dim=1, keepdim=True)
    else:
        raise Exception(
            "Reduction '{}' not implemented! Please contact the developers."
        )

    # Average over spatial dimensions if required
    if reduction_spatial == "none":
        pass
    else:
        dims = tuple(range(1 + (reduction_channel == "none"), ssim.dim()))
        if reduction_spatial == "mean":
            ssim = ssim.mean(dim=dims)

        elif reduction_spatial == "sum":
            ssim = ssim.sum(dim=dims)

        else:
            raise Exception(
                "Reduction '{}' not implemented! Please contact the developers."
            )

    # Return results
    if type_output == "raw":
        return ssim
    else:
        return {"ssim": ssim}


def kld(mean, logvar):
    # # Retrieve required variables
    # logvar = kwargs["logvar"]
    # mu = kwargs["mean"]

    # Calculate kld
    logvar = flatten(logvar, start_dim=1)
    mean = flatten(mean, start_dim=1)
    kld = -0.5 * sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

    # Return results
    return {"kld": kld}
