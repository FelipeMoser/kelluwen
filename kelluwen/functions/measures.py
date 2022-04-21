import torch.nn.functional as ff
import torch as tt
from .transforms import generate_kernel
from typeguard import typechecked
from typing import Union, Dict


@typechecked
def measure_dsc(
    image: tt.Tensor,
    reference: tt.Tensor,
    value_smooth: float = 0.01,
    reduction_channel: str = "mean",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Measures the Dice Similarity Coefficient (DSC) between two tensors. The DSC, also known as the Sørensen–Dice coefficient, is calculated using the method originally described in [1] and [2].

    Parameters
    ----------
    image : torch.bool
        Image being compared. Must be of shape (batch, channel, *).

    reference : torch.bool
        Reference against which the image is compared. Must be of shape (batch, channel, *).

    value_smooth : float, optional (default=0.01)
        Value added to the numerator and denominator in order to avoid division by zero if both image and reference contain no True values. 

    reduction_channel : str, optional (default="mean")
        Determines whether the channel dimension of the DSC tensor is kept or combined. If set to "none", the channel dimension of the DSC is kept. If set to "mean" or "sum", the channel dimension of the DSC is averaged or summed, respectively.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.
    
    Returns
    -------
    dsc : torch.Tensor
        Tensor of shape (batch, channel) if reduction_channel=="none". Otherwise, tensor of shape (batch,).

    References
    -----    
    [1] Sorensen, T. A. (1948). A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on Danish commons. Biol. Skar., 5, 1-34.
    
    [2] Dice, L. R. (1945). Measures of the amount of ecologic association between species. Ecology, 26(3), 297-302."""

    # Validate arguments
    if reduction_channel.lower() not in ("none", "mean", "sum"):
        raise ValueError(f"unknown value {reduction_channel!r} for reduction_channel")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Calculate numerator and denominator
    image = image.flatten(start_dim=2)
    reference = reference.flatten(start_dim=2)
    numerator = (image * reference).sum(dim=-1)
    denominator = image.sum(dim=-1) + reference.sum(-1)

    # Calculate the DSC
    dsc = (2 * numerator + value_smooth) / (denominator + value_smooth)

    # Combine channels if required
    if reduction_channel == "mean":
        dsc = dsc.mean(dim=1)
    elif reduction_channel == "sum":
        dsc = dsc.sum(dim=1)

    # Return results
    if type_output == "positional":
        return dsc
    else:
        return {"dsc": dsc}


@typechecked
def measure_cd(
    image: tt.Tensor,
    reference: tt.Tensor,
    reduction_channel: str = "mean",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Measures the Centroid Distance (CD) between two tensors. The CD is the Euclidean distance between the centroids of the image and reference tensors, weighted by the intensities.

    Parameters
    ----------
    image : torch.bool
        Image being compared. Must be of shape (batch, channel, *).

    reference : torch.bool
        Reference against which the image is compared. Must be of shape (batch, channel, *).

    reduction_channel : str, optional (default="mean")
        Determines whether the channel dimension of the CD tensor is kept or combined. If set to "none", the channel dimension of the CD is kept. If set to "mean" or "sum", the channel dimension of the CD is averaged or summed, respectively.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.
    
    Returns
    -------
    cd : torch.Tensor
        Tensor of shape (batch, channel) if reduction_channel=="none". Otherwise, tensor of shape (batch,).
    """

    # Validate arguments
    if reduction_channel.lower() not in ("none", "mean", "sum"):
        raise ValueError(f"unknown value {reduction_channel!r} for reduction_channel")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Generate grids
    grids = tt.meshgrid(*[tt.arange(x) for x in image.shape[2:]], indexing="ij")
    grids = tt.stack(grids, dim=-1).flatten(end_dim=-2)[None, None, :].to(image.device)

    # Calculate centroids
    image = image.flatten(start_dim=2).unsqueeze(-1)
    reference = reference.flatten(start_dim=2).unsqueeze(-1)
    centroid_image = (grids * image).sum(dim=2) / image.sum(dim=2)
    centroid_reference = (grids * reference).sum(dim=2) / reference.sum(dim=2)

    # Calculate distance between centroids
    cd = tt.sqrt(((centroid_reference - centroid_image) ** 2).sum(dim=-1))

    # Combine channels if required
    if reduction_channel == "mean":
        cd = cd.mean(dim=1)
    elif reduction_channel == "sum":
        cd = cd.sum(dim=1)

    # Return results
    if type_output == "positional":
        return cd
    else:
        return {"cd": cd}


@typechecked
def measure_iou(
    image: tt.Tensor,
    reference: tt.Tensor,
    value_smooth: float = 0.01,
    reduction_channel: str = "mean",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Measures the Intersection Over Union (IOU) between two tensors. The IOU, also known as the Jaccard index, is calculated using the method originally described in [1].

    Parameters
    ----------
    image : torch.bool
        Image being compared. Must be of shape (batch, channel, *).

    reference : torch.bool
        Reference against which the image is compared. Must be of shape (batch, channel, *).

    value_smooth : float, optional (default=0.01)
        Value added to the numerator and denominator in order to avoid division by zero if both image and reference contain no True values. 

    reduction_channel : str, optional (default="mean")
        Determines whether the channel dimension of the IOU tensor is kept or combined. If set to "none", the channel dimension of the IOU is kept. If set to "mean" or "sum", the channel dimension of the IOU is averaged or summed, respectively.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.
    
    Returns
    -------
    iou : torch.Tensor
        Tensor of shape (batch, channel) if reduction_channel=="none". Otherwise, tensor of shape (batch,).

    References
    -----    
    [1] Jaccard, P. (1912). The distribution of the flora in the alpine zone. 1. New phytologist, 11(2), 37-50.
    """

    # Validate arguments
    if reduction_channel.lower() not in ("none", "mean", "sum"):
        raise ValueError(f"unknown value {reduction_channel!r} for reduction_channel")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Calculate intersection and union
    image = image.flatten(start_dim=2)
    reference = reference.flatten(start_dim=2)
    intersection = (image * reference).sum(dim=-1)
    union = image.sum(dim=-1) + reference.sum(-1)

    # Calculate the IOU
    iou = (2 * intersection + value_smooth) / (union + value_smooth)

    # Combine channels if required
    if reduction_channel == "mean":
        iou = iou.mean(dim=1)
    elif reduction_channel == "sum":
        iou = iou.sum(dim=1)

    # Return results
    if type_output == "positional":
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
    if image.dim() > 2:
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
    else:
        mae = (image - reference).abs().mean(dim=1)

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
        kernel = tt.cat(image.shape[1] * [kernel[None, None, :]]).to(image.device)

    # Select convolution type depending on dimensionality
    if image.dim() == 3:
        conv = ff.conv1d
    elif image.dim() == 4:
        conv = ff.conv2d
    else:
        conv = ff.conv3d

    # Calculate means
    mean_x = conv(image, kernel, groups=image.shape[1])
    mean_y = conv(reference, kernel, groups=reference.shape[1])

    # Calculate standard deviations. Note that we use ReLU to remove small negatives from approximations
    std_x = tt.sqrt(
        ff.relu(conv(image ** 2, kernel, groups=image.shape[1]) - mean_x ** 2)
    )
    std_y = tt.sqrt(
        ff.relu(conv(reference ** 2, kernel, groups=reference.shape[1]) - mean_y ** 2)
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


def sc(
    image, reduction_channel="mean", type_output="dict",
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

    # Check image
    if image.dim() != 5:
        raise ValueError(
            "Unsuported image dimensionality. Currently only image.dim()=5 is supported"
        )

    # Calculate symmetry coefficient
    sc_x = measure_dsc(
        image=image[:, :, : image.shape[2] // 2, :, :],
        reference=tt.flip(image[:, :, -(image.shape[2] // 2) :, :, :], dims=[2]),
        reduction_channel="none",
        type_output="raw",
    )
    sc_y = measure_dsc(
        image=image[:, :, :, : image.shape[3] // 2, :],
        reference=tt.flip(image[:, :, :, -(image.shape[3] // 2) :, :], dims=[3]),
        reduction_channel="none",
        type_output="raw",
    )
    sc_z = measure_dsc(
        image=image[:, :, :, :, : image.shape[4] // 2],
        reference=tt.flip(image[:, :, :, :, -(image.shape[4] // 2) :], dims=[4]),
        reduction_channel="none",
        type_output="raw",
    )
    sc = tt.cat([sc_x, sc_y, sc_z], dim=-1)

    # Average over channels if required
    if sc.dim() > 2:
        if reduction_channel == "none":
            pass
        elif reduction_channel == "mean":
            sc = sc.mean(dim=1, keepdim=True)
        elif reduction_channel == "sum":
            sc = sc.sum(dim=1, keepdim=True)
        else:
            raise Exception(
                f"Reduction '{reduction_channel}' not implemented! Please contact the developers."
            )

    # Return results
    if type_output == "raw":
        return sc
    else:
        return {"sc": sc}


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
        kernel = tt.cat(image.shape[1] * [kernel[None, None, :]]).to(image.device)

    # Select convolution type depending on dimensionality
    if image.dim() == 3:
        conv = ff.conv1d
    elif image.dim() == 4:
        conv = ff.conv2d
    else:
        conv = ff.conv3d

    # Calculate means
    mean_x = conv(image, kernel, groups=image.shape[1])
    mean_y = conv(reference, kernel, groups=reference.shape[1])

    # Calculate standard deviations. Note that we use ReLU to remove small negatives from approximations
    std_x = tt.sqrt(
        ff.relu(conv(image ** 2, kernel, groups=image.shape[1]) - mean_x ** 2)
    )
    std_y = tt.sqrt(
        ff.relu(conv(reference ** 2, kernel, groups=reference.shape[1]) - mean_y ** 2)
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
    logvar = tt.flatten(logvar, start_dim=1)
    mean = tt.flatten(mean, start_dim=1)
    kld = -0.5 * sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

    # Return results
    return {"kld": kld}
