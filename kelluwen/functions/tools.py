from typeguard import typechecked
from typing import Union, Dict, List, Tuple
import math
import torch as tt
from .transforms import scale_features
import matplotlib
import matplotlib.pyplot as plt


@typechecked
def centre_crop(
    image: tt.Tensor,
    shape_output: Union[tt.Size, List[int]],
    value_pad: int = 0,
    type_order: str = "after",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Crops image tensor around its centre

    Parameters
    ----------
    image : torch.Tensor
        Image to be cropped. Must be of shape (batch, channel, *).

    shape_output : torch.Size, List[int]
        Output shape of cropped image. Must have the same number of spatial dimensions as the image.

    value_pad : int, optional (default=0)
        Value for padding regions outside initial image.

    type_order : str, optional (default="after)
        This order defines which side will be padded more in the case that asymmetric padding is needed.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    image_cropped : torch.Tensor
    """

    # Validate arguments
    if image.dim() < 3:
        raise ValueError(
            f"expected input image to be at least 3D, got {image.dim()!r}D instead"
        )
    if len(shape_output) + 2 != image.dim():
        raise ValueError(f"shape_output doesn't match image.shape")
    if type_order.lower() not in ("before", "after"):
        raise ValueError(f"unknown value {type_order!r} for type_order")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Retrieve image device
    device = image.device

    # Cast shapes to tuples
    shape_source = tuple(image.shape)
    shape_output = tuple(shape_output)

    # Create padded tensor
    image_cropped = tt.full(
        size=(*shape_source[:2], *shape_output),
        fill_value=value_pad,
        device=device,
        dtype=image.dtype,
    )

    # Calculate shape difference
    difference = [(x - y) / 2 for x, y in zip(shape_output, shape_source[2:])]
    if type_order == "before":
        difference = [math.ceil(d) for d in difference]
    elif type_order == "after":
        difference = [math.floor(d) for d in difference]

    # Generate corresponding indices for unpadded (source) and padded (output) tensors
    diff_source = [abs(min(d, 0)) for d in difference]
    diff_output = [max(d, 0) for d in difference]

    idx_source = [
        slice(dx, min(xs + 1, dx + ys))
        for dx, ys, xs in zip(diff_source, shape_output, shape_source[2:])
    ]
    idx_output = [
        slice(dy, min(ys + 1, dy + xs))
        for dy, ys, xs in zip(diff_output, shape_output, shape_source[2:])
    ]
    idx_source = [slice(None), slice(None)]
    idx_output = [slice(None), slice(None)]
    for dx, dy, xs, ys in zip(diff_source, diff_output, shape_source[2:], shape_output):
        idx_source.append(slice(dx, min(xs + 1, dx + ys)))
        idx_output.append(slice(dy, min(ys + 1, dy + xs)))

    # Populate padded tensor
    image_cropped[idx_output] = image[idx_source]

    # Return results
    if type_output == "positional":
        return image_cropped
    else:
        return {"image": image_cropped}


@typechecked
def get_midplanes(
    image: tt.Tensor,
    type_output: str = "positional",
) -> Union[Tuple, Dict[str, tt.Tensor]]:
    """Returns midplanes of image tensor

    Parameters
    ----------
    image : torch.Tensor
        Input image. Must be of shape (batch, channel, x, y, z).

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    image_midplanes : Tuple[torch.Tensor]
    """
    # Validate arguments
    if image.dim() != 5:
        raise ValueError(f"expected a 5D image, got {image.dim()!r}D instead")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Get midplanes
    image_xy = image[..., image.shape[-1] // 2]
    image_xz = image[..., image.shape[-2] // 2, :]
    image_yz = image[..., image.shape[-3] // 2, :, :]

    # Return results
    if type_output == "positional":
        return image_xy, image_xz, image_yz
    else:
        return {"image_xy": image_xy, "image_xz": image_xz, "image_yz": image_yz}


@typechecked
def show_midplanes(
    image: tt.Tensor,
    show: bool = True,
    title: str = "",
    type_coordinates: str = "xyz",
    type_feature_scaling: str = "",
    type_backend: str = "",
    type_output: str = "positional",
) -> None:
    """Shows midplanes of image tensor

    Parameters
    ----------
    image : torch.Tensor
        Input image. Must be of shape (batch, channel, x, y, z).

    title : str, optional (default="")
        Title of figure.

    show : bool, optional (default=True)
        Controls whether image should be shown or accumulated. When True, all accumulated images will be shown.

    type_coordinates : str, optional (default="xyz")
        Defines the axes orientation of the midplanes. Available: "xyz", "ras".

    type_feature_scaling: str, optional (default="")
        Defines whether the features of the midplanes should be scaled before showing. Available: "" (no scaling), "min_max", "mean_norm", "z_score", "unit_length".

    type_backend: str, optional (default="")
        Defines the backend to use for showing the midplanes. Available: "" (default), "web".

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    image_midplanes : Tuple[torch.Tensor]
    """
    # Validate arguments
    if image.dim() != 5:
        raise ValueError(f"expected a 5D image, got {image.dim()!r}D instead")
    if type_feature_scaling.lower() not in (
        "",
        "min_max",
        "mean_norm",
        "z_score",
        "unit_length",
    ):
        raise ValueError(
            f"unknown value {type_feature_scaling!r} for type_feature_scaling"
        )
    if type_backend.lower() not in ("", "web"):
        raise ValueError(f"unknown value {type_backend!r} for type_backend")

    if type_coordinates.lower() not in ("xyz", "ras"):
        raise ValueError(f"unknown value {type_coordinates!r} for type_coordinates")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Get midplanes
    images = get_midplanes(image.detach().cpu())

    # Scale features if required
    if type_feature_scaling != "":
        images = [
            scale_features(image=img, type_scaling=type_feature_scaling)
            for img in images
        ]

    # Remove batch and channel dimensions if required
    images = [img[0, 0] for img in images]

    # Create figure
    if type_coordinates == "ras":
        images = [img.T for img in images]
    height_figure = max(img.shape[0] for img in images)
    width_figure = sum(img.shape[1] for img in images)
    shape_figure = [width_figure, height_figure]
    shape_figure = [x / max(shape_figure) * 15 for x in shape_figure]

    # Plot midplanes
    if type_backend != "":
        if type_backend.lower() == "web":
            matplotlib.use("WebAgg")
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=shape_figure,
        gridspec_kw=dict(width_ratios=([img.shape[1] for img in images])),
        subplot_kw=dict(anchor="NW"),
        tight_layout=True,
    )
    fig.suptitle(title)
    fig.tight_layout()
    if type_coordinates == "xyz":
        for i, (img, axes) in enumerate(zip(images, ["XY", "XZ", "YZ"])):
            axs[i].imshow(img, cmap="gray", origin="upper")
            axs[i].set(xticks=[0, img.shape[1] - 1], yticks=[0, img.shape[0] - 1])
            axs[i].set(title=axes, xlabel=axes[1], ylabel=axes[0])

    elif type_coordinates == "ras":
        for i, (img, axes) in enumerate(zip(images, ["PRAL", "IRSL", "IASP"])):
            axs[i].imshow(img, cmap="gray", origin="lower")
            axs[i].set(xlim=axs[i].get_xlim()[::-1], xticks=[], yticks=[])
            axs[i].set_xlabel(axes[0], ha="center")
            axs[i].set_ylabel(axes[1], rotation=0, va="center")
            axx1 = axs[i].secondary_xaxis("top")
            axx1.set_xlabel(axes[2], ha="center")
            axx1.set_xticks([])
            axx1 = axs[i].secondary_yaxis("right")
            axx1.set_ylabel(axes[3], rotation=0, va="center")
            axx1.set_yticks([])

    # Show if required
    if show:
        plt.show()
