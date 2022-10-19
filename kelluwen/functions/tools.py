import torch as tt
from typeguard import typechecked
from typing import Union, Dict, List, Tuple
import math


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

    value_pad: int, optional (default=0)
        Value for padding regions outside initial image.

    type_order: str, optional (default="after)
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
    image_midplanes : List[torch.Tensor]
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


def show_midplanes(
    image, title="", show=True, type_coordinates="xyz", type_scaling=None
):

    # Define supported coordinates type
    supported_coordinates = ("ras", "xyz")

    # Define supported feature scaling type
    supported_scaling = (None, "min_max", "mean_norm", "z_score", "unit_length")

    # Check that coordinates type is supported
    if type_coordinates not in supported_coordinates:
        raise ValueError(
            f"Unknown coordinates type '{type_coordinates}'. Supported types: {supported_coordinates}"
        )

    # Check that feature scaling type is supported
    if type_scaling not in supported_scaling:
        raise ValueError(
            f"Unknown feature scaling type '{type_scaling}'. Supported types: {supported_scaling}"
        )

    # Check title
    if title is not None:
        if not isinstance(title, str):
            raise TypeError(f"Title must be a string, got {type(title)} instead.")

    # Check show
    if not isinstance(show, bool):
        raise TypeError(f"Show must be a boolean, got {type(show)} instead.")

    # Get midplanes
    midplanes = get_midplanes(image.detach().cpu())
    keys = ("xy", "xz", "yz")
    xy, xz, yz = [midplanes[x] for x in keys]

    # Scale features if required
    if type_scaling != None:
        xy = scale_features(image=xy, type_scaling=type_scaling)["image"]
        xz = scale_features(image=xz, type_scaling=type_scaling)["image"]
        yz = scale_features(image=yz, type_scaling=type_scaling)["image"]

    # Remove batch and channel dimensions
    xy, xz, yz = xy[0, 0], xz[0, 0], yz[0, 0]

    # Create figure
    if type_coordinates == "ras":
        xy, xz, yz = [xy.T, xz.T, yz.T]
    height_figure = tensor([x.shape[0] for x in (xy, xz, yz)]).max()
    width_figure = tensor([x.shape[1] for x in (xy, xz, yz)]).sum()
    shape_figure = tensor([width_figure, height_figure])
    shape_figure = shape_figure / shape_figure.max() * 15
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=shape_figure,
        gridspec_kw=dict(width_ratios=[x.shape[1] for x in (xy, xz, yz)]),
        subplot_kw=dict(anchor="NW"),
        tight_layout=True,
    )
    fig.suptitle(title)
    fig.tight_layout()

    # Plot midplanes
    if type_coordinates == "xyz":
        axs[0].imshow(xy, cmap="gray", origin="upper")
        axs[0].set(title="XY", xlabel="Y", ylabel="X")
        axs[0].set(xticks=[0, xy.shape[1] - 1], yticks=[0, xy.shape[0] - 1])
        axs[1].imshow(xz, cmap="gray", origin="upper")
        axs[1].set(title="XZ", xlabel="Z", ylabel="X")
        axs[1].set(xticks=[0, xz.shape[1] - 1], yticks=[0, xz.shape[0] - 1])
        axs[2].imshow(yz, cmap="gray", origin="upper")
        axs[2].set(title="YZ", xlabel="Z", ylabel="Y")
        axs[2].set(xticks=[0, yz.shape[1] - 1], yticks=[0, yz.shape[0] - 1])

    elif type_coordinates == "ras":
        axs[0].imshow(xy, cmap="gray", origin="lower")
        axs[0].set(xlim=axs[0].get_xlim()[::-1], xticks=[], yticks=[])
        axs[0].set_xlabel("P", ha="center")  # Bottom
        axs[0].set_ylabel("R", rotation=0, va="center")  # Left
        axx0 = axs[0].secondary_xaxis("top")
        axx0.set_xlabel(xlabel="A", ha="center")  # Top
        axx0.set_xticks([])
        axy0 = axs[0].secondary_yaxis("right")
        axy0.set_ylabel("L", rotation=0, va="center")  # Right
        axy0.set_yticks([])

        axs[1].imshow(xz, cmap="gray", origin="lower")
        axs[1].set(xlim=axs[1].get_xlim()[::-1], xticks=[], yticks=[])
        axs[1].set_xlabel("I", ha="center")  # Bottom
        axs[1].set_ylabel("R", rotation=0, va="center")  # Left
        axx1 = axs[1].secondary_xaxis("top")
        axx1.set_xlabel(xlabel="S", ha="center")  # Top
        axx1.set_xticks([])
        axy1 = axs[1].secondary_yaxis("right")
        axy1.set_ylabel("L", rotation=0, va="center")  # Right
        axy1.set_yticks([])

        axs[2].imshow(yz, cmap="gray", origin="lower")
        axs[2].set(xlim=axs[2].get_xlim()[::-1], xticks=[], yticks=[])
        axs[2].set_xlabel("I", ha="center")  # Bottom
        axs[2].set_ylabel("A", rotation=0, va="center")  # Left
        axx2 = axs[2].secondary_xaxis("top")
        axx2.set_xlabel(xlabel="S", ha="center")  # Top
        axx2.set_xticks([])
        axy2 = axs[2].secondary_yaxis("right")
        axy2.set_ylabel("P", rotation=0, va="center")  # Right
        axy2.set_yticks([])

    else:
        raise Exception(
            f"Coordinates type {type_coordinates} not implemented! Please contact the developers."
        )

    # Show if required
    if show:
        plt.show()
