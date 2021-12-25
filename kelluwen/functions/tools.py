from torch import Tensor, full, tensor, minimum, maximum
import matplotlib.pyplot as plt
from .transforms import scale_features


def centre_crop(
    image, shape_output, value_pad=0, type_order="after", type_output="dict"
):
    # Define supported padding order. This order defines which side will be padded more in the case that asymmetric padding is needed.
    supported_order = ("before", "after")

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that the padding order is supported
    if type_order not in supported_order:
        raise ValueError(
            f"Unknown padding order type {type_order}. Supported types: {supported_order}"
        )

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check image tensor
    if not isinstance(image, Tensor):
        raise TypeError(f"Image must be a tensor, got {type(image)} instead.")

    # Check output shape
    if len(shape_output) != image.dim():
        raise ValueError(
            f"Output shape must have the same dimensionality as the input, got {len(shape_output)} and {image.dim()} instead."
        )
    elif list(image.shape[:2]) != list(shape_output[:2]):
        raise ValueError(
            f"Batch and channel dimensions of the output shape must match that of the input, got {shape_output[:2]} and {image.shape[:2]} instead."
        )

    # Convert shapes to tensors
    shape_source = tensor(image.shape)
    shape_output = tensor(shape_output)

    # Create padded tensor
    output = full(size=list(shape_output), fill_value=value_pad).type(image.type())

    # Calculate shape difference
    difference = (shape_output - shape_source) / 2
    # tensor([(y - x) / 2 for (x, y) in zip(shape_input, shape_output)])

    # Arrange difference based on padding order
    if type_order == "before":
        difference = difference.ceil().long()
    elif type_order == "after":
        difference = difference.floor().long()
    else:
        raise Exception(
            f"Padding order {type_order} not implemented! Please contact the developers."
        )

    # Generate corresponding indices for unpadded (source) and padded (output) tensors
    diff_source = minimum(difference, tensor([0])).abs()
    diff_output = maximum(difference, tensor([0]))
    idx_source = [
        slice(dx, minimum(xs + 1, dx + ys))
        for dx, ys, xs in zip(diff_source, shape_output, shape_source)
    ]
    idx_output = [
        slice(dy, minimum(ys + 1, dy + xs))
        for dy, ys, xs in zip(diff_output, shape_output, shape_source)
    ]

    # Populate padded tensor
    output[idx_output] = image[idx_source]

    # Return padded tensor
    if type_output == "raw":
        return output
    else:
        return {"image": output}


def get_midplanes(image, type_output="dict"):
    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check input image
    if not isinstance(image, Tensor):
        raise TypeError(f"Image must be a tensor, got {type(image)} instead.")
    elif image.dim() != 5:
        raise ValueError(
            f"Image must be a 5D tensor of shape BxCxDxHxW, got a {image.dim()}D tensor instead."
        )

    # Get volume shape
    idx = (tensor(image.shape) / 2).floor().long()

    # Get midplanes
    xy = image[:, :, :, :, idx[4]]
    xz = image[:, :, :, idx[3], :]
    yz = image[:, :, idx[2], :, :]

    # Return midplanes
    if type_output == "raw":
        return xy, xz, yz
    else:
        return {"xy": xy, "xz": xz, "yz": yz}


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
    midplanes = get_midplanes(image)
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
