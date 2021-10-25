import os
import torch
import torch.fft
import matplotlib.pyplot as plt
import math


def gaussian_kernel(size, sigma, dtype=torch.float):
    """Returns 1D, 2D, or 3D Gaussian kernel

    Parameters
    ----------
    size : int or list or tuple
        size of Gaussian kernel
    sigma : int or float or list or tuple
        standard deviation of Gaussian kernel
    dtype : torch.Tensor, optional
        kernel output type, by default torch.double

    Returns
    -------
    torch.Tensor
        1D, 2D, or 3D Gaussian kernel
    """
    # Check size parameter
    if not isinstance(size, (int, list, tuple)):
        raise TypeError("size must be an integer or a list (or tuple) of integers")
    if isinstance(size, int):
        size = [size]  # Convert size to list if it's an int
    if any((not isinstance(elem, int) for elem in size)):
        raise TypeError("size elements must be integers")
    elif len(size) not in [1, 2, 3]:
        raise ValueError(
            "size can have either 1 element (1D), two elements (2D) or three elements (3D); got {} elements instead".format(
                len(size)
            )
        )
    elif any((elem % 2 == 0 or elem <= 0 for elem in size)):
        raise ValueError("size elements must be odd and positive")

    # Check sigma parameter
    if not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError("sigma must be a number or a list (or tuple) of numbers")
    if isinstance(sigma, (int, float)):
        sigma = [sigma]  # Convert sigma to list if it's a number
    if any((not isinstance(elem, (int, float)) for elem in sigma)):
        raise TypeError("sigma elements must be numbers")
    elif len(sigma) not in [1, 2, 3]:
        raise ValueError(
            "sigma can have either 1 element (1D), two elements (2D) or three elements (3D); got {} elements instead".format(
                len(sigma)
            )
        )
    elif any((elem <= 0 for elem in sigma)):
        raise ValueError("sigma elements must be positive")

    # Check size and sigma are the same shape
    if len(size) != len(sigma):
        raise ValueError(
            "size and sigma must have the same number of elements; got size:{} and sigma:{} instead".format(
                len(size), len(sigma)
            )
        )

    # Generate kernel
    coordinates = [
        torch.arange(elem) - torch.div(elem, 2, rounding_mode="trunc") for elem in size
    ]
    grids = torch.meshgrid(coordinates)
    kernel = torch.exp(-sum([(g ** 2) / (2 * s ** 2) for g, s in zip(grids, sigma)]))
    return kernel.type(dtype)


class logger:
    def __init__(self, path, force_flush=True) -> None:
        """Logger object

        Parameters
        ----------
        path : str
            path to log file
        force_flush : bool, optional
            force clears the buffers after every log, by default True
        """
        # Check path parameter
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        # Check force_flush parameter
        if not isinstance(force_flush, bool):
            raise TypeError("force_flush must be a boolean")
        self.file = open(path, "w")
        self.force_flush = force_flush

    def log(self, text):
        """Logs text to log file

        Parameters
        ----------
        text : string
            string to be logged to log file
        """
        self.file.write(text)
        if self.force_flush:
            self.file.flush()

    def close(self):
        """Closes the log file
        """
        self.file.close()


def show_midplanes(input, title=None, norm=False, show=True):
    """Show the midplanes of a 3D volume

    Parameters
    ----------
    input : torch.Tensor
        tensor containing 3D volume, must have 3, 4, or 5 dimensions
    title : string, optional
        figure title
    normalise : bool
        determines wether the midplanes are normalised or not
    show : bool, optional
        determines wether the figure is shown or not

    """
    # Check input parameter
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif len(input.shape) < 3 or len(input.shape) > 5:
        raise ValueError("input must have 3, 4, or 5 dimensions")
    # Check title parameter
    if title is not None:
        if not isinstance(title, str):
            raise TypeError("title has to be a string")
    # Check normalise parameter
    if not isinstance(norm, bool):
        raise TypeError("normalise must be a boolean")
    # Check show parameter
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    # Remove non-spatial dimensions
    if len(input.shape) == 5:
        volume = input[0, 0, :, :, :].clone().detach()
    elif len(input.shape) == 4:
        volume = input[0, :, :, :].clone().detach()
    else:
        volume = input.clone().detach()

    # Get midplanes
    xy, xz, yz = get_midplanes(volume, norm=norm)
    # Create figures
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(xy)
    plt.title("XY")
    plt.subplot(1, 3, 2)
    plt.imshow(xz)
    plt.title("XZ")
    plt.subplot(1, 3, 3)
    plt.imshow(yz)
    plt.title("YZ")

    # Add title
    if title is not None:
        plt.suptitle(title)

    # Show if required
    if show:
        plt.show()


def get_midplanes(input, norm=False, pad=False):
    """Returns the midplanes of a 3D volume

    Parameters
    ----------
    input : torch.Tensor
        tensor containing 3D volume, must have 3, 4, or 5 dimensions
    normalise : bool
        determines wether the midplanes are normalised or not
    pad : bool
        determines wither the midplanes are centre-padded to the largest dimension or not
    
    Returns
    -------
    list
        midplanes of tensor
    """
    # Check input parameter
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif len(input.shape) < 3 or len(input.shape) > 5:
        raise ValueError("input must have 3, 4, or 5 dimensions")
    # Check normalise parameter
    if not isinstance(norm, bool):
        raise TypeError("normalise must be a boolean")

    # Check pad parameter
    if not isinstance(pad, bool):
        raise TypeError("pad must be a boolean")

    # Remove non-spatial dimensions
    if len(input.shape) == 5:
        volume = input[0, 0, :, :, :].clone().detach()
    elif len(input.shape) == 4:
        volume = input[0, :, :, :].clone().detach()
    else:
        volume = input.clone().detach()

    # Get volume shape
    shape = volume.shape

    # Get midplanes
    xy = volume[:, :, shape[2] // 2].clone()
    xz = volume[:, shape[1] // 2, :].clone()
    yz = volume[shape[0] // 2, :, :].clone()

    # Normalise if required
    if norm:
        xy = normalise(xy)
        xz = normalise(xz)
        yz = normalise(yz)

    # Centre-pad if required
    if pad:
        xy = centre_pad(xy, size=[max(shape)] * 2)
        xz = centre_pad(xz, size=[max(shape)] * 2)
        yz = centre_pad(yz, size=[max(shape)] * 2)

    return xy, xz, yz


def normalise(input):
    """Returns normalised tensor

    Parameters
    ----------
    input : torch.Tensor
        tensor to be normalised
        
    Returns
    -------
    torch.Tensor
        normalised tensor
    """
    input -= torch.min(input)
    input /= torch.max(input)
    return input


def bandpass_filter(input, filter=None, spacing=None, min_size=None, max_size=None):
    """Returns a bandpass-filtered tensor

    Parameters
    ----------
    input : torch.Tensor
        input tensor of shape (B, C, *) where * is two, or three spatial dimensions
    filter : torch.Tensor, optional
        frequency filter to be applied, by default None
    spacing : list
        spacing of the pixels/voxels of the data, optional
    min_size : float, optional
        smallest structural resultion to keep, must be in the same units as spacing, by default None
    max_size : float, optional
        largest structural resoltion to keep, must be in the same units as spacing, by default None

    Returns
    -------
    torch.Tensor
        filtered tensor
    """
    # Check input parameter
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif not 5 >= len(input.shape) >= 4:
        raise ValueError("input must have 4 dimensions (2D) or 5 dimensions (3D)")

    # Check filter parameter
    if filter is not None:
        if type(filter) != type(input):
            raise ValueError("filter must be the same type as input")
        elif filter.shape != input.shape:
            raise ValueError("filter must be the same shape as input")
    else:
        # Check spacing parameter
        if spacing is None:
            raise ValueError("spacing required if no filter is provided")
        elif not isinstance(spacing, list):
            raise TypeError("spacing must be a list")
        elif any((not isinstance(elem, (int, float)) for elem in spacing)):
            raise ValueError("spacing must be a list of numbers")
        elif not 3 >= len(spacing) >= 2:
            raise ValueError("spacing must have 2 (2D) or 3 (3D) values")
        # Check min_size parameter
        if min_size is not None:
            if not isinstance(min_size, (int, float)):
                raise TypeError("min_size must be a number")
            elif min_size <= 0:
                raise TypeError("min_size must be positive")
        # Check max_size parameter
        if max_size is not None:
            if not isinstance(max_size, (int, float)):
                raise TypeError("max_size must be a number")
            elif max_size <= 0:
                raise TypeError("max_size must be positive")

    # Get shape of input and spacing
    input_shape = torch.tensor(input.shape)
    spacing = torch.tensor(spacing)

    # Generate filter if required
    if filter is None:
        # Calculate coordinates and grid
        coordinates = [torch.arange(elem) - elem // 2 for elem in input_shape[2:]]
        grids = torch.meshgrid(coordinates)

        # Generate low-pass filter to remove structures smaller than min_size
        if min_size is None:
            filter_lowpass = torch.ones_like(input)
        else:
            # Convert min_size to cycle_lim_lowpass
            cycle_lim_lowpass = (input_shape[2:] * spacing) / min_size
            # Generate filter_lowpass
            filter_lowpass = (
                (sum([g ** 2 / (f ** 2) for g, f in zip(grids, cycle_lim_lowpass)]) < 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(input.device)
            )

        # Generate high-pass filter to remove structures larger than max_size
        if max_size is None:
            filter_highpass = torch.ones_like(input)
        else:
            # Convert min_size to cycle_lim_highpass
            cycle_lim_highpass = (input_shape[2:] * spacing) / max_size

            # Generate filter_highpass
            filter_highpass = (
                (
                    sum([g ** 2 / (f ** 2) for g, f in zip(grids, cycle_lim_highpass)])
                    > 1
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(input.device)
            )
        # Generate bandpass filter
        filter_bandpass = torch.logical_and(
            filter_highpass.bool(), filter_lowpass.bool()
        )
    else:
        filter_bandpass = filter
    # Transform input
    input_fft = torch.fft.fftshift(torch.fft.fftn(input))
    # Apply filter
    input_fft_filtered = input_fft * filter_bandpass
    # Transform back
    input_filtered = (torch.fft.ifftn(torch.fft.ifftshift(input_fft_filtered))).real

    # Return filtered image
    return input_filtered


def centre_crop(input, size):
    """Returns centre-cropped tensor

    Parameters
    ----------
    input : torch.Tensor
        tensor to be cropped
    size : int or list or tuple
        size of crop

    Returns
    -------
    torch.Tensor
        cropped tensor
    """
    # Check input parameter
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif not 5 >= len(input.shape) >= 3:
        raise ValueError(
            "input must have 3 dimensions (1D), 4 dimensions (2D), or 5 dimensions (3D)"
        )
    # Check size parameter
    if not isinstance(size, (list, tuple)):
        raise TypeError("size must be a list or tuple of integers")
    elif any((not isinstance(elem, (int)) for elem in size)):
        raise TypeError("size must be a list or tuple of integers")
    if len(size) != len(input.shape):
        raise ValueError(
            "size must have the same number of elements as dimensions in input"
        )
    if any([(x - y) < 0 for (x, y) in zip(input.shape, size)]):
        raise ValueError(
            "size must be smaller or equal to input shape in all dimensions"
        )

    # Calculate difference
    diff = [x - y for (x, y) in zip(input.shape, size)]
    crop_left = [math.floor(x / 2) for x in diff]
    crop_rigth = [math.ceil(x / 2) for x in diff]

    # Return centre-cropped input
    if len(input.shape) == 3:
        return input[
            crop_left[0] : -crop_rigth[0],
            crop_left[1] : -crop_rigth[1],
            crop_left[2] : -crop_rigth[2],
        ]
    elif len(input.shape) == 4:
        return input[
            crop_left[0] : -crop_rigth[0],
            crop_left[1] : -crop_rigth[1],
            crop_left[2] : -crop_rigth[2],
            crop_left[3] : -crop_rigth[3],
        ]
    else:
        return input[
            crop_left[0] : -crop_rigth[0],
            crop_left[1] : -crop_rigth[1],
            crop_left[2] : -crop_rigth[2],
            crop_left[3] : -crop_rigth[3],
            crop_left[4] : -crop_rigth[4],
        ]


def centre_pad(input, size, value=0):
    """Returns centre-padded tensor

    Parameters
    ----------
    input : torch.Tensor
        tensor to be padded
    size : int or list or tuple
        tensor size after padding
    value: number
        padding value
        

    Returns
    -------
    torch.Tensor
        padded tensor
    """
    # Check input parameter
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif not 5 >= len(input.shape) >= 1:
        raise ValueError("input must have between 1 and 5 dimensions")
    # Check size parameter
    if not isinstance(size, (list, tuple)):
        raise TypeError("size must be a list or tuple of integers")
    elif any((not isinstance(elem, (int)) for elem in size)):
        raise TypeError("size must be a list or tuple of integers")
    if len(size) != len(input.shape):
        raise ValueError(
            "size must have the same number of elements as dimensions in input"
        )
    if any([(x - y) > 0 for (x, y) in zip(input.shape, size)]):
        raise ValueError(
            "size must be larger or equal to input shape in all dimensions"
        )

    # Create empty padded tensor
    padded = torch.ones(size=size) * value

    # Calculate difference
    pad_before = [(y - x) // 2 for (x, y) in zip(input.shape, size)]

    # Pad input
    if len(input.shape) == 1:
        padded[pad_before[0] : pad_before[0] + input.shape[0],] = input
    elif len(input.shape) == 2:
        padded[
            pad_before[0] : pad_before[0] + input.shape[0],
            pad_before[1] : pad_before[1] + input.shape[1],
        ] = input
    elif len(input.shape) == 3:
        padded[
            pad_before[0] : pad_before[0] + input.shape[0],
            pad_before[1] : pad_before[1] + input.shape[1],
            pad_before[2] : pad_before[2] + input.shape[2],
        ] = input
    elif len(input.shape) == 4:
        padded[
            pad_before[0] : pad_before[0] + input.shape[0],
            pad_before[1] : pad_before[1] + input.shape[1],
            pad_before[2] : pad_before[2] + input.shape[2],
            pad_before[3] : pad_before[3] + input.shape[3],
        ] = input
    else:
        padded[
            pad_before[0] : pad_before[0] + input.shape[0],
            pad_before[1] : pad_before[1] + input.shape[1],
            pad_before[2] : pad_before[2] + input.shape[2],
            pad_before[3] : pad_before[3] + input.shape[3],
            pad_before[4] : pad_before[4] + input.shape[4],
        ] = input

    # Return centre-padded input
    return padded


if __name__ == "__main__":
    x = torch.rand(10, 3, 40, 50, 60)

    y = centre_pad(x, (10, 3, 100, 100, 100), value=1)
    show_midplanes(x, title="Org", show=False)
    show_midplanes(y, title="Pad")

