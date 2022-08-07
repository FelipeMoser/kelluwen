import torch as tt
from typeguard import typechecked
from typing import Union, Dict, List, Tuple


@typechecked
def apply_affine(
    image: tt.Tensor,
    transform_affine: tt.Tensor,
    shape_output: Union[tt.Size, List[int]] = None,
    type_resampling: str = "bilinear",
    type_origin: str = "centre",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Applies affine transform to tensor.

    Parameters:
    ----------
    image : torch.Tensor
        Image being transformed. Must be of shape (batch, channel, *).

    transform_affine : torch.Tensor
        Affine transform being applied. Must be of shape (batch, channel, *), (batch, 1, *), or (batch, *).

    shape_output : torch.Size, optional (default=None)
        Output shape of transformed image. Must have the same batch and channel as image. If None, the output_shape=image.shape

    type_resampling: str, optional (default=bilinear)
        Interpolation algorithm used when sampling image. Available: bilinear, nearest

    type_origin: str, optional (default=centre)
        Point around which the transform is applied. Available: centre, origin

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    image_transformed : torch.Tensor
    """

    # Validate arguments
    if image.dim() not in (4, 5):
        raise ValueError(f"expected a 4D or 5D image, got {image.dim()!r}D instead")
    if transform_affine.dim() not in (3, 4):
        raise ValueError(
            f"expected a 3D or 4D transform_affine, got {transform_affine.dim()!r}D instead"
        )
    if transform_affine.shape[0] != image.shape[0]:
        raise ValueError(f"transform_affine.shape doesn't match image.shape")
    if transform_affine.dim() - 2 == image.dim() - 3:
        if transform_affine.shape[0] != image.shape[0]:
            raise ValueError(f"transform_affine.shape doesn't match image.shape")
    if transform_affine.shape[-2:] != (*[image.dim() - 1] * 2,):
        raise ValueError(f"transform_affine.shape doesn't match image.shape")
    if shape_output != None:
        if image.dim() != len(shape_output) or image.shape[:2] != shape_output[:2]:
            raise ValueError(f"shape_output doesn't match image.shape")
    if type_resampling.lower() not in ("nearest", "bilinear"):
        raise ValueError(f"unknown value {type_resampling!r} for type_resampling")
    if type_origin.lower() not in ("centre", "origin"):
        raise ValueError(f"unknown value {type_origin!r} for type_origin")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_resampling = type_resampling.lower()
    type_origin = type_origin.lower()
    type_output = type_output.lower()
    if shape_output == None:
        shape_output = image.shape
    if transform_affine.dim() == 3:
        transform_affine = transform_affine[:, None, :, :]
    if image.type() not in (tt.float, tt.double):
        image = image.float()
    if transform_affine.type() != (tt.float, tt.double):
        transform_affine = transform_affine.float()

    # Translate origin if required
    if type_origin == "centre":
        transform_origin = tt.eye(image.dim() - 1).to(transform_affine.device)
        transform_origin = transform_origin.tile(*transform_affine.shape[:2], 1, 1)
        transform_origin[..., :-1, -1] = (
            -(tt.tensor(image.shape[2 : image.dim()]) - 1) / 2
        )
        transform_affine = transform_origin.inverse().matmul(
            transform_affine.matmul(transform_origin)
        )

    # Generate transformed coordinates
    transform_affine = transform_affine.inverse()[..., :-1, :]
    coordinates = tt.meshgrid(*(tt.arange(s) for s in shape_output[2:]), indexing="ij")
    coordinates = tt.stack((*coordinates, tt.ones(*shape_output[2:]))).to(image.device)

    coordinates = transform_affine.matmul(
        coordinates.reshape((1, 1, image.dim() - 1, -1))
    )

    # Prepare indices for readability
    batch = tt.arange(shape_output[0])[:, None, None]
    channel = tt.arange(shape_output[1])[None, :, None]
    x = coordinates[..., 0, :]
    y = coordinates[..., 1, :]
    if image.dim() == 5:
        z = coordinates[:, :, 2, :]

    # Find transformed coordinates that lie outside image
    mask = ~(
        tt.any(coordinates < 0, dim=2)
        | (x > image.shape[2] - 1)
        | (y > image.shape[3] - 1)
    )
    if image.dim() == 5:
        mask = mask & ~(z > image.shape[4] - 1)

    # Clip coordinates outside image
    coordinates *= mask[:, :, None, :]

    # Resample
    if type_resampling == "nearest":
        # Prepare indices and weights for readability
        c0 = lambda x: (x.ceil() - 1).long()
        c1 = lambda x: x.ceil().long()
        w0 = lambda x: x.ceil() - x.round()
        w1 = lambda x: x.round() - (x.ceil() - 1)

    elif type_resampling == "bilinear":
        # Prepare indices and weights for readability
        c0 = lambda x: (x.ceil() - 1).long()
        c1 = lambda x: x.ceil().long()
        w0 = lambda x: x.ceil() - x
        w1 = lambda x: x - (x.ceil() - 1)

    # Sample transformed image
    if image.dim() == 4:
        image_transformed = (
            image[batch, channel, c0(x), c0(y)] * (w0(x) * w0(y))
            + image[batch, channel, c1(x), c0(y)] * (w1(x) * w0(y))
            + image[batch, channel, c0(x), c1(y)] * (w0(x) * w1(y))
            + image[batch, channel, c1(x), c1(y)] * (w1(x) * w1(y))
        )
    else:
        image_transformed = (
            image[batch, channel, c0(x), c0(y), c0(z)] * (w0(x) * w0(y) * w0(z))
            + image[batch, channel, c1(x), c0(y), c0(z)] * (w1(x) * w0(y) * w0(z))
            + image[batch, channel, c0(x), c1(y), c0(z)] * (w0(x) * w1(y) * w0(z))
            + image[batch, channel, c1(x), c1(y), c0(z)] * (w1(x) * w1(y) * w0(z))
            + image[batch, channel, c0(x), c0(y), c1(z)] * (w0(x) * w0(y) * w1(z))
            + image[batch, channel, c1(x), c0(y), c1(z)] * (w1(x) * w0(y) * w1(z))
            + image[batch, channel, c0(x), c1(y), c1(z)] * (w0(x) * w1(y) * w1(z))
            + image[batch, channel, c1(x), c1(y), c1(z)] * (w1(x) * w1(y) * w1(z))
        )

    # Mask transformed image
    image_transformed *= mask

    # Reshape transformed image
    image_transformed = image_transformed.reshape(shape_output)

    # Return results
    if type_output == "positional":
        return image_transformed
    else:
        return {"image": image_transformed}


@typechecked
def apply_anisotropic(
    image: tt.Tensor,
    size_step: float = 0.25,
    size_kappa: int = 100,
    type_diffusion: int = 0,
    size_iterations: int = 1,
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Applies an anisotropic diffusion filter on the input tensor, calculated using the method originally described in [1].

    Parameters
    ----------
    image : torch.Tensor
        Image being compared. Must be of shape (batch, channel, *). Currently only for 3D images.

    size_step : float, optional (default=0.25)
        Size of time step of each iteration. Must be <0.25 for stability.

    size_kappa : int, float, optional (default=100)
        Magnitude of edge strength threshold.

    type_diffusion : int, optional (default=0)
        Defines which diffusion function g to use, as proposed in [1]:
            0: g(∇I) = exp(-((||∇I|| / K) ** 2))
            1: g(∇I) = 1 / (1 + (||∇I|| / K) ** 2)

    size_iterations : int, optional (default=1)
        Number of times the filter is applied on the tensor.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------

    image_filtered : torch.Tensor

    References
    -----
    [1] Perona, P., & Malik, J. (1990). Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7), 629-639.
    """

    # Validate arguments
    if type_diffusion.lower() not in (0, 1):
        raise ValueError(f"unknown value {type_diffusion!r} for type_diffusion")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_output = type_output.lower()

    # Define index combinations
    idx_image = [
        [slice(None), slice(None), slice(1, None), slice(None), slice(None)],
        [slice(None), slice(None), slice(None), slice(1, None), slice(None)],
        [slice(None), slice(None), slice(None), slice(None), slice(1, None)],
    ]
    idx_neighbours = [
        [slice(None), slice(None), slice(None, -1), slice(None), slice(None)],
        [slice(None), slice(None), slice(None), slice(None, -1), slice(None)],
        [slice(None), slice(None), slice(None), slice(None), slice(None, -1)],
    ]

    # Apply anisotropic diffusion
    for _ in range(size_iterations):
        flux = tt.zeros_like(image)
        for idx_i, idx_n in zip(idx_image, idx_neighbours):
            # Calculate the flux contribution of the current indices
            delta_image = image[idx_n] - image[idx_i]
            if type_diffusion == 0:
                delta_image = tt.exp(-((delta_image / size_kappa) ** 2)) * delta_image
            elif type_diffusion == 1:
                delta_image = delta_image / (1 + (delta_image / size_kappa) ** 2)

            # Add the flux contribution of the current indices, as well as the contributions of the antisymmetric indices
            flux[idx_i] += delta_image
            flux[idx_n] -= delta_image

        # Update image
        image += size_step * flux

    # Return results
    if type_output == "positional":
        return image
    else:
        return {"image": image}


@typechecked
def deconstruct_affine(
    transform_affine: tt.Tensor,
    transform_order: str = "srt",
    type_rotation: str = "euler_xyz",
    type_output: str = "positional",
) -> Union[Tuple, Dict[str, tt.Tensor]]:
    """Deconstructs the affine transform into its conforming translation, rotation, and scaling parameters.

    Parameters
    ----------
    transform_affine : torch.Tensor
        Affine transform being deconstructed. Must be of shape (batch, channel, *) or (batch, *)

    transform_order : str, optional (default="srt)
        Order of multiplication of translation, rotation, and scaling transforms

    type_rotation: str, optional (default="euler_xyz)
        Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the multiplication of the rotations around x, y, and z is represented in the name (euler_xyz, euler_yzx, etc.)

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    parameter_translation : torch.Tensor

    parameter_rotation : torch.Tensor

    parameter_scaling : torch.Tensor

    """

    # Validate arguments
    if transform_affine.dim() not in (3, 4):
        raise ValueError(
            f"expected a 3D or 4D transform_affine, got {transform_affine.dim()!r}D instead"
        )
    if transform_affine.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError(
            f"unexpected shape of transform_affine {transform_affine.shape!r}"
        )
    if transform_order.lower() not in ("trs", "tsr", "rts", "rst", "str", "srt"):
        raise ValueError(f"unknown value {transform_order!r} for transform_order")
    if type_rotation.lower() not in (
        "euler_xyz",
        "euler_xzy",
        "euler_yxz",
        "euler_yzx",
        "euler_zxy",
        "euler_zyx",
        "quaternions",
    ):
        raise ValueError(f"unknown value {type_rotation!r} for type_rotation")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    transform_order = transform_order.lower()
    type_rotation = type_rotation.lower()
    type_output = type_output.lower()
    if transform_affine.dim() == 4:
        channel_dimension = True
    elif transform_affine.dim() == 3:
        channel_dimension = False
        transform_affine = transform_affine[:, None, ...]

    # Extract scaling parameters
    if transform_order in ("srt, str, tsr"):
        parameter_scaling = transform_affine[..., :-1, :-1].norm(dim=3)
    else:
        parameter_scaling = transform_affine[..., :-1, :-1].norm(dim=2)

    # Extract scaling transform
    transform_scaling = generate_scaling(parameter_scaling)

    # Extract rotation transform
    if transform_order in ("srt, str, tsr"):
        transform_rotation = transform_scaling.inverse().matmul(transform_affine)
    else:
        transform_rotation = transform_affine.matmul(transform_scaling.inverse())
    transform_rotation[..., :-1, -1] = 0

    # Extract translation transform
    if transform_order in ("trs", "tsr"):
        transform_translation = tt.eye(4).tile((*transform_affine.shape[:2], 1, 1))
        transform_translation[..., :-1, -1] = transform_affine[..., :-1, -1]
    elif transform_order == "str":
        transform_translation = tt.eye(4).tile((*transform_affine.shape[:2], 1, 1))
        transform_translation[..., :-1, -1] = transform_scaling.inverse().matmul(
            transform_affine
        )[..., :-1, -1]
    elif transform_order == "rts":
        transform_translation = (
            transform_rotation.inverse()
            .matmul(transform_affine)
            .matmul(transform_scaling.inverse())
        )
    elif transform_order == "rst":
        transform_translation = (
            transform_scaling.inverse()
            .matmul(transform_rotation.inverse())
            .matmul(transform_affine)
        )
    elif transform_order == "srt":
        transform_translation = (
            transform_rotation.inverse()
            .matmul(transform_scaling.inverse())
            .matmul(transform_affine)
        )

    # Extract translation parameters
    parameter_translation = transform_translation[..., :-1, -1]

    # Extract rotation parameters
    if transform_rotation.shape[2] == 2:  # 2D rotation
        parameter_rotation = tt.asin(transform_rotation[..., 1, 0])
    else:  # 3D rotation
        if type_rotation == "quaternions":
            # Extract quaternions from rotation transform
            # This section has been adapted to pytorch from nibabel's implementation: https://nipy.org/nibabel/reference/nibabel.quaternions.html
            transform_rotation = transform_rotation[..., :-1, :-1].flatten(start_dim=-2)
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = [
                transform_rotation[..., i] for i in range(transform_rotation.shape[-1])
            ]

            K = tt.eye(4).tile(*transform_rotation.shape[:-1], 1, 1)
            K[..., 0, 0] = Qxx - Qyy - Qzz
            K[..., 1, 0] = Qyx + Qxy
            K[..., 1, 1] = Qyy - Qxx - Qzz
            K[..., 2, 0] = Qzx + Qxz
            K[..., 2, 1] = Qzy + Qyz
            K[..., 2, 2] = Qzz - Qxx - Qyy
            K[..., 3, 0] = Qyz - Qzy
            K[..., 3, 1] = Qzx - Qxz
            K[..., 3, 2] = Qxy - Qyx
            K[..., 3, 3] = Qxx + Qyy + Qzz
            K /= 3
            vals, vecs = tt.linalg.eigh(K)
            q = vecs[..., [3, 0, 1, 2], :]
            parameter_rotation = tt.zeros(
                [*transform_rotation.shape[:2], 4], device=transform_affine.device
            )
            idx = tt.argmax(vals, dim=-1)
            for i in range(q.shape[0]):
                for j in range(q.shape[1]):
                    parameter_rotation[i, j] = q[i, j, :, idx[i, j]]
                    if parameter_rotation[i, j, 0] < 0:
                        parameter_rotation[i, j] *= -1
        else:
            # Get indices for the necessary transform components
            euler_idx = dict(
                euler_xyz=dict(
                    s=-1, alpha=[[1, 2], [2, 2]], beta=[0, 2], gamma=[[0, 1], [0, 0]]
                ),
                euler_xzy=dict(
                    s=1, alpha=[[2, 1], [1, 1]], beta=[0, 1], gamma=[[0, 2], [0, 0]]
                ),
                euler_yxz=dict(
                    s=1, alpha=[[0, 2], [2, 2]], beta=[1, 2], gamma=[[1, 0], [1, 1]]
                ),
                euler_yzx=dict(
                    s=-1, alpha=[[2, 0], [0, 0]], beta=[1, 0], gamma=[[1, 2], [1, 1]]
                ),
                euler_zxy=dict(
                    s=-1, alpha=[[0, 1], [1, 1]], beta=[2, 1], gamma=[[2, 0], [2, 2]]
                ),
                euler_zyx=dict(
                    s=1, alpha=[[1, 0], [0, 0]], beta=[2, 0], gamma=[[2, 1], [2, 2]]
                ),
            )
            idx_s = euler_idx[type_rotation]["s"]
            idx_alpha = euler_idx[type_rotation]["alpha"]
            idx_beta = euler_idx[type_rotation]["beta"]
            idx_gamma = euler_idx[type_rotation]["gamma"]

            # Get Euler angles
            alpha = tt.atan2(
                idx_s * transform_rotation[..., idx_alpha[0][0], idx_alpha[0][1]],
                transform_rotation[..., idx_alpha[1][0], idx_alpha[1][1]],
            )
            beta = -idx_s * tt.asin(transform_rotation[..., idx_beta[0], idx_beta[1]])
            gamma = tt.atan2(
                idx_s * transform_rotation[..., idx_gamma[0][0], idx_gamma[0][1]],
                transform_rotation[..., idx_gamma[1][0], idx_gamma[1][1]],
            )
            parameter_rotation = tt.stack([alpha, beta, gamma], dim=2)

    # Remove channels if required
    if channel_dimension == False:
        parameter_scaling = parameter_scaling[:, 0, :]
        parameter_translation = parameter_translation[:, 0, :]
        parameter_rotation = parameter_rotation[:, 0, :]

    # Return results
    if type_output == "positional":
        return parameter_translation, parameter_rotation, parameter_scaling
    else:
        return {
            "parameter_translation": parameter_translation,
            "parameter_rotation": parameter_rotation,
            "parameter_scaling": parameter_scaling,
        }


@typechecked
def generate_affine(
    parameter_translation: tt.Tensor,
    parameter_rotation: tt.Tensor,
    parameter_scaling: tt.Tensor,
    type_rotation: str = "euler_xyz",
    transform_order: str = "trs",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Generates an affine transform from translation, rotation, and scaling parameters.

    Parameters
    ----------
    parameter_translation : torch.Tensor
        Translation parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.

    parameter_rotation : torch.Tensor
        Rotation parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=1, 3 or 4, for 2D, 3D Euler angles, and 3D quaternions, respectively.

    parameter_scaling : torch.Tensor
        Scaling parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.

    type_rotation: str, optional (default="euler_xyz)
        Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the multiplication of the rotations around x, y, and z is represented in the name (euler_xyz, euler_yzx, etc.)

    transform_order : str, optional (default="srt)
        Order of multiplication of translation, rotation, and scaling transforms

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    transform_affine: torch.Tensor
    """

    # Validate arguments
    if (
        parameter_translation.shape[:-1] != parameter_rotation.shape[:-1]
        or parameter_translation.shape[:-1] != parameter_scaling.shape[:-1]
    ):
        raise ValueError(f"mismatched shape of parameters")
    if (
        parameter_translation.device != parameter_rotation.device
        or parameter_translation.device != parameter_scaling.device
    ):
        raise ValueError(f"mismatched devices of parameters")
    if transform_order.lower() not in ("trs", "tsr", "rts", "rst", "str", "srt"):
        raise ValueError(f"unknown value {transform_order!r} for transform_order")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    transform_order = transform_order.lower()
    type_output = type_output.lower()

    # Generate required transforms
    transform_translation = generate_translation(parameter_translation)
    transform_rotation = generate_rotation(parameter_rotation, type_rotation)
    transform_scaling = generate_scaling(parameter_scaling)

    # Sort order of operations
    key = (transform_order.index(x) for x in ("t", "r", "s"))
    operations = (transform_translation, transform_rotation, transform_scaling)
    operations = [x for _, x in sorted(zip(key, operations))]

    # Generate affine transform
    transform_affine = operations[0].matmul(operations[1]).matmul(operations[2])

    # Return results
    if type_output == "positional":
        return transform_affine
    else:
        return {"transform_affine": transform_affine}


@typechecked
def generate_kernel(
    type_kernel: str,
    shape_kernel: List[int],
    sigma_kernel: List[Union[float, int]],
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Generates an kernel.

    Parameters
    ----------
    type_kernel : str
        Probability distribution function used to generated the kernel: uniform, gaussian

    shape_kernel : int
        Shape of kernel. Must be an integer or tuple/list of integers.

    sigma_kernel : float
        Standard deviation of the probability distribution function used to generated the kernel. Must be an integer or tuple/list of integers or floats.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    kernel: torch.Tensor
    """

    # Validate arguments
    if len(shape_kernel) != len(sigma_kernel):
        raise ValueError(
            f"mismatched number of elements in shape_kernel and sigma_kernel"
        )
    if type_kernel.lower() not in ("uniform", "gaussian"):
        raise ValueError(f"unknown value {type_kernel!r} for type_kernel")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_kernel = type_kernel.lower()
    type_output = type_output.lower()

    # Generate kernel base
    coordinates = [tt.arange(elem) - elem // 2 for elem in shape_kernel]
    grids = tt.meshgrid(coordinates, indexing="ij")
    kernel = tt.ones_like(grids[0]).float()

    # Generate kernel
    if type_kernel == "uniform":
        for g, s in zip(grids, sigma_kernel):
            d = tt.sqrt(tt.tensor([3])) * s  # Distance from centre to edge
            kernel *= (g.abs() <= d) / (2 * d)
        # Normalise kernel to compensate for small approximation errors
        kernel /= kernel.sum()

    elif type_kernel == "gaussian":
        for g, s in zip(grids, sigma_kernel):
            kernel *= tt.exp(-(g**2) / (2 * (s**2))) / (
                tt.sqrt(tt.tensor([2 * tt.pi])) * s
            )
        # Normalise kernel to compensate for small approximation errors
        kernel /= kernel.sum()

    # Return results
    if type_output == "positional":
        return kernel
    else:
        return {"kernel": kernel}


@typechecked
def generate_scaling(
    parameter_scaling: tt.Tensor,
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Generates a scaling transform from scaling parameters.

    Parameters
    ----------
    parameter_scaling : torch.Tensor
        Scaling parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    transform_scaling : torch.Tensor
    """
    # Validate arguments
    if parameter_scaling.dim() not in (1, 2, 3):
        raise ValueError(
            f"expected a 1D, 2D, or 3D parameter_scaling, got {parameter_scaling.dim()!r}D instead"
        )
    if parameter_scaling.shape[-1] not in (2, 3):
        raise ValueError(f"unexpected shape of parameter_scaling")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_output = type_output.lower()
    device = parameter_scaling.device

    # Generate scaling transform
    transform_tiling = (*parameter_scaling.shape[:-1], 1, 1)
    transform_scaling = tt.eye(parameter_scaling.shape[-1] + 1, device=device)
    transform_scaling = transform_scaling.tile(transform_tiling)

    # Populate scaling transform
    for i in range(parameter_scaling.shape[-1]):
        transform_scaling[..., i, i] = parameter_scaling[..., i]

    # Return results
    if type_output == "positional":
        return transform_scaling
    else:
        return {"transform_scaling": transform_scaling}


@typechecked
def generate_rotation(
    parameter_rotation: tt.Tensor,
    type_rotation: str = "euler_xyz",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Generates a rotation transform from rotation parameters.

    Parameters
    ----------
    parameter_rotation : torch.Tensor
        Rotation parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=1, 3 or 4, for 2D, 3D Euler angles, and 3D quaternions, respectively.

    type_rotation: str, optional (default="euler_xyz)
        Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the multiplication of the rotations around x, y, and z is represented in the name (euler_xyz, euler_yzx, etc.) This variable with be ignored for 2D rotations.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    transform_rotation : torch.Tensor
    """
    # Validate arguments
    if parameter_rotation.dim() not in (1, 2, 3):
        raise ValueError(
            f"expected a 1D, 2D, or 3D parameter_rotation, got {parameter_rotation.dim()!r}D instead"
        )
    if parameter_rotation.shape[-1] not in (1, 3, 4):
        raise ValueError(f"unexpected shape of parameter_scaling")
    if type_rotation.lower() not in (
        "euler_xyz",
        "euler_xzy",
        "euler_yxz",
        "euler_yzx",
        "euler_zxy",
        "euler_zyx",
        "quaternions",
    ):
        raise ValueError(f"unknown value {type_rotation!r} for type_rotation")
    if parameter_rotation.shape[-1] != 1:
        if type_rotation.lower()[:5] == "euler" and parameter_rotation.shape[-1] != 3:
            raise ValueError(
                f"mismatch between type_rotation and shape of parameter_rotation"
            )
        if type_rotation.lower() == "quaternions" and parameter_rotation.shape[-1] != 4:
            raise ValueError(
                f"mismatch between type_rotation and shape of parameter_rotation"
            )
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_rotation = type_rotation.lower()
    type_output = type_output.lower()
    device = parameter_rotation.device

    # Generate identity transform
    transform_tiling = (*parameter_rotation.shape[:-1], 1, 1)
    transform_identity = tt.eye(4 - (parameter_rotation.shape[-1] == 1), device=device)
    transform_identity = transform_identity.tile(transform_tiling)

    # Generate and populate 2D rotation transform
    if parameter_rotation.shape[-1] == 1:
        transform_rotation = transform_identity
        transform_rotation[..., 0, 0] = tt.cos(parameter_rotation[..., 0])
        transform_rotation[..., 0, 1] = -tt.sin(parameter_rotation[..., 0])
        transform_rotation[..., 1, 0] = tt.sin(parameter_rotation[..., 0])
        transform_rotation[..., 1, 1] = tt.cos(parameter_rotation[..., 0])

    # Generate and populate 3D Euler rotation transform
    elif parameter_rotation.shape[-1] == 3:
        # Define rotation transform indices for rotation around x, y, and z
        index = {
            "x": tt.tensor([[1, 1, 2, 2], [1, 2, 1, 2], [1, -1, 1, 1]]),
            "y": tt.tensor([[0, 0, 2, 2], [0, 2, 0, 2], [1, 1, -1, 1]]),
            "z": tt.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, -1, 1, 1]]),
        }

        # Populate rotation transforms
        transform_rotation = transform_identity.clone()
        for i in range(3):
            i0 = index[type_rotation[6 + i]][0]
            i1 = index[type_rotation[6 + i]][1]
            q0 = index[type_rotation[6 + i]][2].to(device)
            angle = parameter_rotation[..., i]
            transform_temp = transform_identity.clone()
            transform_temp[..., i0, i1] = q0 * tt.stack(
                [tt.cos(angle), tt.sin(angle), tt.sin(angle), tt.cos(angle)], dim=-1
            )
            transform_rotation = transform_rotation.matmul(transform_temp)

    # Generate and populate 3D Quaternion rotation transform
    else:
        # Check if quaternions are normalised
        if tt.any(parameter_rotation.norm(dim=-1) < 1e-5):
            raise ValueError(
                f"parameter_rotation of type Quaternion must be normalised. Got parameter_rotation.norm(dim=-1)={parameter_rotation.norm(dim=-1)} instead",
            )

        # Separate quaternion components for readability
        if parameter_rotation.dim() == 3:
            q0, q1, q2, q3 = parameter_rotation.permute(dims=(2, 0, 1))
        else:
            q0, q1, q2, q3 = parameter_rotation.permute(dims=(1, 0))

        # Generate rotation transform
        transform_rotation = transform_identity
        transform_rotation[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
        transform_rotation[..., 0, 1] = 2 * (q1 * q2 - q3 * q0)
        transform_rotation[..., 0, 2] = 2 * (q1 * q3 + q2 * q0)
        transform_rotation[..., 1, 0] = 2 * (q1 * q2 + q3 * q0)
        transform_rotation[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
        transform_rotation[..., 1, 2] = 2 * (q2 * q3 - q1 * q0)
        transform_rotation[..., 2, 0] = 2 * (q1 * q3 - q2 * q0)
        transform_rotation[..., 2, 1] = 2 * (q2 * q3 + q1 * q0)
        transform_rotation[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)

    # Return results
    if type_output == "positional":
        return transform_rotation
    else:
        return {"transform_rotation": transform_rotation}


@typechecked
def generate_translation(
    parameter_translation: tt.Tensor,
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Generates a translation transform from translation parameters.

    Parameters
    ----------
    parameter_translation : torch.Tensor
        Translation parameters. Must be of shape (batch, channel, parameters) or (batch, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    transform_translation : torch.Tensor
    """

    # Validate arguments
    if parameter_translation.dim() not in (1, 2, 3):
        raise ValueError(
            f"expected a 1D, 2D, or 3D parameter_translation, got {parameter_translation.dim()!r}D instead"
        )
    if parameter_translation.shape[-1] not in (2, 3):
        raise ValueError(f"unexpected shape of parameter_translation")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_output = type_output.lower()
    device = parameter_translation.device

    # Generate scaling transform
    transform_tiling = (*parameter_translation.shape[:-1], 1, 1)
    transform_translation = tt.eye(parameter_translation.shape[-1] + 1, device=device)
    transform_translation = transform_translation.tile(transform_tiling)

    # Populate translation transform
    transform_translation[..., :-1, -1] = parameter_translation

    # Return results
    if type_output == "positional":
        return transform_translation
    else:
        return {"transform_translation": transform_translation}


@typechecked
def scale_features(
    image: tt.Tensor,
    type_scaling: str = "min_max",
    type_output: str = "positional",
) -> Union[tt.Tensor, Dict[str, tt.Tensor]]:
    """Scales the features of the tensor.

    Parameters
    ----------
    image : torch.Tensor
        Image to scale. Must be of shape (batch, channel, *).

    type_scaling : str, optional (default="min_max")
        Type of feature scaling used: min_max, mean_norm, z_score, unit_length

    type_output : str, optional (default="positional")
        Determines how the outputs are returned. If set to "positional", it returns positional outputs. If set to "named", it returns a dictionary with named outputs.

    Returns
    -------
    image_scaled : torch.Tensor
    """

    # Validate arguments
    if type_scaling.lower() not in ("min_max", "mean_norm", "z_score", "unit_length"):
        raise ValueError(f"unknown value {type_scaling!r} for type_scaling")
    if type_output.lower() not in ("positional", "named"):
        raise ValueError(f"unknown value {type_output!r} for type_output")

    # Update variables if required
    type_scaling = type_scaling.lower()
    type_output = type_output.lower()

    # Scale tensor
    shape_image = (*image.shape[:2], *[1] * (image.dim() - 2))
    if type_scaling == "min_max":
        min_source = image.reshape((*image.shape[:2], -1)).min(dim=2)[0]
        min_source = min_source.reshape(shape_image)
        max_source = image.reshape((*image.shape[:2], -1)).max(dim=2)[0]
        max_source = max_source.reshape(shape_image)
        image_scaled = (image - min_source) / (max_source - min_source)

    elif type_scaling == "mean_norm":
        min_source = image.reshape((*image.shape[:2], -1)).min(dim=2)[0]
        min_source = min_source.reshape(shape_image)
        max_source = image.reshape((*image.shape[:2], -1)).max(dim=2)[0]
        max_source = max_source.reshape(shape_image)
        mean_source = image.reshape((*image.shape[:2], -1)).mean(dim=2)
        mean_source = mean_source.reshape(shape_image)
        image_scaled = (image - mean_source) / (max_source - min_source)

    elif type_scaling == "z_score":
        mean_source = image.reshape((*image.shape[:2], -1)).mean(dim=2)
        mean_source = mean_source.reshape(shape_image)
        std_source = image.reshape((*image.shape[:2], -1)).std(dim=2)
        std_source = std_source.reshape(shape_image)
        image_scaled = (image - mean_source) / std_source

    elif type_scaling == "unit_length":
        norm_source = image.reshape((*image.shape[:2], -1)).norm(dim=2)
        norm_source = norm_source.reshape(shape_image)
        image_scaled = image / norm_source

    # Return results
    if type_output == "positional":
        return image_scaled
    else:
        return {"image": image_scaled}
