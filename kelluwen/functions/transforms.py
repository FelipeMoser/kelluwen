from math import pi
from torch import (
    arange,
    meshgrid,
    ones_like,
    exp,
    eye,
    Tensor,
    cos,
    sin,
    matmul,
    stack,
    atan2,
    asin,
    ones,
    any,
    all,
    tensor,
    allclose,
    sqrt,
    zeros_like,
)


def apply_affine(
    image,
    transform_affine,
    shape_output=None,
    type_resampling="bilinear",
    type_origin="centre",
    type_output="dict",
):
    # Retrieve required variables
    if shape_output == None:
        shape_output = image.shape

    # Define supported resampling types
    supported_resampling = ("nearest", "bilinear")
    supported_origin = ("zero", "centre")
    supported_output = ("dict", "raw")

    # Check that the transform resampling is supported
    if type_resampling not in supported_resampling:
        raise ValueError(
            f"Unknown transform resampling type '{type_resampling}'. Supported types: {supported_resampling}"
        )

    # Check that transform origin is supported
    if type_origin not in supported_origin:
        raise ValueError(
            f"Unknown transform origin type '{type_origin}'. Supported types: {supported_origin}"
        )

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check input image
    if image.dim() not in (4, 5):
        raise ValueError(
            f"Input image must be a 4D (2D space) or 5D tensor (3D space), got {image.dim()} instead."
        )

    # Check affine transform
    if transform_affine.dim() != 3 and transform_affine.dim() != 4:
        raise ValueError(
            f"Affine transform must be a 3D or 4D tensor, got {transform_affine.dim()}D instead."
        )
    elif transform_affine.dim() == 3:
        transform_affine = transform_affine[:, None, :, :]
    elif transform_affine.shape[1] != image.shape[1] and transform_affine.shape[1] != 1:
        raise ValueError(
            f"Channel dimension of affine transform must match input image dimension, got {transform_affine.shape[1]} and {image.shape[1]}."
        )
    elif transform_affine.shape[2] != transform_affine.shape[3]:
        raise ValueError(
            f"Affine transform must be of shape BxCx3x3 (2D space) or BxCx4x4 (3D space), got {transform_affine.shape} instead"
        )
    elif transform_affine.shape[2] != image.dim() - 1:
        raise ValueError(
            f"Affine transform must be of shape BxCx{image.dim()- 1}x{image.dim()- 1} for a {image.dim()-2}D image, got {transform_affine.shape} instead"
        )

    # Check output shape
    if len(shape_output) != image.dim():
        raise ValueError(
            f"Output shape must have the same dimensionality as the input image, got {len(shape_output)} and {image.dim()} instead."
        )

    elif list(image.shape[:2]) != list(shape_output[:2]):
        raise ValueError(
            f"Batch and channel dimensions of the output shape must match that of the input image, got {shape_output[:2]} and {image.shape[:2]} instead."
        )

    # Cast affine to float and move it to device
    transform_affine = transform_affine.float().to(image.device)

    # Add transfrom origin translation if required
    if type_origin == "zero":
        pass
    elif type_origin == "centre":
        T_zero = (
            eye(image.dim() - 1)
            .tile(transform_affine.shape[0], transform_affine.shape[1], 1, 1)
            .to(image.device)
        )
        T_centre = T_zero.clone()
        T_zero[:, :, :-1, -1] = -(tensor(image.shape[2 : image.dim()]) - 1) / 2
        T_centre[:, :, :-1, -1] = (tensor(shape_output[2 : image.dim()]) - 1) / 2
        transform_affine = T_centre.matmul(transform_affine.matmul(T_zero))

    else:
        raise Exception(
            f"Transform origin '{type_resampling}' not implemented! Please contact the developers."
        )
    # Generate transformed coordinates
    transform_affine = transform_affine.inverse()[:, :, :-1, :]
    coordinates = meshgrid(*(arange(s) for s in shape_output[2:]), indexing="ij")
    coordinates = stack((*coordinates, ones(*shape_output[2:]))).to(image.device)
    coordinates = transform_affine.matmul(
        coordinates.reshape((1, 1, image.dim() - 1, -1))
    )

    # Prepare indices for readability
    batch = arange(shape_output[0])[:, None, None]
    channel = arange(shape_output[1])[None, :, None]
    x = coordinates[:, :, 0, :]
    y = coordinates[:, :, 1, :]
    if image.dim() == 5:
        z = coordinates[:, :, 2, :]

    # Find transformed coordinates outside image
    mask = ~(
        any(coordinates < 0, dim=2)
        | (x > image.shape[2] - 1)
        | (y > image.shape[3] - 1)
    )
    if image.dim() == 5:
        mask = mask & ~(z > image.shape[4] - 1)

    # Clip coordinates outside image
    coordinates *= mask[:, :, None, :]

    # Resample
    if type_resampling in ("nearest", "bilinear"):

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
            output = (
                image[batch, channel, c0(x), c0(y)] * (w0(x) * w0(y))
                + image[batch, channel, c1(x), c0(y)] * (w1(x) * w0(y))
                + image[batch, channel, c0(x), c1(y)] * (w0(x) * w1(y))
                + image[batch, channel, c1(x), c1(y)] * (w1(x) * w1(y))
            )
        else:
            output = (
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
        output *= mask

        # Reshape transformed image
        output = output.reshape(shape_output)

    else:
        raise Exception(
            f"Transform resampling type '{type_resampling}' not implemented! Please contact the developers."
        )
    # Return transformed image
    if type_output == "raw":
        return output
    else:
        return {"image": output}


def apply_anisotropic(
    image,
    iterations=1,
    delta_t=0.25,
    k=100,
    type_diffusion=0,
    type_output="dict",
):
    """
    Based on Perona et al., Scale-Space and Edge Detection Using Anisotropic Diffusion, IEEE T PATTERN ANAL, 1990
    """

    # Retrieve required variables
    type_output = type_output.lower()

    # Define supported rotation parameter types
    supported_diffusion = (0, 1)
    supported_output = ("dict", "raw")

    # Check that the diffusion type is supported
    if type_diffusion not in supported_diffusion:
        raise ValueError(
            f"Unknown diffusion type '{type_diffusion}'. Supported types: {supported_diffusion}"
        )

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

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
    for _ in range(iterations):
        flux = zeros_like(image)
        for idx_i, idx_n in zip(idx_image, idx_neighbours):
            # Calculate the flux contribution of the current indices
            temp = image[idx_n] - image[idx_i]
            if type_diffusion == 0:
                temp = exp(-((temp / k) ** 2)) * temp
            elif type_diffusion == 1:
                temp = temp / (1 + (temp / k) ** 2)

            # Add the flux contribution of the current indices, as well as the contributions of the antisymmetric indices
            flux[idx_i] += temp
            flux[idx_n] -= temp

        # Update image
        image += delta_t * flux

    # Return diffused image
    if type_output == "raw":
        return image
    else:
        return {"image": image}


def deconstruct_affine(
    transform_affine,
    transform_order="srt",
    type_rotation="euler_xyz",
    type_output="dict",
):
    # Retrieve required variables
    transform_order = transform_order.lower()
    type_output = type_output.lower()
    type_rotation = type_rotation.lower()

    # Define supported transform orders
    supported_order = ("trs", "tsr", "rts", "rst", "str", "srt")

    # Define supported rotation parameter types
    supported_rotation = ("euler_xyz", "quaternion")
    supported_output = ("dict", "raw")

    # Check that the transform order is supported
    if transform_order not in supported_order:
        raise ValueError(
            f"Unknown transform order '{transform_order}'. Supported transform orders: {supported_order}"
        )

    # Check that the rotation type is supported
    if type_rotation not in supported_rotation:
        raise ValueError(
            f"Unknown rotation type '{type_rotation}'. Supported types: {supported_rotation}"
        )

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check affine transform
    if transform_affine.dim() not in (3, 4):
        raise ValueError(
            f"Affine transform must be a 3D or 4D tensor, got {transform_affine.dim()}D instead."
        )
    elif (transform_affine.shape[-2] != transform_affine.shape[-1]) or (
        transform_affine.shape[-1] not in (3, 4)
    ):
        raise ValueError(
            f"Affine transform must be of shape BxCx3x3 (2D space) or BxCx4x4 (3D space), got {transform_affine.shape} instead"
        )
    elif transform_affine.dim() == 4:
        channel_dimension = True
    elif transform_affine.dim() == 3:
        channel_dimension = False
        transform_affine = transform_affine[:, None, :, :]

    # Extract scaling parameters
    if transform_order in ("srt, str, tsr"):
        parameter_scaling = transform_affine[:, :, :-1, :-1].norm(dim=3)
    else:
        parameter_scaling = transform_affine[:, :, :-1, :-1].norm(dim=2)

    # Extract scaling transform
    transform_scaling = generate_scaling(parameter_scaling, type_output="raw")

    # Extract rotation transform
    if transform_order in ("srt, str, tsr"):
        transform_rotation = matmul(transform_scaling.inverse(), transform_affine)
    else:
        transform_rotation = matmul(transform_affine, transform_scaling.inverse())
    transform_rotation[:, :, :-1, -1] = 0

    # Extract translation transform
    if transform_order == "trs":
        transform_translation = matmul(
            transform_affine,
            matmul(transform_scaling.inverse(), transform_rotation.inverse()),
        )
    elif transform_order == "tsr":
        transform_translation = matmul(
            transform_affine,
            matmul(transform_rotation.inverse(), transform_scaling.inverse()),
        )
    elif transform_order == "rts":
        transform_translation = matmul(
            transform_rotation.inverse(),
            matmul(transform_affine, transform_scaling.inverse()),
        )
    elif transform_order == "rst":
        transform_translation = matmul(
            transform_scaling.inverse(),
            matmul(transform_rotation.inverse(), transform_affine),
        )
    elif transform_order == "srt":
        transform_translation = matmul(
            transform_rotation.inverse(),
            matmul(transform_scaling.inverse(), transform_affine),
        )
    elif transform_order == "str":
        transform_translation = matmul(
            transform_scaling.inverse(),
            matmul(transform_affine, transform_rotation.inverse()),
        )

    # Extract translation parameters
    parameter_translation = transform_translation[:, :, :-1, -1]

    # Extract rotation parameters
    if transform_rotation.shape[2] == 2:  # 2D rotation
        parameter_rotation = asin(transform_rotation[:, :, 1, 0])
    else:  # 3D rotation
        if type_rotation == "quaternion":
            # Extract quaternions from rotation transform
            trace_transform_rotation = (
                transform_rotation[:, :, 0, 0]
                + transform_rotation[:, :, 1, 1]
                + transform_rotation[:, :, 2, 2]
            )
            q0 = sqrt(1 + trace_transform_rotation) / 2
            q1 = (transform_rotation[:, :, 2, 1] - transform_rotation[:, :, 1, 2]) / (
                2 * sqrt(1 + trace_transform_rotation)
            )
            q2 = (transform_rotation[:, :, 0, 2] - transform_rotation[:, :, 2, 0]) / (
                2 * sqrt(1 + trace_transform_rotation)
            )
            q3 = (transform_rotation[:, :, 1, 0] - transform_rotation[:, :, 0, 1]) / (
                2 * sqrt(1 + trace_transform_rotation)
            )
            parameter_rotation = stack([q0, q1, q2, q3], dim=2)

        elif type_rotation == "euler_xyz":
            # Extract xyz Euler angles from rotation transform
            alpha = atan2(
                -transform_rotation[:, :, 1, 2], transform_rotation[:, :, 2, 2]
            )
            beta = asin(transform_rotation[:, :, 0, 2])
            gamma = atan2(
                -transform_rotation[:, :, 0, 1], transform_rotation[:, :, 0, 0]
            )
            parameter_rotation = stack([alpha, beta, gamma], dim=2)

        else:
            raise Exception(
                f"Rotation type '{type_rotation}' not implemented! Please contact the developers."
            )

    # Return deconstruction parameters
    if channel_dimension == False:
        parameter_scaling = parameter_scaling[:, 0, :]
        parameter_translation = parameter_translation[:, 0, :]
        parameter_rotation = parameter_rotation[:, 0, :]
    return {
        "parameter_scaling": parameter_scaling,
        "parameter_translation": parameter_translation,
        "parameter_rotation": parameter_rotation,
    }


def generate_affine(
    parameter_translation,
    parameter_rotation,
    parameter_scaling,
    type_rotation="",
    transform_order="trs",
    type_output="dict",
):
    # Retrieve required variables
    type_output = type_output.lower()

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Generate required transforms
    transform_translation = generate_translation(
        parameter_translation,
    )["transform_translation"]
    transform_rotation = generate_rotation(
        parameter_rotation,
        type_rotation=type_rotation,
    )["transform_rotation"]
    transform_scaling = generate_scaling(
        parameter_scaling,
    )["transform_scaling"]

    # Define supported transform orders
    supported_order = ("trs", "tsr", "rts", "rst", "str", "srt")

    # Check that the transform order is supported
    if transform_order not in supported_order:
        raise ValueError(
            f"Unknown transform order '{transform_order}'. Supported transform orders: {supported_order}"
        )

    # Sort order of operations
    key = (transform_order.index(x) for x in ("t", "r", "s"))
    operations = (transform_translation, transform_rotation, transform_scaling)
    operations = [x for _, x in sorted(zip(key, operations))]

    # Generate affine transform
    transform_affine = matmul(operations[0], matmul(operations[1], operations[2]))

    # Return affine transform
    if type_output == "raw":
        return transform_affine
    else:
        return {"transform_affine": transform_affine}


def generate_kernel(**kwargs):
    # Retrieve required variables
    type_kernel = kwargs["type_kernel"].lower()
    shape_kernel = kwargs["shape_kernel"]
    sigma_kernel = kwargs["sigma_kernel"]

    # Define supported kernels
    supported_type = ("uniform", "gaussian")

    # Check that the kernel function is supported
    if type_kernel not in supported_type:
        raise ValueError(
            f"Unknown kernel type '{type_kernel}'. Supported types: {supported_type}"
        )

    # Check that kernel size fulfills requirements
    if isinstance(shape_kernel, int):
        shape_kernel = (shape_kernel,)
    elif not isinstance(shape_kernel, (tuple, list)) or not all(
        tensor([isinstance(x, int) for x in shape_kernel])
    ):
        raise ValueError(
            "Variable shape_kernel must be an integer or tuple/list of integers."
        )

    # Check that kernel sigma fulfills requirements
    if isinstance(sigma_kernel, (int, float)):
        sigma_kernel = (sigma_kernel,)
    elif not isinstance(sigma_kernel, (tuple, list)) or not all(
        tensor([isinstance(x, (int, float)) for x in sigma_kernel])
    ):
        raise ValueError(
            "Variable sigma_kernel must be a number or tuple/list of numbers."
        )
    elif len(shape_kernel) != len(sigma_kernel):
        raise ValueError(
            f"Variables shape_kernel and sigma_kernel must have the same number of parameters. Got {len(shape_kernel)} and {len(sigma_kernel)} instead."
        )

    # Generate kernel base
    coordinates = [arange(elem) - elem // 2 for elem in shape_kernel]
    grids = meshgrid(coordinates, indexing="xy")
    kernel = ones_like(grids[0]).float()

    # Generate kernel
    if type_kernel == "uniform":
        for g, s in zip(grids, sigma_kernel):
            d = sqrt(tensor([3])) * s  # Distance from centre to edge
            kernel *= (g.abs() <= d) / (2 * d)
        # Normalise kernel to compensate for small approximation errors
        kernel /= kernel.sum()
        return {"kernel": kernel}

    elif type_kernel == "gaussian":
        for g, s in zip(grids, sigma_kernel):
            kernel *= exp(-(g ** 2) / (2 * (s ** 2))) / (sqrt(tensor([2 * pi])) * s)
        # Normalise kernel to compensate for small approximation errors
        kernel /= kernel.sum()
        return {"kernel": kernel}


def generate_scaling(parameter_scaling, type_output="dict"):
    # Retrieve required variables
    type_output = type_output.lower()

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check scaling parameter
    if not isinstance(parameter_scaling, Tensor):
        raise TypeError("Scaling parameter must be a tensor")
    elif (parameter_scaling.dim() not in (2, 3)) or (
        parameter_scaling.shape[-1] not in (2, 3)
    ):
        raise ValueError(
            f"Scaling parameter must be a tensor of shape NxCx2 (2D) or NxCx3 (3D), got {parameter_scaling.shape} instead."
        )
    elif parameter_scaling.dim() == 2:
        channel_dimension = False
        parameter_scaling = parameter_scaling[:, None, :]
    else:
        channel_dimension = True

    # Generate scaling transform
    transform_scaling = eye(
        parameter_scaling.shape[2] + 1,
        device=parameter_scaling.device,
    )[None, None, :].type(parameter_scaling.type())
    transform_scaling = transform_scaling.tile(
        (parameter_scaling.shape[0], parameter_scaling.shape[1], 1, 1)
    )

    # Populate scaling transform
    transform_scaling[:, :, 0, 0] = parameter_scaling[:, :, 0]
    transform_scaling[:, :, 1, 1] = parameter_scaling[:, :, 1]
    if parameter_scaling.shape[2] == 3:
        transform_scaling[:, :, 2, 2] = parameter_scaling[:, :, 2]

    # Return scaling transform
    if channel_dimension == False:
        transform_scaling = transform_scaling[:, 0, :, :]
    if type_output == "raw":
        return transform_scaling
    else:
        return {"transform_scaling": transform_scaling}


def generate_rotation(parameter_rotation, type_rotation="", type_output="dict"):
    # Retrieve required variables
    type_output = type_output.lower()
    type_rotation = type_rotation.lower()
    if type_rotation == "":
        if parameter_rotation.shape[-1] in (1, 3):
            type_rotation = "euler_xyz"
        elif parameter_rotation.shape[-1] == 4:
            type_rotation = "quaternion"

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Define supported rotation parameter types
    supported_types = (
        "euler_xyz",
        "euler_xzy",
        "euler_yxz",
        "euler_yzx",
        "euler_zxy",
        "euler_zyx",
        "euler_xzx",
        "euler_xyx",
        "euler_yxy",
        "euler_yzy",
        "euler_zxz",
        "euler_zyz",
        "quaternion",
    )

    # Check that the rotation type is supported
    if type_rotation not in supported_types:
        raise ValueError(
            f"Unknown rotation type '{type_rotation}'. Supported types: {supported_types}"
        )

    # Check rotation parameter
    if not isinstance(parameter_rotation, Tensor):
        raise TypeError("Rotation parameter must be a tensor")
    elif (parameter_rotation.dim() not in (2, 3)) or (
        parameter_rotation.shape[-1] not in (1, 3, 4)
    ):
        raise ValueError(
            f"Rotation parameter must be a tensor of shape NxCx1 (2D), NxCx3 (3D Euler), or NxCx4 (3D quaternion), got {parameter_rotation.shape} instead."
        )
    elif parameter_rotation.dim() == 2:  # Add channel dimension if required
        channel_dimension = False
        parameter_rotation = parameter_rotation[:, None, :]
    else:
        channel_dimension = True

    # Generate identity transform
    identity = eye(
        4 - (parameter_rotation.shape[2] == 1), device=parameter_rotation.device
    )[None, None, :].type(parameter_rotation.type())
    identity = identity.tile(
        (parameter_rotation.shape[0], parameter_rotation.shape[1], 1, 1)
    )

    # Generate and populate rotation transform
    if parameter_rotation.shape[2] == 1:
        transform_rotation = identity
        transform_rotation[:, :, 0, 0] = cos(parameter_rotation[:, :, 0])
        transform_rotation[:, :, 0, 1] = -sin(parameter_rotation[:, :, 0])
        transform_rotation[:, :, 1, 0] = sin(parameter_rotation[:, :, 0])
        transform_rotation[:, :, 1, 1] = cos(parameter_rotation[:, :, 0])

    elif parameter_rotation.shape[2] == 3:

        # Check that type matches number of parameters
        if type_rotation[:5] != "euler":
            raise ValueError(
                f"Rotation type for rotation parameter of shape NxCx3 must be Euler, got {type_rotation} instead. "
            )

        index = {
            "x": tensor([[1, 1, 2, 2], [1, 2, 1, 2], [1, -1, 1, 1]]),
            "y": tensor([[0, 0, 2, 2], [0, 2, 0, 2], [1, 1, -1, 1]]),
            "z": tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, -1, 1, 1]]),
        }

        # Sort order of operations
        operations = [identity.clone(), identity.clone(), identity]
        for i in range(3):
            i0 = index[type_rotation[6 + i]][0]
            i1 = index[type_rotation[6 + i]][1]
            q0 = index[type_rotation[6 + i]][2].to(parameter_rotation.device)
            angle = parameter_rotation[:, :, i]
            operations[i][:, :, i0, i1] = q0 * stack(
                [cos(angle), sin(angle), sin(angle), cos(angle)], dim=2
            )

        # Generate rotation transform
        transform_rotation = matmul(operations[0], matmul(operations[1], operations[2]))

    else:
        # Check that type matches number of parameters
        if type_rotation != "quaternion":
            raise ValueError(
                f"Rotation type for rotation parameter of shape NxCx4 must be Quaternion, got {type_rotation} instead. "
            )

        # Check quaternions
        if not allclose(parameter_rotation.norm(dim=2), tensor([1.0])):
            raise ValueError(
                f"Rotation parameters for rotation type Quaternion must be a unit vector, got tensors with norms {parameter_rotation.norm(dim=2)} instead."
            )

        # Separate quaternion components for readability
        q0, q1, q2, q3 = parameter_rotation.permute(dims=(2, 0, 1))

        # Generate rotation transform
        transform_rotation = identity
        transform_rotation[:, :, 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
        transform_rotation[:, :, 0, 1] = 2 * (q1 * q2 - q3 * q0)
        transform_rotation[:, :, 0, 2] = 2 * (q1 * q3 + q2 * q0)
        transform_rotation[:, :, 1, 0] = 2 * (q1 * q2 + q3 * q0)
        transform_rotation[:, :, 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
        transform_rotation[:, :, 1, 2] = 2 * (q2 * q3 - q1 * q0)
        transform_rotation[:, :, 2, 0] = 2 * (q1 * q3 - q2 * q0)
        transform_rotation[:, :, 2, 1] = 2 * (q2 * q3 + q1 * q0)
        transform_rotation[:, :, 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)

    # Return rotation transform
    if channel_dimension == False:  # Remove channel dimension if required
        transform_rotation = transform_rotation[:, 0, :, :]
    if type_output == "raw":
        return transform_rotation
    else:
        return {"transform_rotation": transform_rotation}


def generate_translation(parameter_translation, type_output="dict"):
    # Retrieve required variables
    type_output = type_output.lower()

    # Define supported output types
    supported_output = ("dict", "raw")

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Check translation parameter
    if not isinstance(parameter_translation, Tensor):
        raise TypeError("Rotation parameter must be a tensor")
    elif (parameter_translation.dim() not in (2, 3)) or (
        parameter_translation.shape[-1] not in (2, 3)
    ):
        raise ValueError(
            f"Translation parameter must be a tensor of shape NxCx2 (2D) or NxCx3 (3D), got {parameter_translation.shape} instead."
        )
    elif parameter_translation.dim() == 2:
        channel_dimension = False
        parameter_translation = parameter_translation[:, None, :]
    else:
        channel_dimension = True

    # Generate translation transform
    transform_translation = eye(
        parameter_translation.shape[2] + 1,
        device=parameter_translation.device,
    )[None, None, :].type(parameter_translation.type())
    transform_translation = transform_translation.tile(
        (parameter_translation.shape[0], parameter_translation.shape[1], 1, 1)
    )

    # Populate translation transform
    transform_translation[:, :, :-1, -1] = parameter_translation[:, :]

    # Return translation transform
    if channel_dimension == False:
        transform_translation = transform_translation[:, 0, :, :]
    if type_output == "raw":
        return transform_translation
    else:
        return {"transform_translation": transform_translation}


def scale_features(image, type_scaling="min_max", type_output="dict"):
    # Retrieve required variables
    type_scaling = type_scaling.lower()
    type_output = type_output.lower()

    # Define supported output types
    supported_output = ("dict", "raw")

    # Define supported scaling types
    supported_scaling = ("min_max", "mean_norm", "z_score", "unit_length")

    # Check that the transform resampling is supported
    if type_scaling not in supported_scaling:
        raise ValueError(
            f"Unknown feature scaling type '{type_scaling}'. Supported types: {supported_scaling}"
        )

    # Check that output type is supported
    if type_output not in supported_output:
        raise ValueError(
            f"Unknown output type '{type_output}'. Supported types: {supported_output}"
        )

    # Scale tensor
    shape = (*image.shape[:2], *[1] * (image.dim() - 2))
    if type_scaling == "min_max":
        min_source = image.reshape((*image.shape[:2], -1)).min(dim=2)[0]
        min_source = min_source.reshape(shape)
        max_source = image.reshape((*image.shape[:2], -1)).max(dim=2)[0]
        max_source = max_source.reshape(shape)
        output = (image - min_source) / (max_source - min_source)

    elif type_scaling == "mean_norm":
        min_source = image.reshape((*image.shape[:2], -1)).min(dim=2)[0]
        min_source = min_source.reshape(shape)
        max_source = image.reshape((*image.shape[:2], -1)).max(dim=2)[0]
        max_source = max_source.reshape(shape)
        mean_source = image.reshape((*image.shape[:2], -1)).mean(dim=2)
        mean_source = mean_source.reshape(shape)
        output = (image - mean_source) / (max_source - min_source)

    elif type_scaling == "z_score":
        mean_source = image.reshape((*image.shape[:2], -1)).mean(dim=2)
        mean_source = mean_source.reshape(shape)
        std_source = image.reshape((*image.shape[:2], -1)).std(dim=2)
        std_source = std_source.reshape(shape)
        output = (image - mean_source) / std_source

    elif type_scaling == "unit_length":
        norm_source = image.reshape((*image.shape[:2], -1)).norm(dim=2)
        norm_source = norm_source.reshape(shape)
        output = image / norm_source

    # Return scaled tensor
    if type_output == "raw":
        return output
    else:
        return {"image": output}
