import torch
import torch.nn.functional
import numpy as np
import math


class AffineTransformer3d(torch.nn.Module):
    def __init__(self):
        """3D Affine transformer module
        """
        super().__init__()

    def forward(self, input, rotation, translation, scaling):
        """Returns input transformed with affine_transform_3d

        Parameters
        ----------
        input: torch.Tensor
            input tensor to be transformed with affine transform matrix
        rotation : torch.Tensor
            rotation angles for affine transform matrix, only Euler angles currently supported
        translation : torch.Tensor
            translation parameters for affine transform matrix, normalised to [-1, 1] where 1 is half the size of input, i.e. for a 160x160x160 input, [-1, 1] represents [-80, 80].
        scaling : torch.Tensor
            scaling parameters for affine transform matrix

        Returns
        -------
        torch.Tensor
            transformed input
        """
        # Check input
        if not isinstance(rotation, torch.Tensor):
            raise TypeError("input must be a tensor")
        elif input.dim() != 5:
            raise ValueError(
                "input must be an 5-dimensional tensor of shape NxCxDxHxW)"
            )
        # Check rotation parameter
        if not isinstance(rotation, torch.Tensor):
            raise TypeError("rotation must be a tensor")
        elif rotation.dim() != 2 or rotation.shape[1] != 3:
            raise ValueError("rotation must be an Nx3 tensor")
        # Check translation parameter
        if not isinstance(translation, torch.Tensor):
            raise TypeError("translation must be a tensor")
        elif translation.dim() != 2 or translation.shape[1] != 3:
            raise ValueError("translation must be an Nx3 tensor")
        # Check scaling parameter
        if not isinstance(scaling, torch.Tensor):
            raise TypeError("scaling must be a tensor")
        elif scaling.dim() != 2 or scaling.shape[1] != 3:
            raise ValueError("scaling must be an Nx3 tensor")

        # Generate affine transform matrix
        transform = affine_transform_3d(rotation, translation, scaling).to(input.device)

        # Generate sampling grid
        grid = torch.nn.functional.affine_grid(
            transform[:, :3, :], size=input.shape, align_corners=False,
        ).to(input.device)

        # Transform input
        output = torch.nn.functional.grid_sample(input, grid, align_corners=False)

        return output


def affine_transform_3d(rotation, translation, scaling):
    """Returns 3D affine transform matrix. 
    It is calculated in the following order: 
    M = T*R*S

    Parameters
    ----------
    rotation : torch.Tensor
        rotation angles for affine transform matrix, only Euler angles currently supported
    translation : torch.Tensor
        translation parameters for affine transform matrix, normalised to [-1, 1]
    scaling : torch.Tensor
        scaling parameters for affine transform matrix

    Returns
    -------
    torch.Tensor
        3D affine transform matrix
    """
    # Check rotation parameter
    if not isinstance(rotation, torch.Tensor):
        raise TypeError("rotation must be a tensor")
    elif rotation.dim() != 2 or rotation.shape[1] != 3:
        raise ValueError("rotation must be an Nx3 tensor")
    # Check translation parameter
    if not isinstance(translation, torch.Tensor):
        raise TypeError("translation must be a tensor")
    elif translation.dim() != 2 or translation.shape[1] != 3:
        raise ValueError("translation must be an Nx3 tensor")
    # Check scaling parameter
    if not isinstance(scaling, torch.Tensor):
        raise TypeError("scaling must be a tensor")
    elif scaling.dim() != 2 or scaling.shape[1] != 3:
        raise ValueError("scaling must be an Nx3 tensor")

    # Calculate rotation transform
    R = rotation_transform_3d(rotation)

    # Calculate translation transform
    T = translation_transform_3d(translation)

    # Calculate scaling transform
    S = scaling_transform_3d(scaling)

    # Calculate affine transform
    transform = torch.matmul(T, torch.matmul(R, S))
    return transform


def rotation_transform_3d(rotation):
    """Returns 3D rotation transform matrix. 
    It is calculated in the following order: 
    R = Rx*Ry*Rz

    Parameters
    ----------
    rotation : torch.Tensor
        rotation angles for affine transform, only Euler angles currently supported
    
    Returns
    -------
    torch.Tensor
        3D rotation transform matrix
    """
    # Check rotation parameter
    if not isinstance(rotation, torch.Tensor):
        raise TypeError("rotation must be a tensor")
    elif rotation.dim() != 2 or rotation.shape[1] != 3:
        raise ValueError("rotation must be an Nx3 tensor")

    # Create identiy matrix to generate rotation transforms
    Id = torch.eye(4).reshape((1, 4, 4)).repeat(rotation.size()[0], 1, 1)

    # Calculate rotation transform
    ea0 = rotation[:, 0]
    ea1 = rotation[:, 1]
    ea2 = rotation[:, 2]
    Rx = Id.clone()
    Rx[:, 0, 0] = torch.cos(ea0)
    Rx[:, 0, 1] = torch.sin(ea0)
    Rx[:, 1, 0] = -torch.sin(ea0)
    Rx[:, 1, 1] = torch.cos(ea0)
    Ry = Id.clone()
    Ry[:, 0, 0] = torch.cos(ea1)
    Ry[:, 0, 2] = -torch.sin(ea1)
    Ry[:, 2, 0] = torch.sin(ea1)
    Ry[:, 2, 2] = torch.cos(ea1)
    Rz = Id.clone()
    Rz[:, 1, 1] = torch.cos(ea2)
    Rz[:, 1, 2] = torch.sin(ea2)
    Rz[:, 2, 1] = -torch.sin(ea2)
    Rz[:, 2, 2] = torch.cos(ea2)
    R = torch.matmul(Rx, torch.matmul(Ry, Rz))

    return R


def translation_transform_3d(translation):
    """Returns 3D translation transform matrix. 

    Parameters
    ----------
    translation : torch.Tensor
        translation parameters for affine transform matrix, normalised to [-1, 1]

    Returns
    -------
    torch.Tensor
        3D translation transform matrix
    """
    # Check translation parameter
    if not isinstance(translation, torch.Tensor):
        raise TypeError("translation must be a tensor")
    elif translation.dim() != 2 or translation.shape[1] != 3:
        raise ValueError("translation must be an Nx3 tensor")

    # Calculate translation transform
    T = torch.eye(4).reshape((1, 4, 4)).repeat(translation.size()[0], 1, 1)
    T[:, 0, 3] = translation[:, 0]
    T[:, 1, 3] = translation[:, 1]
    T[:, 2, 3] = translation[:, 2]

    return T


def scaling_transform_3d(scaling):
    """Returns 3D scaling transform matrix. 

    Parameters
    ----------
    scaling : torch.Tensor
        scaling parameters for affine transform matrix

    Returns
    -------
    torch.Tensor
        3D scaling transform matrix
    """
    # Check scaling parameter
    if not isinstance(scaling, torch.Tensor):
        raise TypeError("scaling must be a tensor")
    elif scaling.dim() != 2 or scaling.shape[1] != 3:
        raise ValueError("scaling must be an Nx3 tensor")

    # Calculate scaling transform
    S = torch.eye(4).reshape((1, 4, 4)).repeat(scaling.size()[0], 1, 1)
    S[:, 0, 0] = scaling[:, 0]
    S[:, 1, 1] = scaling[:, 1]
    S[:, 2, 2] = scaling[:, 2]

    return S


def parameters_from_homogeneous_affine_3d(transform):
    """Returns rotation parameters, translation parameters, and scaling parameters from a homogeneous 3D affine transform matrix . Only Euler angles currently supported for the rotation parameters.

    Parameters
    ----------
    transform : torch.Tensor
        homogeneous 3D affine transform matrix

    Returns
    -------
    torch.Tensor
        3D rotation parameters (Euler angles)
    torch.Tensor
        3D translation parameters
    torch.Tensor
        3D scaling parameters
    """
    # Check transform parameter
    if not isinstance(transform, torch.Tensor):
        raise TypeError("transform must be a tensor")
    elif transform.dim() != 3 or transform.shape[1] != 4 or transform.shape[2] != 4:
        raise ValueError("transform must be of shape Nx4x4")

    # Calculate scaling parameters
    Sx = torch.linalg.norm(transform[:, :-1, 0], dim=1)
    Sy = torch.linalg.norm(transform[:, :-1, 1], dim=1)
    Sz = torch.linalg.norm(transform[:, :-1, 2], dim=1)
    scaling_params = torch.stack([Sx, Sy, Sz], dim=1)

    # Calculate translation parameters
    Tx = transform[:, 0, 3]
    Ty = transform[:, 1, 3]
    Tz = transform[:, 2, 3]
    translation_params = torch.stack([Tx, Ty, Tz], dim=1)

    # Calculate rotation parameters (Euler angles)
    Rx = torch.atan2(-transform[:, 1, 0], transform[:, 0, 0])
    Ry = torch.asin(transform[:, 2, 0] / Sx)
    Rz = torch.atan2(-transform[:, 2, 1] / Sy, transform[:, 2, 2] / Sz)
    rotation_params = torch.stack([Rx, Ry, Rz], dim=1)

    # Return parameters
    return rotation_params, translation_params, scaling_params


def anisotropic_diffusion_3d(
    input, iterations, delta_t=0.09, k=10, neighbours=26, mode=0
):
    """Returns input tensor after applying anisotropic diffusion filter
    This algorithm is based on Perona et al. 1990: Scale-Space and Edge Detection Using Anisotropic Diffusion


    Parameters
    ----------
    input : torch.Tensor
        input tensor to be filtered, of shape (B, C, D, H, W)
    iterations: int
        number of times the anisotropic diffusion filter is applied to the tensor
    delta_t : float
        diffusion filter integration constant, should be less than 1/7, 1/11, or 1/14 when using 8, 18, or 26 neighbours, respectively; by default 0.09
    k : float
        diffusion constant; by default 10
    neighbours : int
        nearest neighbours used to calculate diffusion, by default 26
    mode : int
        diffusion function used, 0 and 1 are the examples offered in Perona et al. 1990by default 0


    Returns
    -------
    torch.Tensor
        filtered tensor
    """
    # Check input
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    elif input.dim() != 5:
        raise ValueError("input must be a 5D tensor of shape (B, C, D, H, W)")
    # Check iterations
    if not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    elif iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    # Check delta_t
    if not isinstance(delta_t, (int, float)):
        raise TypeError("delta_t must be a number")
    elif delta_t <= 0:
        raise ValueError("delta_t must be a positive number")
    # Check k
    if not isinstance(k, (int, float)):
        raise TypeError("k must be a number")
    elif k <= 0:
        raise ValueError("k must be a positive number")
    # Check neighbours
    if not isinstance(neighbours, (int)):
        raise TypeError("neighbours must be an integer")
    elif neighbours not in [6, 18, 26]:
        raise ValueError("neighbours must be either 6, 18, or 26")
    # Check mode
    if not isinstance(mode, int):
        raise TypeError("mode must be an integer")
    elif mode not in [0, 1]:
        raise ValueError("mode must be either 0 or 1")

    # Define neighbour indices (index_i), their distances to node, and the node indices (index_0)
    index_i = np.asarray(
        [
            [[None, -1], [None, None], [None, None]],  # North
            [[None, None], [None, -1], [None, None]],  # West
            [[None, None], [None, None], [None, -1]],  # Up
            [[None, -1], [None, -1], [None, None]],  # North-west
            [[None, -1], [1, None], [None, None]],  # North-east
            [[None, -1], [None, None], [None, -1]],  # North-up
            [[None, -1], [None, None], [1, None]],  # North-down
            [[None, None], [None, -1], [None, -1]],  # West-up
            [[None, None], [None, -1], [1, None]],  # West-down
            [[None, -1], [None, -1], [None, -1]],  # North-west-up
            [[None, -1], [None, -1], [1, None]],  # North-west-down
            [[None, -1], [1, None], [None, -1]],  # North-east-up
            [[None, -1], [1, None], [1, None]],  # North-east-down
        ]
    )
    dist_i = np.asarray([1] * 3 + [math.sqrt(2)] * 6 + [math.sqrt(3)] * 4)
    index_0 = np.flip(index_i.copy(), axis=2)
    index_0[index_0 != None] *= -1

    # Remove unwanted neighbours
    if neighbours == 6:
        filt = dist_i == 1
    elif neighbours == 18:
        filt = (dist_i == 1) + (dist_i == math.sqrt(2))
    else:
        filt = (dist_i == 1) + (dist_i == math.sqrt(2)) + (dist_i == math.sqrt(3))
    index_i = index_i[filt]
    dist_i = dist_i[filt]
    index_0 = index_0[filt]

    # Begin filtering
    for _ in range(iterations):
        flux = torch.zeros_like(input)
        for idx_i, d_i, idx_0 in zip(index_i, dist_i, index_0):
            xi, yi, zi = idx_i
            x0, y0, z0 = idx_0

            
            # Calculate difference
            diff = (
                input[:, :, xi[0] : xi[1], yi[0] : yi[1], zi[0] : zi[1]]
                - input[:, :, x0[0] : x0[1], y0[0] : y0[1], z0[0] : z0[1]]
            )
            
            # Calculate the flow contribution
            if mode == 0:
                flow = delta_t * torch.exp(-((diff / k) ** 2)) * diff / d_i
            else:
                flow = delta_t * 1 / (1 + (diff / k) ** 2) * diff / d_i
            # Update image
            flux[:, :, x0[0] : x0[1], y0[0] : y0[1], z0[0] : z0[1]] += flow
            flux[:, :, xi[0] : xi[1], yi[0] : yi[1], zi[0] : zi[1]] -= flow
        input += flux
    return input
