import torch

def affine_transform_3d(rotation, translation, scaling):
    """Returns 3D affine transform matrix. 
    It is calculated in the following order: 
    M = T*R*S

    Parameters
    ----------
    rotation : torch.Tensor
        rotation angles for affine transform matrix, only Euler angles currently supported
    translation : torch.Tensor
        translation parameters for affine transform matrix
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
    elif rotation.dim() != 2 or rotation.shape[1]!=3:
        raise ValueError("rotation must be an Nx3 tensor")
    # Check translation parameter
    if not isinstance(translation, torch.Tensor):
        raise TypeError("translation must be a tensor")
    elif translation.dim() != 2 or translation.shape[1]!=3:
        raise ValueError("translation must be an Nx3 tensor")
    # Check scaling parameter
    if not isinstance(scaling, torch.Tensor):
        raise TypeError("scaling must be a tensor")
    elif scaling.dim() != 2 or scaling.shape[1]!=3:
        raise ValueError("scaling must be an Nx3 tensor")  

    # Calculate rotation transform
    R = rotation_transform_3d(rotation)

    # Calculate translation transform
    T =translation_transform_3d(translation)
    
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
    elif rotation.dim() != 2 or rotation.shape[1]!=3:
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
        translation parameters for affine transform matrix

    Returns
    -------
    torch.Tensor
        3D translation transform matrix
    """
    # Check translation parameter
    if not isinstance(translation, torch.Tensor):
        raise TypeError("translation must be a tensor")
    elif translation.dim() != 2 or translation.shape[1]!=3:
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
    elif scaling.dim() != 2 or scaling.shape[1]!=3:
        raise ValueError("scaling must be an Nx3 tensor")  

    # Calculate scaling transform
    S = torch.eye(4).reshape((1, 4, 4)).repeat(scaling.size()[0], 1, 1)
    S[:, 0, 0] = scaling[:, 0]
    S[:, 1, 1] = scaling[:, 1]
    S[:, 2, 2] = scaling[:, 2]

    return S
