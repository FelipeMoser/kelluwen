import nibabel as nib
from pathlib import Path
from torch import from_numpy, permute, eye, tensor, diag, flip
from .transforms import apply_affine


def read_data(
    path_data,
    transform_data=False,
    shape_data=None,
    spacing_data=[1, 1, 1],
):
    # Retrieve required variables
    path_data = Path(path_data)
    path_data_name = path_data.name
    path_data_ext = "".join(path_data.suffixes)

    # Define supported output types
    supported_output = ("dict", "raw")

    # Define supported extensions
    supported_ext = (".nii", ".nii.gz")

    # Check that path exists
    if not path_data.is_file():
        raise ValueError("File {} not found".format(path_data))

    # Check that extension is supported
    if path_data_ext not in supported_ext:
        raise ValueError(
            "Cant read file '{}'. Supported extensions: {}".format(
                path_data_name, supported_ext
            )
        )

    # Call required read function
    if path_data_ext == ".nii" or ".nii.gz":
        nii_data = nib.load(str(path_data))
        nii_image = from_numpy(nii_data.get_fdata())
        nii_header = nii_data.header
        nii_affine = from_numpy(nii_data.affine)

        # Permute or add channel dimension of image
        if nii_image.dim() == 3:
            nii_image = nii_image[None, :]
        elif nii_image.dim() == 4:
            nii_image = permute(nii_image, (3, 0, 1, 2))
        else:
            raise ValueError(
                f"Expected nifti image to be 3D or 4D, got {nii_image.dim()}D instead."
            )

        # Add channel dimension of  affine
        nii_affine = nii_affine[None, :]

        # Apply transform if required
        if transform_data:
            # Add batch dimesions
            nii_image = nii_image[None, :]
            nii_affine = nii_affine[None, :]

            # Prepare shape data
            if shape_data == None:
                shape_data = list(nii_image.shape)
            elif len(shape_data) == 3:
                shape_data = [nii_image.shape[0], nii_image.shape[1]] + shape_data

            # Generate transform to match centre of RAS coordinates with centre of image
            t_centre = eye(4)[None, None, :].type(nii_affine.type())
            t_centre[:, :, :3, 3] = (tensor(shape_data[2:]) - 1) / 2

            # Generate transform to define voxel spacing of image
            t_voxel = diag(
                tensor(
                    [1 / spacing_data[0], 1 / spacing_data[1], 1 / spacing_data[1], 1]
                )
            )[None, None, :].type(nii_affine.type())

            # Generate required affine transform
            t_affine = t_centre.matmul(t_voxel.matmul(nii_affine))

            # Apply transform
            nii_image = apply_affine(
                image=nii_image,
                transform_affine=t_affine,
                shape_output=shape_data,
                type_origin="zero",
            )["image"]

            # Remove batch dimension
            nii_image = nii_image[0]
            nii_affine = nii_affine[0]
            t_affine = t_affine[0]

        # Return data
        return {
            "image": nii_image,
            "header": nii_header,
            "transform_affine": nii_affine,
        }

    else:
        raise Exception("Format '{}' not implemented! Please contact the developers.")


def write_data(image, path_data, transform_affine=None):
    # Retrieve required variables
    path_data = Path(path_data)
    path_data_ext = "".join(path_data.suffixes)
    # image = kwargs["image"]
    # transform_affine = kwargs.get("transform_affine")

    # Define supported extensions
    supported_ext = (".nii", ".nii.gz")

    # Check that folder path exists
    if not path_data.parent.is_dir():
        raise ValueError(f"Folder {path_data.parent} not found")

    # Check that extension is supported
    if path_data_ext not in supported_ext:
        raise ValueError(
            f"Unsupported extension for file {path_data}. Supported extensions: {supported_ext}"
        )

    # Check that the image dimensions are compatible
    if image.dim() not in (3, 4):
        raise ValueError(f"Image must be a 3D or 4D tensor, got {image.dim()} instead.")

    # Check that the affine transform dimensions are compatible
    if transform_affine != None and transform_affine.dim() != 2:
        raise ValueError(
            f"Affine transform must be a 2D tensor, got {transform_affine.dim()} instead."
        )

    # Call required write function
    if path_data_ext == ".nii" or ".nii.gz":
        # Prepare image
        if image.dim() == 3:
            image = image[None, :, :, :]

        # Prepare data
        if transform_affine == None:
            image = flip(image, dims=[1]).permute((1, 2, 3, 0))
            data = nib.Nifti1Image(image.numpy(), affine=None)

        else:
            image = image.permute((1, 2, 3, 0))
            data = nib.Nifti1Image(image.numpy(), affine=transform_affine)

        # Write data
        nib.save(data, path_data)
