# Kelluwen

Open AI library for research and education designed to be easy to use and develop with.

## Installation

```python
pip install kelluwen
```

## Usage

```python
from kelluwen.functions.transforms import generate_affine, apply_affine
from kelluwen.functions.tools import show_midplanes
from torch import rand

# Create image tensor of shape BxCxDxHxW
img = rand(2, 3, 100, 100, 100)

# Define parameters for affine transform. These transforms can be independent for each channel. If there is no channel dimension, the same transform will be applied for all channels.
parameters = dict(parameter_translation=rand(2,3),
    parameter_rotation=rand(2,3),
    parameter_scaling=rand(2,3),
    type_rotation="euler_xyz",
    transform_order="trs",
    type_output="dict",
)

# Generate affine transform
transform = generate_affine(**parameters)

# Apply affine transform to image tensor
img_transformed = apply_affine(image=img, **transform)["image"]


# Show midplanes of volume. In this example we're using the RAS coordinate system, and we're scaling the features using a min-max method.
show_midplanes(
    image=img_transformed,
    title="Example midplanes",
    show=True,
    type_coordinates="ras",
    type_scaling="min_max")
# ```