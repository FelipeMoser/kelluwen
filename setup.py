from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="kelluwen",
    version="0.0.26",
    description="Open AI library for research and education.",
    packages=find_packages(),
    url="https://github.com/felipemoser/kelluwen",
    author="Felipe Moser",
    author_email="felipe.moser@univ.ox.ac.uk",
    install_requires=["torch>=1.10", "nibabel>=3.2", "matplotlib>=3.5", "typeguard>=2.13.3"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license_files=("LICENSE.txt"),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
