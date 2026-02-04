from setuptools import setup, find_packages

setup(
    name="pefm",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "pefm": ["configs/*.yaml"],
    },
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy",
        "einops",
        "hydra-core",
        "omegaconf",
        "wandb",
        "tqdm",
        "diffusers",
        "opencv-python",
    ],
)
