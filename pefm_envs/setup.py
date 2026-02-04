from setuptools import setup, find_packages

setup(
    name="pefm_envs",
    version="0.1",
    packages=find_packages(),
    package_data={
        "pefm_envs": [
            "sim_mobile/assets/**/*",
        ],
    },
    install_requires=[
        "numpy",
        "pybullet",
        "gym",
        "scipy",
        "cloudpickle",
        "opencv-python",
        "tqdm",
    ],
)
