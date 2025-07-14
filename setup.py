from setuptools import setup, find_packages

setup(
    name="loft",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "datasets"
    ],
    entry_points={
        "console_scripts": [
            "loft=loft.cli:main",
        ],
    },
)