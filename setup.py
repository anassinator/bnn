#!/usr/bin/env python
"""Package setup."""

import os
from setuptools import setup, find_packages


def read(fname):
    """Reads a file's contents as a string.

    Args:
        fname: Filename.

    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


BASE_URL = "https://github.com/anassinator/bnn"
INSTALL_REQUIRES = [
    "torch==0.4.1",
]

# Parse version information.
# yapf: disable
version_info = {}
exec(read("bnn/__version__.py"), version_info)
version = version_info["__version__"]
# yapf: enable

setup(
    name="bnn",
    version=version,
    description="Bayesian neural network implementation in PyTorch",
    long_description=read("README.rst"),
    author="Anass Al",
    author_email="dev@anassinator.com",
    license="MIT",
    url=BASE_URL,
    download_url="{}/tarball/{}".format(BASE_URL, version),
    packages=find_packages(),
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
