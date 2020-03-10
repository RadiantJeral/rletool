# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = ["numpy"]


def get_extensions():

    return


setup(
    name="rletool",
    version="0.1",
    author="Clyde Lee",
    url="",
    description=" ",
    packages=find_packages(),
    package_dir={'rletool': 'rletool'},
    install_requires=requirements,
    # ext_modules=get_extensions(),
    # cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
