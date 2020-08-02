import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize



extensions = [
    Extension(
        'so.correspond_pixels',
        [os.path.join('so', 'correspond_pixels.pyx')],
    ),
]

setup(

    ext_modules = cythonize(extensions)
)
