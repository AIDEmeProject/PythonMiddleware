import sys
from os import path

from setuptools import setup, find_packages
from setuptools.extension import Extension


def read_README_file():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()


def get_extension_modules(use_cython):
    ext = '.pyx' if use_cython else '.c'

    extension_modules = [
        Extension(
            'version_space_helper',
            sources=['aideme/active_learning/version_space/sampling/version_space_helper' + ext]
        )
    ]

    if use_cython:
        from Cython.Build import cythonize
        extension_modules = cythonize(extension_modules, language_level='3')

    return extension_modules


def get_install_requirements(use_cython):
    install_req = [
        'numpy>=1.17.4',
        'pandas>=0.25.3',
        'scipy>=1.3.1',
        'scikit-learn>=0.22.1',
    ]

    if use_cython:
        install_req.append('cython>=0.29.14')

    return install_req


USE_CYTHON = sys.argv[1] == 'build_ext'  # Cython is only needed when building, not installing


# from setuptools.command.build_ext import build_ext as _build_ext
#
# class build_ext(_build_ext):
#     def finalize_options(self):
#         _build_ext.finalize_options(self)
#         # Prevent numpy from thinking it is still in its setup process:
#         __builtins__.__NUMPY_SETUP__ = False
#         import numpy
#         self.include_dirs.append(numpy.get_include())




setup(
    # METADATA
    name='aideme',  # package name id -> used when 'pip install ...'
    version='1.0.dev',  # project version

    # DESCRIPTION
    description='An Active Learning-based interactive data exploration tool',
    long_description=read_README_file(),
    long_description_content_type='text/markdown',

    # CONTACT and URLS
    maintainer='Luciano Di Palma',
    maintainer_email='luciano.di-palma@polytechnique.edu',
    url='http://aideme.netflify.com/',  # project home page
    project_urls={
        'Source Code': 'https://gitlab.inria.fr/ldipalma/aideme/'
    },

    # REQUIREMENTS
    python_requires=">=3.6, <4",  # python required version
    install_requires=get_install_requirements(USE_CYTHON),  # minimum required packages - these packages will be installed when running 'pip install'

    # easily run our test suite using pytest
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # TAGGING OUR PROJECT
    keywords='machine-learning data-exploration artificial-intelligence active-learning',
    classifiers=[  # tags used to index our project
        'Development Status :: 3 - Alpha',  # How mature is this project?
        'Intended Audience :: Science/Research',  # Indicate who your project is intended for
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',  # our project license

        # Specify the Python versions you support here
        'Programming Language :: Python :: 3',  #  Python 3 only
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        # Topics
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    # LINKING MODULES
    zip_safe=False,
    packages=find_packages(),  # include all python packages within 'aideme' package
    ext_modules=get_extension_modules(USE_CYTHON),
)
