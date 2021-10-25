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
            'aideme.active_learning.factorization.utils',
            sources=['aideme/active_learning/factorization/utils' + ext]
        ),
    ]

    if use_cython:
        from Cython.Build import cythonize
        extension_modules = cythonize(extension_modules, language_level='3', annotate=True)

    return extension_modules


def get_install_requirements(use_cython):
    install_req = [
        'numpy>=1.17.4',
        'scipy>=1.3.1',
        'scikit-learn>=0.22.1',
    ]

    if use_cython:
        install_req.append('cython>=0.29.14')

    return install_req


if '--use-cython' in sys.argv:
    USE_CYTHON = True  # Cython is only needed when building, not installing
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

setup(
    # METADATA
    name='aideme',  # package name id -> used when 'pip install ...'
    version='1.0.dev0',  # project version

    # DESCRIPTION
    description='An Active Learning-based interactive data exploration tool',
    long_description=read_README_file(),
    long_description_content_type='text/markdown',

    # CONTACT and URLS
    maintainer='Luciano Di Palma',
    maintainer_email='luciano.di-palma@polytechnique.edu',
    url='https://aideme.netlify.app',  # project home page
    project_urls={
        'Source Code': 'https://gitlab.inria.fr/ldipalma/aideme/'
    },

    # REQUIREMENTS
    python_requires=">=3.7, <4",  # python required version
    install_requires=get_install_requirements(USE_CYTHON),  # minimum required packages - these packages will be installed when running 'pip install'

    # easily run our test suite using pytest
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # TAGGING OUR PROJECT
    keywords='machine-learning data-exploration artificial-intelligence active-learning',
    classifiers=[  # tags used to index our project
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here
        'Programming Language :: Python :: 3',
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
