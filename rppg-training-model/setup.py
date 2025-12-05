from setuptools import setup, find_packages

setup(
    name='rppg-training-model',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for training models using Remote Photoplethysmography (rPPG) data.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'opencv-python',
        'mediapipe',
        'torch',
        'torchvision',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pyyaml',
        'jupyter',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)