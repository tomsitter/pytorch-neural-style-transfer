from setuptools import setup, find_packages

setup(
    name='pytorch-nst',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click>=7.1',
        'matplotlib>=3.2.1',
        'psutil>=5.7.0',
        'torch>=1.5.0',
        'torchvision>=0.6.0'
    ],
    entry_points='''
    [console_scripts]
    pytorch-nst=cli:cli
    ''',
    url="http://tomsitter.com/",
    description="Simple CLI for running Neural Style Transfer in PyTorch",
    keywords="pytorch neural style transfer ai cli",
    project_urls={
        "Source Code": "https://github.com/tomsitter/pytorch-neural-style-transfer"
    },
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)