# Copyright (c) OpenMMLab. All rights reserved.
from setuptools import find_packages, setup

def get_version(version_file='mmseg/version.py'):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(
        name='egohos',
        version=get_version(),
        description='Open MMLab Semantic Segmentation Toolbox and Benchmark',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        author='MMSegmentation Contributors',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, semantic segmentation',
        url='http://github.com/open-mmlab/mmsegmentation',
        packages=['egohos', 'mmseg'],
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        install_requires=[
            'torch', 'torchvision', 
            'mmcls>=0.20.1', 
            'mmcv-full>=1.4.4,<=1.6.0', 
            'supervision',
            'prettytable',
        ],
        extras_require={
        },
        ext_modules=[],
        zip_safe=False)
