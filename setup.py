from setuptools import setup, find_packages

setup(
    name='anime-face-detector',
    version='0.1.2',
    description='Anime Face Detector using mmdet and mmpose',
    readme='README.md',
    license='MIT',
    author='hysts',
    python_requires='>=3.10,<3.12',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.21.3,<2.0',
        'opencv-python-headless>=4.5.4.58',
        'torch==2.9.1',
        'torchvision',
    ],
    extras_require={
        'demo': [
            'gradio>=4.0.0',
        ],
        'openmmlab': [
            'mmengine==0.10.7',
            'mmcv==2.1.0',
            'mmdet==3.2.0',
            'mmpose==1.3.2',
        ],
    },
    url='https://github.com/hysts/anime-face-detector',
)