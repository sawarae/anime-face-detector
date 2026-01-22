from setuptools import setup, find_packages

setup(
    name='anime-face-detector',
    version='0.2.0',
    description='Anime Face Detector using YOLOv8 and HRNetV2 (pure PyTorch)',
    readme='README.md',
    license='MIT',
    author='hysts',
    python_requires='>=3.10,<3.12',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.21.3,<2.0',
        'opencv-python-headless>=4.5.4.58',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'huggingface-hub>=0.20.0',
        'ultralytics>=8.0.0',
    ],
    extras_require={
        'demo': [
            'gradio>=5.0.0',
        ],
    },
    url='https://github.com/hysts/anime-face-detector',
)