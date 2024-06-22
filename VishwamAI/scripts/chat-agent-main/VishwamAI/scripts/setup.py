from setuptools import setup, find_packages

setup(
    name='VishwamAI',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'torch',
        'onnx',
        'onnx2keras',
        'transformers',
        'tf-keras',
        'Pillow',
        'numpy',
        'scipy',
        'requests',
        'beautifulsoup4',
    ],
    entry_points={
        'console_scripts': [
            'vishwamai = VishwamAI.scripts.vishwamai_prototype:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='VishwamAI: An image-generating chat model with self-improvement mechanisms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/VishwamAI',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
