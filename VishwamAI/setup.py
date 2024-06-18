from setuptools import setup, find_packages

setup(
    name='VishwamAI',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'haiku',
        'jax',
        'jaxlib',
        'transformers'
    ],
    entry_points={
        'console_scripts': [
            'vishwamai=VishwamAI.scripts.main:main',
        ],
    },
    author='VishwamAI Team',
    description='VishwamAI Chat Agent',
    long_description=open('chat-agent/README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/VishwamAI/chat-agent',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
