from setuptools import setup, find_packages

setup(
    name="todo-app",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'todo-app=src.main:main',
        ],
    },
)