from setuptools import setup, find_packages

setup(
    name='trading_gym',
    version='0.1',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy', 'pandas', 'matplotlib'],  # Added matplotlib
)