from setuptools import setup, find_packages
import sys

setup(name='karpul',
      packages=[package for package in find_packages()
                if package.startswith('karpul')],
      install_requires=[
        'tensorboard>=2.0',
        'opencv-python',
        'matplotlib',
        'numpy',
        'gitpython'],
      description='Personal Utility Library',
      author='Jinwei Xing',
      url='https://github.com/KarlXing/karpul',
      author_email='jinweixing1006@gmail.com',
      version='0.1',
      license='MIT')