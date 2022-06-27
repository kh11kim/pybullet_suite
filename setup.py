from setuptools import setup, find_packages

setup(
   name='pybullet_suite',
   version='0.0.1',
   description='Pybullet wrapper for easy use',
   author='Kanghyun Kim',
   author_email='kh11kim@kaist.ac.kr',
   packages=find_packages(exclude=["numpy", "scipy", "pybullet", "trimesh"]),  #same as name
)