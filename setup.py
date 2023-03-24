import os
from setuptools import setup, find_packages

setup_py_dir = os.path.dirname(os.path.realpath(__file__))
need_files = []
datadir = "pybullet_suite/assets"
hh = setup_py_dir + "/" + datadir
for root, dirs, files in os.walk(hh):
   for fn in files:
      ext = os.path.splitext(fn)[1][1:]
      if ext and ext in 'yaml index meta data-00000-of-00001 png gif jpg urdf sdf obj txt mtl dae off stl STL xml gin npy '.split():
         fn = root + "/" + fn
         need_files.append(datadir+"/"+fn[1 + len(hh):])

setup(
   name='pybullet_suite',
   version='0.0.8', 
   description='Pybullet wrapper for easy use',
   author='Kanghyun Kim',
   author_email='kh11kim@kaist.ac.kr',
   packages=find_packages(),  #same as name
   install_requires=[
      "numpy",
      "scipy",
      "pybullet",
      "trimesh"
   ],
   include_package_data=True,
   package_data={"pybullet_suite": need_files}
)

# python setup.py sdist bdist_wheel
# twine upload dist/*
