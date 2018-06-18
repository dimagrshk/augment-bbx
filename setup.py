from setuptools import setup

setup(name='augment-bbx',
      author='Dmytro Hrishko',
      author_email='dimagrshk@gmail.com',
      packages=['augmentation'],
      install_requires=["imgaug", "cv2", "scikit-image"])
