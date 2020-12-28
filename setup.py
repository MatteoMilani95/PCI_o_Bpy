"""Setuptools setup script."""
from setuptools import setup

setup(name='PCI_o_B',
      version='0.1',
      description='Analyze speckle fields, compute correlations, derive motion maps',
      url='https://github.com/MatteoMilani95/PCI_on_Bpy',
      author='Matteo Milani',
      author_email='matteo.milani@umontpellier.fr',
      license='GNU GPL',
      packages=['PCI_o_B'],
      install_requires=[
            'numpy',
            'scipy',
            'configparser',
            'pynverse',
            'IPython',
            'pandas'
      ], 
      #NOTE: other modules optionally used: emcee
      #      - emcee (VelMaps)
      #      - pandas (VelMaps)
      #      - astropy (PostProcFunctions)
      #test_suite='nose.collector',
      #tests_require=['nose'],
      zip_safe=False)
