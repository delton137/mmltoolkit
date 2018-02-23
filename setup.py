from setuptools import setup
from setuptools import find_packages

__version__ = '0.1.0'

setup(name='mmltoolkit',
      version='0.1.0',
      description='Library for molecular featurizations and utility functions for machine learning with molecules',
      author='Daniel C. Elton',
      author_email='delton17@gmail.com',
      url='https://github.com/delton137/mmltoolkit',
      include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Chemistry',
	    'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
      	'Intended Audience :: Science/Research'
     	],
      license='MIT',
      install_requires=['numpy', 'matplotlib', 'scikit-learn'],
      packages=find_packages(),
      zip_safe=False)
