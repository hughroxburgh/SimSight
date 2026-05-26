from setuptools import find_packages, setup

NAME = 'SimSight'
DESCRIPTION = 'Sightline ray tracer and computation suite for cosmological simulations'
URL = 'https://github.com/rhoxu/SimSight'
EMAIL = 'roxburghhugh@gmail.com'
AUTHOR ='Hugh Roxburgh'
VERSION = '0.0.1'
REQUIRED = ['astropy',
            'healpy',
            'shapely',
            'tqdm',
            'scipy',
            'numba',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'joblib'
            ]



setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author_email=EMAIL,
    author=AUTHOR,
    license='MIT',
    packages=find_packages(),#['tessellate'],
    install_requires=REQUIRED,
    include_package_data=True
)




