import setuptools

setuptools.setup(
    name='creditpredict',
    version='0.2.0',
    author='Group 7',
    author_email='raja.pandey20@iimb.ac.in',
    description='Predicting whether the credit customer will buy a vehicle or not',
    url='https://github.com/rajap20/MLOps-project',
    license='MIT',
    packages=['creditpredict'],
    include_package_data=True,
    install_requires=['scikit-learn>=0.22.2']
)
