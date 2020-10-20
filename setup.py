from setuptools import find_packages, setup

setup(
    name='DIFFI',
    packages=find_packages(),
    version='0.1.0',
    description='Interpretable Anomaly Detection with DIFFI: Depth-based Feature Importance for the Isolation Forest',
    author='Carletti, Mattia and Terzi, Matteo and Susto, Gian Antonio',
    install_requires=[
        'scikit-learn',
    ],
    license='MIT',
    package_data={'data': ['*']},
    include_package_data=True,
)
