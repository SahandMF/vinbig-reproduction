from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='cxr_detection_replication',
    version='0.1.0',
    description='Unified CXR detection project with EfficientDet and MMDetection tools',
    author='Your Name',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
