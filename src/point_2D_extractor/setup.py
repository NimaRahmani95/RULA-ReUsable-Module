from setuptools import find_packages, setup

package_name = 'point_2D_extractor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shayan',
    maintainer_email='sshayan1997@gmail.com',
    description='Multi-camera 3D skeletal pose extraction using AlphaPose and RealSense depth',
    license='MIT', 
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'point_2D = point_2D_extractor.point_2D:main',
        ],
    },
)
