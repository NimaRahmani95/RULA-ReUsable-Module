from setuptools import find_packages, setup

package_name = 'rula_calculator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/ergonomic_assistant.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shayan',
    maintainer_email='sshayan1997@gmail.com',
    description='3D RULA ergonomic scoring and UR5e gradient-descent Z-height optimiser',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rula_calculator = rula_calculator.rula_calculator:main',
            'proactive_rtde = rula_calculator.proactive_rtde_controller:main',
            'pcb_ergonomic_assistant = rula_calculator.pcb_ergonomic_assistant:main',
            'gesture = rula_calculator.gesture:main',
        ],
    },
)
