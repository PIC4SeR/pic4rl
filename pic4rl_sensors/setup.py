from setuptools import setup

package_name = 'pic4rl_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='enricosutera',
    maintainer_email='enricosutera@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #'my_script = pic4rl_sensors.lidar:main',
            'test_sensor = pic4rl_sensors.test_sensor:main',
            'laser_scan_sensor = pic4rl_sensors.laser_scan_sensor:main'
        ],
    },
)
