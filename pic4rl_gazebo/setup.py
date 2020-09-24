from setuptools import setup
import os
from glob import glob   

package_name = 'pic4rl_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds/empty_worlds'), glob('worlds/empty_worlds/*.model')),
        (os.path.join('share', package_name, 'worlds/4wall'), glob('worlds/4wall/*.model')),
        #('share/' + package_name + '/worlds' + '/empty_worlds', glob.glob(os.path.join('worlds', os.path.join('empty_worlds', 'empty_world_omnirob.model')))),
        #('share/' + package_name + '/worlds' + '/empty_worlds', glob.glob(os.path.join('worlds', os.path.join('empty_worlds', 'empty_world_burger.model')))),
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
        ],
    },
)
