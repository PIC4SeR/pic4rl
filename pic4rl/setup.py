import os
from glob import glob
from setuptools import setup
package_name = 'pic4rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
        
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
        #'pic4rl_2_RL = pic4rl.pic4rl_2_RL:main',
        'pic4rl_training = pic4rl.pic4rl_training:main',
        'pic4rl_gazebo = pic4rl.pic4rl_gazebo:main',
        'pic4rl_environment = pic4rl.pic4rl_environment:main',
        'ddpg_agent = ddpgagent:main',
        ],
    },
)
