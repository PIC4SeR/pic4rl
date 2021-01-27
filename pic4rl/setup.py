import os

from glob import glob, iglob
from setuptools import setup, find_packages
package_name = 'pic4rl'


# for installing Gazebo models with subfolders
def get_data_files_list(install_path,relative_path):
    data_files_list = []

    for filename in iglob(relative_path+'/**/*', recursive=True):

        # avoid directories
        if not os.path.isdir(filename):  
            # create a 2 element tuple with directory and file
            data_files_list.append(
                (
                os.path.join(install_path,os.path.split(filename)[0]),
                [filename]
                )
            )
            

    # returns list
    #print(data_files_list)
    return data_files_list

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=[]),
    install_requires=['setuptools'],
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        #(os.path.join(os.path.join('share', package_name),"agents"),glob('agents/*.py')),
        #(os.path.join(os.path.join('share', package_name),"sensors"),glob('sensors/*.py')),
        #(os.path.join(os.path.join('share', package_name),"tasks"),glob('tasks/*.py')),
        #(os.path.join(os.path.join('share', package_name),"utils"),glob('utils/*.py')),

        # TO DO automatize setup of models
        #(os.path.join('share', package_name,"models"),glob('models/robots/s7b3/*.*')),       
        #(os.path.join('share', package_name,"worlds"),glob('worlds/*.world')),               
    #
    ]+ get_data_files_list(os.path.join('share', package_name),'gazebo/models')
    + get_data_files_list(os.path.join('share', package_name),'gazebo/worlds')
    + get_data_files_list(os.path.join('share', package_name),'config/')
    ,
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
