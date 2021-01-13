#from setuptools import setup, find_packages
#packages = find_packages()
#print(packages)

import os

from glob import glob, iglob
from setuptools import setup, find_packages
package_name = 'pic4rl'

# for installing Gazebo models with subfolders
def get_data_files_list(install_path,relative_path):
	data_files_list = []

	for filename in iglob(relative_path+'/**/*', recursive=True):
		print(filename)
		# avoid directories
		if not os.path.isdir(filename):  
			# create a 2 element tuple with directory and file
			data_files_list.append(
				(
				os.path.join(install_path,os.path.split(filename)[0]),
				filename
				)
			)
	# returns list
	print(data_files_list)
	return data_files_list

get_data_files_list(os.path.join('share', package_name),'gazebo')