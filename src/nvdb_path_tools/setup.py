from setuptools import find_packages, setup

package_name = 'nvdb_path_tools'

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
    maintainer='itk',
    maintainer_email='itk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_publisher = nvdb_path_tools.path_publisher:main',
            'path_plotter = nvdb_path_tools.path_plotter:main',
            'nvdb_route_publisher = nvdb_path_tools.nvdb_route_publisher:main',
            'nmpc_path_follower_node = nvdb_path_tools.nmpc_path_follower_node:main',
        ],
    },
)
