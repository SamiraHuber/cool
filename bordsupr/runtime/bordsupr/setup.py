from setuptools import find_packages, setup
from glob import glob
package_name = 'bordsupr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/msg', glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dynosam_optical_flow_node = bordsupr.dynosam_optical_flow_node:main',
            'dynosam_segmentation_node = bordsupr.dynosam_segmentation_node:main',
            'yolo_segmentation_node = bordsupr.yolo_segmentation_node:main',
            'face_detector_node = bordsupr.face_detector_node:main',
            'scene_description_node = bordsupr.scene_description_node:main',
            'stitched_scene_description_node = bordsupr.stitched_scene_description_node:main',
            'interaction_description_node = bordsupr.interaction_description_node:main',
            'database_node = bordsupr.database_node:main',
            'rotate_topics = bordsupr.rotate_topics:main',
            'slam_listener = bordsupr.slam_listener:main',
            'lidar_occupancy_map_node = bordsupr.lidar_occupancy_map_node:main',
            'video_recorder_node = bordsupr.video_recorder_node:main',
        ],
    },
)
