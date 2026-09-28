from setuptools import find_packages, setup

package_name = 'bordsupr_services'

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
            'dinov3_embedding_service = bordsupr_services.dinov3_embedding_service:main',
            'convnext_embedding_service = bordsupr_services.convnext_embedding_service:main',
            'osnet_embedding_service = bordsupr_services.osnet_embedding_service:main',
            'face_embedding_service = bordsupr_services.face_embedding_service:main',
        ],
    },
)
