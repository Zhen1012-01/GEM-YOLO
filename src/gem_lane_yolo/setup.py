from setuptools import setup

package_name = 'gem_lane_yolo'


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
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='YOLO lane detection node for GEM simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'dataset_collect_keyboard = gem_lane_yolo.dataset_collect_keyboard:main',
        'image_capture = gem_lane_yolo.image_capture:main',
        'lane_yolo_node = gem_lane_yolo.lane_yolo_node:main',
        'keyboard_drive_collect = gem_lane_yolo.keyboard_drive_collect:main',
        'lane_follow_node = gem_lane_yolo.lane_follow_node:main',
    ],
},

)
