from setuptools import setup

setup(
    entry_points={
        'console_scripts': [
            'photogrammetry-photos-pipeline=photogrammetry_photos_pipeline.scripts.pipeline:main',
        ],
    }
)
