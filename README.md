# photogrammetry-photos-pipeline

A pipeline to prepare jpeg photos for photogrammetry reconstruction, with:
 * dehazing
    * he-2009: Single Image Haze Removal Using Dark Channel Prior
    * meng-2013: Efficient Image Dehazing with Boundary Constraint and Contextual Regularization
 * calibrating with external GPX coordinates

## Install

    pip install photogrammetry-photos-pipeline

## Run

    photogrammetry-photos-pipeline \
        --images /path/to/orig/*.JPG \
        --output /path/to/output \
        --dehaze he-2009|meng-2013 \
        --gpx /path/to/track.gpx


## Dependencies
 * he-2009 method was copied from https://github.com/He-Zhang/image_dehaze
 * meng-2013 method from https://pypi.org/project/image-dehazer/
 * coordinates-label-photos from coordinates-label-photos

## License
MIT

## Author

Alexandre Masselot
