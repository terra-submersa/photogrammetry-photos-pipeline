# photogrammetry-photos-pipeline

A pipeline to prepare jpeg photos for photogrammetry reconstruction, with:
 * dehazing
   * meng-2013: Efficient Image Dehazing with Boundary Constraint and Contextual Regularization
   * he-2009: Single Image Haze Removal Using Dark Channel Prior
 * calibrating with external GPX coordinates

## Install

    pip install photogrammetry-photos-pipeline

## Run

    photogrammetry-photos-pipeline \
        --images /path/to/orig/jpeg-images
        --output /path/to/output
        --dehaze he-2009|meng-2013
        --gpx /path/to/track.gpx


## Dependencies
 * he-2009 method was copied from https://github.com/He-Zhang/image_dehaze
 * meng-2013 method from https://pypi.org/project/image-dehazer/
 * coordinates-label-photos from coordinates-label-photos

## License
MIT

## Author

Alexandre Masselot
