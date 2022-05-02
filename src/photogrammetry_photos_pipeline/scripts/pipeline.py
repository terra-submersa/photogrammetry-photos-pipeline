import logging
import os
import pathlib
import shutil

import argparse
from coordinates_label_photos.gpx import gpx_parser
from coordinates_label_photos.photos import list_photo_filenames, calibrate_photo
from tqdm import tqdm

from photogrammetry_photos_pipeline.dehaze import dehaze_he_2009, dehaze_meng_2013
from photogrammetry_photos_pipeline.exif import copy_exif


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='run a pipeline to prepare photo a photogrammetry asembly')

    parser.add_argument("--images", help="Images source directory", type=str, required=True,
                        metavar='/path/to/images-source')
    parser.add_argument("--output", help="Images destination directory", type=str, required=True,
                        metavar='/path/to/image-dest')
    parser.add_argument("--gpx", help=".gpx coordinates file for external GPS recording", type=str, required=False,
                        metavar='/path/to/gpx')
    parser.add_argument("--dehaze",
                        help="dehaze method, either "
                             "'meng-2013' (Efficient Image Dehazing with Boundary Constraint and Contextual Regularization)"
                             " or 'he-2009' (Single Image Haze Removal Using Dark Channel Prior)",
                        type=str, required=False, metavar='method',
                        choices=['meng-2013', 'he-2009']
                        )

    args = parser.parse_args()

    input_directory = getattr(args, 'images')
    output_directory = getattr(args, 'output')
    gpx_file = getattr(args, 'gpx')
    dehaze_method = getattr(args, 'dehaze')

    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    photos = list_photo_filenames(input_directory)
    for orig_photo in tqdm(photos):
        basename = os.path.basename(orig_photo)
        target = output_directory + '/' + basename
        if dehaze_method is None:
            shutil.copy(orig_photo, target)
        elif dehaze_method == 'he-2009':
            dehaze_he_2009(orig_photo, target)
            copy_exif(orig_photo, target)
        elif dehaze_method == 'meng-2013':
            dehaze_meng_2013(orig_photo, target)
            copy_exif(orig_photo, target)
        else:
            raise Exception('Cannot dehaze with method [%s]' % dehaze_method)

    if gpx_file is not None:
        logging.info('calibrating photos coordinates with %s' % gpx_file)
        track_coords = gpx_parser(args.gpx)
        calibrate_photo(photos=photos, track_coords=track_coords)


if __name__ == '__main__':
    main()
