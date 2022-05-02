import piexif


def copy_exif(src: str, target: str):
    piexif.transplant(src, target)
