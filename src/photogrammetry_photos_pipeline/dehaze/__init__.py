import cv2

from photogrammetry_photos_pipeline.dehaze.he_2009 import dark_channel, atm_light, transmission_estimate, \
    transmission_refine, recover
import image_dehazer


def dehaze_he_2009(src: str, target: str):
    src = cv2.imread(src)

    i = src.astype('float64') / 255

    dark = dark_channel(i, 15)
    a = atm_light(i, dark)
    te = transmission_estimate(i, a, 15)
    t = transmission_refine(src, te)
    j = recover(i, t, a, 0.1)

    cv2.imwrite(target, j * 255)


def dehaze_meng_2013(src: str, target: str):
    src = cv2.imread(src)
    dehazed = image_dehazer.remove_haze(src)
    cv2.imwrite(target, dehazed)
