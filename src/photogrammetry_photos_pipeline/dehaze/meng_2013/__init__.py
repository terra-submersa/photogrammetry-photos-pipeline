import cv2
import numpy as np
import copy

from photogrammetry_photos_pipeline.dehaze.meng_2013.air_light import air_light
from photogrammetry_photos_pipeline.dehaze.meng_2013.bon_con import bound_con
from photogrammetry_photos_pipeline.dehaze.meng_2013.cal_transmission import cal_transmission


def dehaze(HazeImg):
    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = air_light(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = bound_con(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = cal_transmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    return remove_haze(HazeImg, Transmission, A, 0.85)

#copied from https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python
def remove_haze(haze_img, transmission, a, delta):
    '''
    :param haze_img: Hazy input image
    :param transmission: estimated transmission
    :param a: estimated airlight
    :param delta: fineTuning parameter for dehazing --> default = 0.85
    :return: result --> Dehazed image
    '''

    # This function will implement equation(3) in the paper
    # " https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf "

    epsilon = 0.0001
    transmission = pow(np.maximum(abs(transmission), epsilon), delta)

    haze_corrected_image = copy.deepcopy(haze_img)
    if len(haze_img.shape) == 3:
        for ch in range(len(haze_img.shape)):
            temp = ((haze_img[:, :, ch].astype(float) - a[ch]) / transmission) + a[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            haze_corrected_image[:, :, ch] = temp
    else:
        temp = ((haze_img.astype(float) - a[0]) / transmission) + a[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        haze_corrected_image = temp
    return haze_corrected_image
