import numpy as np
import cv2


def cal_transmission(haze_img, transmission, regularize_lambda, sigma):
    rows, cols = transmission.shape

    kirsch_filters = load_filter_bank()

    # Normalize the filters
    for idx, currentFilter in enumerate(kirsch_filters):
        kirsch_filters[idx] = kirsch_filters[idx] / np.linalg.norm(currentFilter)

    # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
    w_fun = []
    for idx, currentFilter in enumerate(kirsch_filters):
        w_fun.append(calculate_weighting_function(haze_img, currentFilter, sigma))

    # Precompute the constants that are later needed in the optimization step
    t_f = np.fft.fft2(transmission)
    ds = 0

    for i in range(len(kirsch_filters)):
        d = psf2otf(kirsch_filters[i], (rows, cols))
        ds = ds + (abs(d) ** 2)

    # Cyclic loop for refining t and u --> Section III in the paper
    beta = 1  # Start Beta value --> selected from the paper
    beta_max = 2 ** 8  # Selected from the paper --> Section III --> "Scene Transmission Estimation"
    beta_rate = 2 * np.sqrt(2)  # Selected from the paper

    while beta < beta_max:
        gamma = regularize_lambda / beta

        # Fixing t first and solving for u
        du = 0
        for i in range(len(kirsch_filters)):
            dt = circular_conv_filt(transmission, kirsch_filters[i])
            u = np.maximum((abs(dt) - (w_fun[i] / (len(kirsch_filters) * beta))), 0) * np.sign(dt)
            du = du + np.fft.fft2(circular_conv_filt(u, cv2.flip(kirsch_filters[i], -1)))

        # Fixing u and solving t --> Equation 26 in the paper
        # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
        # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

        transmission = np.abs(np.fft.ifft2((gamma * t_f + du) / (gamma + ds)))
        beta = beta * beta_rate
    return (transmission)


def load_filter_bank():
    kirsch_filters = [np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                      np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                      np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                      np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                      np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                      np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                      np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                      np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                      np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])]
    return kirsch_filters


def calculate_weighting_function(HazeImg, Filter, sigma):
    # Computing the weight function... Eq (17) in the paper

    haze_image_double = HazeImg.astype(float) / 255.0
    if (len(HazeImg.shape) == 3):
        red = haze_image_double[:, :, 2]
        d_r = circular_conv_filt(red, Filter)

        green = haze_image_double[:, :, 1]
        d_g = circular_conv_filt(green, Filter)

        blue = haze_image_double[:, :, 0]
        d_b = circular_conv_filt(blue, Filter)

        w_fun = np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * sigma * sigma))
    else:
        d = circular_conv_filt(haze_image_double, Filter)
        w_fun = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
    return w_fun


def circular_conv_filt(Img, Filter):
    filter_height, filter_width = Filter.shape
    assert (filter_height == filter_width), 'Filter must be square in shape --> Height must be same as width'
    assert (filter_height % 2 == 1), 'Filter dimension must be a odd number.'

    filter_hals_size = int((filter_height - 1) / 2)
    rows, cols = Img.shape
    padded_img = cv2.copyMakeBorder(Img, filter_hals_size, filter_hals_size, filter_hals_size, filter_hals_size,
                                   borderType=cv2.BORDER_WRAP)
    filtered_img = cv2.filter2D(padded_img, -1, Filter)
    result = filtered_img[filter_hals_size:rows + filter_hals_size, filter_hals_size:cols + filter_hals_size]

    return result


##################
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img
