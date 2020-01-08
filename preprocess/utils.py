import cv2


def mean_preprocess(image, rMean, gMean, bMean):
    # get each channel of image
    (R, G, B) = cv2.split(image)

    # subtract the means values from each channel
    R -= rMean
    G -= gMean
    B -= bMean

    # merge the channels back together and return the image
    return cv2.merge([R, G, B])
