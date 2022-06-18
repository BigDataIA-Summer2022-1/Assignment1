# Author: Zifeng 
# Date: Jun 18th, 2022

import numpy as np

def rle_decode(mask_rle: str, shape = (768, 768)):
    '''
    Input: mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    try:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
    except:
        return "The input run-length string cannot be decode"

def test_rle_decode(rle_decode):
    assert rle_decode('501676 3 502441 6 503209 7 503978 6 504746 6 505515 6 506283 6 507051 6 507820 6 508588 6 509357 6 510125 6 510893 6 511662 6 512430 6 513199 3 513967 1') != "The input run-length string cannot be decode"