# Author: Zifeng 
# Date: Jun 18th, 2022

from PIL import Image
import numpy as np

def rle_encode(img: str):
    '''
    Input: img: mask image file name
    Return: run length encoding as string formated
    '''
    try:
        image = Image.open('input/train_v2/' + img) # read from local
    except FileNotFoundError:
        print("No such file!")
        return "No such file"
    image_array = np.array(image)
    pixels = image_array.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def test_rle_encode(rle_encode):
    assert rle_encode("000155de5.jpg") != "No such file"