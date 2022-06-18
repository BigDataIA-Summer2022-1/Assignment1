# Author: Zifeng 
# Date: Jun 18th, 2022

from skimage.io import imread
import numpy as np
import pandas as pd
import matplotlib as plt


def img_and_masks(ImageId: str, ImgShape = (768, 768)):
    ''' 
    Input: ImageId : A string that contains the file name of the image in dataset; ImgShape was set to default as 768 by 768
    Output: A plot that shows the original image and the mask and the original image covered by the mask
    Return: img: a numpy array represents the original image, all_masks: a numpy array represents the mask of the image
    If the name of the file is invalid, return "No such file"
    '''
    # Reuse the decode function
    def rle_decode(mask_rle: str, shape):
        '''
        Input: mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    masks = pd.read_csv('input/train_ship_segmentations_v2.csv')
    num_masks = masks.shape[0]
    print('number of training images', num_masks)
    try:
        img = imread('input/train_v2/' + ImageId) # read local data
    except FileNotFoundError:
        print("No such file!")
        return "No such file"

    
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(ImgShape)

    for mask in img_masks:
        # Note that NaN should compare as not equal to itself
        if mask == mask:
            all_masks += rle_decode(mask, ImgShape).T

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
    return img, all_masks

def test_img_and_masks(img_and_masks):
    assert img_and_masks("000155de5.jpg") != "No such file" 