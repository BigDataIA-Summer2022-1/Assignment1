# Author: Zifeng 
# Date: Jun 18th, 2022

import pandas as pd

def image_num_ships():
    # input: An integer: number of ships in an image.
    # output: How many images in our dataset with this certain number of ships.
    data = pd.read_csv('input/train_ship_segmentations_v2.csv')
    num = None
    while type(num) is not int or num < 0 or num > 15:
        try:
            num = input("Please enter an integer between 0 and 15:")
            num = int(num)
        except ValueError:
            print("%s is not an integer.\n" % num)
        
    #num = input("Please enter an integer between 0 and 15:")
    data['ships'] = data['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = data.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

    #print(f"Count of images with number (0, 1, 2 etc.) of ships \n{unique_img_ids['ships'].value_counts()}\n\n")
    count = 0
    for i in unique_img_ids['ships']:
        if i == num:
            count += 1
    return count

def test_image_num_ships(image_num_ships):
    assert image_num_ships()
