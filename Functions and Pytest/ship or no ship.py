#!/usr/bin/env python
# coding: utf-8

from skimage.io import imread

def search_ship():
    '''
    Input: type: A string 'ship' or 'noship'
    Output: One of the images' numpy array with ship(s) or noship in our dataset if input is 'ship' or 'noship', respectively.
    '''
    UserInput = input("Enter ship or noship")
    
    if UserInput == "ship":
        return imread('input/train_v2/' + "000155de5.jpg")
    elif UserInput == "noship":
        return imread('input/train_v2/' + "00003e153.jpg")
    else:
        return "Please type in 'ship' or 'noship'."

    
def test_search_ship(search_ship):
    assert search_ship()!="Please type in 'ship' or 'noship'."

