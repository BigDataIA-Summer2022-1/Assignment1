#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def search_ship():
    '''
    Input: type: A string 'ship' or 'noship'
    Output: One of the images with ship(s) or noship in our dataset if input is 'ship' or 'noship', respectively.
    '''
    UserInput = input("Enter ship or noship")
    
    if UserInput == "ship":
        return "000155de5.jpg"
    elif UserInput == "noship":
        return "00003e153.jpg"
    else:
        return "Please type in 'ship' or 'noship'."

    
def test_search_ship(search_ship):
    assert search_ship()!="Please type in 'ship' or 'noship'."

