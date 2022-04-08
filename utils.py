def get_key(dict, val):
    # function to return key for any value
    for key, value in dict.items():
         if val == value:
             return key
 
    return False