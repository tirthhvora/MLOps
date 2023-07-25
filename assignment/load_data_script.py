def load_and_clean_data():
    
    import pandas as pd
    import numpy as np
    
    data = pd.read_csv("https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/banking.csv")
    
    print("Null/missingalues available in the data: \n")
    print(data.isna().sum())
    data = data.dropna()
    print("The data after dropping the na values are: \n")
    print(data.isna().sum())
    
    print("--------data imported and cleaned----------")

    return data