from preprocessing_script import preprocessing
import pandas as pd
def train_test_split(final_df: pd.DataFrame):

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    #final_df = preprocessing()
    
    X = final_df.loc[:, final_df.columns != 'y']
    y = final_df.loc[:, final_df.columns == 'y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 47)
    
    # np.save(f'data/X_train.npy', X_train) # saved as a numpy binary file (efficient to save and load)
    # np.save(f'data/X_test.npy', X_test)
    # np.save(f'data/y_train.npy', y_train)
    # np.save(f'data/y_test.npy', y_test)
    
    print("\n---- X_train ----")
    print("\n")
    print(X_train.head())
    
    print("\n---- X_test ----")
    print("\n")
    print(X_test.head())
    
    print("\n---- y_test ----")
    print("\n")
    print(y_test.head())

    return X_train, X_test, y_train, y_test