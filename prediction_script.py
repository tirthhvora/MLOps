from model_building_script import training_basic_classifier

def predict_on_test_data(model,X_test):
    import pandas as pd
    import numpy as np
    import sklearn

    #model = training_basic_classifier()

    print("---- Inside predict_on_test_data component ----")
    
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_pred = model.predict(X_test)
    np.save('data/y_pred.npy', y_pred)

    print("\n---- Predicted classes ----")
    print("\n")
    print(y_pred)

    return y_pred