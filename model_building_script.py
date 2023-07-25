from data_split_script import train_test_split
def training_basic_classifier(X_train,y_train):
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    import pickle
    
    #X_train, X_test, y_train, y_test = train_test_split()

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train, y_train)

    #mlflow.set_tracking_uri("http://localhost:8000")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 150)
        # Log any other hyperparameters you want to track
        
        mlflow.sklearn.log_model(model, "model")
        
        with open(f'data/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
    print("\nRandomForest classifier is trained on banking data and saved to PV location /data/model.pkl ----")
    return model
