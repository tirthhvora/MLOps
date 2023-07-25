from load_data_script import load_and_clean_data

def preprocessing(data):
    
    import pandas as pd
    import numpy as np
    
    #data = load_and_clean_data()

    data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
    data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])
    
    categorical_vars = ['job','marital','education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    for var in categorical_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(data[var], prefix = var) # one hot encoding
        data_new = data.join(cat_list)
        data = data_new
    
    categorical_vars = ['job','marital','education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    data_vars = data.columns.values.tolist()
    
    keeping = [i for i in data_vars if i not in categorical_vars]
    
    final_df = data[keeping]
    
    final_df.columns = final_df.columns.str.replace(".", "_")
    final_df.columns = final_df.columns.str.replace(" ", "_")
    
    print(final_df.head())
    
    print("Education column pre-processed, categorical variables one-hot encoded. Ready to input data to model")
    
    return final_df
    
