import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
################################
# Non Editable Region Starting #
################################
def my_fit(Z_train):
################################
#  Non Editable Region Ending  #
################################

    xor_data_map = {}
    
    # Transforming training data based on selection bits
    for i in range(Z_train.shape[0]):
        x = int(Z_train[i][-6] + 2 * Z_train[i][-7] + 4 * Z_train[i][-8] + 8 * Z_train[i][-9])
        y = int(Z_train[i][-2] + 2 * Z_train[i][-3] + 4 * Z_train[i][-4] + 8 * Z_train[i][-5])
        model_key = 16 * min(x, y) + max(x, y)  # Ensure x <= y for consistent keys

        # Initialize the entry in the map if it doesn't exist
        if model_key not in xor_data_map:
            xor_data_map[model_key] = []

        row_copy = Z_train[i].copy()
        if x > y:
            row_copy[-1] = 1 - row_copy[-1]  # Flip response if needed

        xor_data_map[model_key].append(row_copy)

    # Convert lists to numpy arrays for model training
    for key in xor_data_map:
        xor_data_map[key] = np.array(xor_data_map[key])

    trained_models = {}

    # Train LinearSVC models for each XORRO configuration
    for i in range(15):
        for j in range(i + 1, 16):
            model_key = 16 * i + j
            if model_key in xor_data_map:
             
                clf = LinearSVC(loss='squared_hinge',penalty='l2',tol=1e-4,c=11,dual='auto')                
                clf.fit(xor_data_map[model_key][:, :64], xor_data_map[model_key][:, -1])
                trained_models[model_key] = clf               
    return trained_models

################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, models):
################################
#  Non Editable Region Ending  #
################################

    X_pred = np.zeros(X_tst.shape[0])
    
    for i in range(X_tst.shape[0]):
        x = int(X_tst[i][-5] + 2 * X_tst[i][-6] + 4 * X_tst[i][-7] + 8 * X_tst[i][-8])
        y = int(X_tst[i][-1] + 2 * X_tst[i][-2] + 4 * X_tst[i][-3] + 8 * X_tst[i][-4])
        model_key = 16 * min(x, y) + max(x, y)  # Ensure x <= y for consistent keys

        # Check if model_key exists before prediction
        if model_key not in models:
            #print(f"Warning: Model for key {model_key} not found. Available keys: {list(models.keys())}")
            continue  # Skip this iteration if the model is missing

        feature_array = X_tst[i, :64].reshape(1, 64)  # Reshape for prediction
        pred = models[model_key].predict(feature_array)

        # Flip the prediction if needed
        if x > y:
            pred = 1 - pred
            
        X_pred[i] = pred
    
    return X_pred
#0.9613512399999309 2.244928359999176 83997.8 0.94276
