
import os
import joblib

def prepare_data(df):
    X= df.drop("y", axis= 1)
    y= df["y"]
    return X,y

## Save model
def save_model(model, filename):
    model_dir= "models"
    os.makedirs(model_dir, exist_ok=True)# create only if dir doesn't exist
    filePath= os.path.join(model_dir, filename)
    joblib.dump(model, filePath)

def load_model(filename):
    model_dir= "models"
    filePath= os.path.join(model_dir, filename)
    return joblib.load(filePath)