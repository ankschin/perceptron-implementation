
## Solved module not found(utils) error using --> https://stackoverflow.com/a/61532947
## Python path is set to oneNeuron dir, so we need to create all imported methods and 
## classes inside that root folder which has python path 
from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model
import pandas as pd
import numpy as np


AND={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df= pd.DataFrame(AND)

df

X,y= prepare_data(df)
ETA= 0.3
EPOCHS= 10
model= Perceptron(eta= ETA, epochs=EPOCHS)
model.fit(X,y)

l= model.total_loss()


save_model(model,"and.model")


# to load and predict 
# loaded_model= load_model("and.model")
# loaded_model.predict(input)



# #passing single datapoint to predict /test
# input = np.array([[0,1],[1,1]])
# y_pred= model.predict(input)
# print(f"y pred: \n{y_pred}")
# loss= model.total_loss()
