
import logging
import os

#Setting up logs
logging_str= "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging_dir= "logs"
os.makedirs(os.path.join(logging_dir, "latest_log.log"), exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"latest_logs.log"), level=logging.INFO, format=logging_str)

def main(data, eta, epochs):
    ## Solved module not found(utils) error using --> https://stackoverflow.com/a/61532947
    ## Python path is set to oneNeuron dir, so we need to create all imported methods and 
    ## classes inside that root folder which has python path 
    from utils.model import Perceptron
    from utils.all_utils import prepare_data,save_model
    import pandas as pd
    import numpy as np
    df= pd.DataFrame(data)
    logging.info(f"this is dataframe passed to prepare data: \n {df}")
    X,y= prepare_data(df)
    model= Perceptron(eta, epochs)
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

## entry point of a python file at runtime
if __name__ == '__main__':

    AND={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
    }
    ETA= 0.3
    EPOCHS= 10

    try:
        main(data= AND, eta= ETA, epochs= EPOCHS)
    except Exception as e:
        logging.exception(e)
        raise e
