import numpy as np
import logging
from tqdm import tqdm



class Perceptron():
    
    def __init__(self, eta, epochs) -> None:
        #initialize weights randomly--> these are when NN is initialized
        # create 3 weights w1,w2 and bias w0 , and multiply by very small no. to get a small value (randomly)
        self.weights = np.random.randn(3) * 1e-4
        logging.info(f'initial weights before training {self.weights}')
        self.eta= eta
        self.epochs = epochs

    
    # this is a unit step function
    def activationFunction(self, inputs, weights):
        """To perform dot product and apply activation fn. z= W*X and then act(z) .

        Args:
            inputs (array): array of inputs in array form
            weights (array): array of weights

        Returns:
            array: returns 1 wherever z elements is greater than 0
        """
        z= np.dot(inputs, weights)  
        # unit step act. fn.
        return np.where(z>0, 1, 0) 


    def fit(self, X, y):
        ## self making variables accesible globally in class
        self.X= X
        self.y= y
        
        ## Concatenating using np,.c_ ... X (3x1) and bias ([-1,-1,-1]) (3x1)
        X_with_bias= np.c_[self.X, -np.ones((len(self.X),1))]
        logging.info(f'x with bias : \n{X_with_bias}')

        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="Training the model"):
            logging.info("-"*20)
            logging.info(f"for epoch: {epoch}")
            logging.info("-"*20)

            #forward propagation
            y_hat= self.activationFunction(X_with_bias, self.weights)
            logging.info(f"predicted value after forward pass: \n{y_hat}")
            
            self.error= self.y - y_hat
            logging.info(f"Error: \n{self.error}")

            #backward propagation --> to update weights based on prediction error (using gradient of loss fn.) 
            ## gradient of simple diff error y-y' wrt W is below 
            self.weights= self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            logging.info(f"updated weights after this epoch: \n{epoch}/{self.epochs} : {self.weights}")
            logging.info("##"*20)


    def predict(self, X):
        X_with_bias= np.c_[X, -np.ones((len(X),1))]
        return self.activationFunction(X_with_bias, self.weights)         

    def total_loss(self):
        loss_total= np.sum(self.error)
        logging.info(f"total loss: \n{loss_total}")
