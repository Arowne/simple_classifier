from sklearn import metrics
import numpy as np

# Dataset output if customer got car
# Loss function => MSE
# Activation function => Sigmoid
class SingleLayerPerceptron():
    
    def __init__(self, *args, **kwargs):
        np.random.seed(42)
        self.y = kwargs["y"]
        self.X = kwargs["X"]
        self.learning_rate = kwargs["learning_rate"]
        self.epoch = kwargs["epoch"]
        
        self.error = None
        self.predictions = None
        
        self.weights = np.random.rand(3,1)
        self.bias = np.random.rand(1)

    # Compute activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Derivate sigmoid ouput
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    # Turn data into valid output
    def output_layer(self, hidden_layer):
        np.place(hidden_layer, hidden_layer >= 0.5, 1)
        np.place(hidden_layer, hidden_layer < 0.5, 0)
        return hidden_layer

    def forward(self):
        z = np.dot(self.X, self.weights)+ self.bias
        return self.sigmoid(z)
    
    def backpropagation(self, predictions):

        self.predictions = predictions
        
        # outputs derivative
        dpred = self.sigmoid_derivative(predictions)
        error = self.loss_function()

        # Matricial chain rule applications with derivative of each output
        z_del = self.error * dpred
        inputs = self.X.T

        # Update weights with chain rule
        self.weights = self.weights - self.learning_rate*np.dot(inputs, z_del)
        
        for num in z_del:
            self.bias = self.bias - self.learning_rate*num
        
        return self.sigmoid(self.X)

    def loss_function(self):
        self.error = self.predictions - self.y
        return self.error
    
    def mae(self):
        return metrics.mean_absolute_error(self.y, self.predictions)

    def train(self):
        for epoch in range(self.epoch):
            predictions = self.forward()
            self.backpropagation(predictions)

            # MODEL PERFORMANCE INDICATION WITH MEAN ABSOLUTE ERROR (Notice that 0.5 is the limit for random model performance)
            print(self.mae())

    def output(self):
        hidden_layer = np.dot(self.X, self.weights) + self.bias
        result = self.output_layer(hidden_layer)
        return result

    def auc_roc(self, inputs_data=None, trained_data=None):
        fpr, tpr, thresholds = metrics.roc_curve(inputs_data, trained_data, pos_label=1)
        accuracy = metrics.auc(fpr, tpr)
        return accuracy
    
if __name__ == '__main__':

    # INPUTS => have job, have child, have garden
    input_set = np.array([[0,1,0],
                        [0,0,1],
                        [1,0,0],
                        [1,1,0],
                        [1,1,1],
                        [0,1,1],
                        [0,1,0]])

    # OUTPUTS => Got car (T/F)
    labels = np.array([[1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1]])

    labels = labels.reshape(7,1) #to convert labels to vector
    
    neural_network = SingleLayerPerceptron(X=input_set, y=labels, learning_rate=0.05, epoch=25000)
    neural_network.train()
    output = neural_network.output()
    model_accuracy = neural_network.auc_roc(inputs_data=labels, trained_data=output)

    print("MODEL ACCURACY USING AUC_ROC CURVE ( to compute true positive and false negative) => ")
    print(str(model_accuracy*100) + "%")