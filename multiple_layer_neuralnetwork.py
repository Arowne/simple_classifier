import numpy as np

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)


# loss function => AUC_ROCurve
# activation function => sigmoid
# dataset => XOR inputs
class NeuralNetwork():

    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs_len=len(self.inputs)
        self.input_length = len(self.inputs[0])

        self.wi=np.random.rand(self.input_length, self.outputs_len)
        self.wh=np.random.rand(self.outputs_len, 1)

    def forward_propagation(self, inp):
        layer_activation_1=sigmoid(np.dot(inp, self.wi))
        layer_activation_2=sigmoid(np.dot(layer_activation_1, self.wh))
        return layer_activation_2

    def train(self, inputs,outputs, it):
        for i in range(it):
            input_layer=inputs
            layer_1=sigmoid(np.dot(input_layer, self.wi))
            layer_2=sigmoid(np.dot(layer_1, self.wh))
            
            # Backpropagation
            layer_2_err = outputs - layer_2

            # Chain error with output derivative
            layer_2_delta = np.multiply(layer_2_err, sigmoid_der(layer_2))

            # Chain second layer output derivative to weight
            l1_err=np.dot(layer_2_delta, self.wh.T)

            # Get first layer output derivative to seconde layer derivative
            l1_delta=np.multiply(l1_err, sigmoid_der(layer_1))

            # Update weights
            self.wh+=np.dot(layer_1.T, layer_2_delta)
            self.wi+=np.dot(input_layer.T, l1_delta)

inputs=np.array([[0,0], [0,1], [1,0], [1,1] ])
outputs=np.array([ [0], [1], [1], [0] ])

n=NeuralNetwork(inputs)
n.train(inputs, outputs, 10000)
print(n.forward_propagation(inputs))