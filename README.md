Briefly describe the conceptual approach you chose! What are the tradeoffs? 
To solve this classification problem, I implemented a four-layer feed forward neural network.  The network consists of three hidden layers of size 128,64 and 32 respectively and an output layer. After the output layer there is a SoftMax layer which predicts the probability of each class. 
Activation Function – ReLU  
Loss Function – Cross Entropy Loss 
Optimizer – Adam Optimizer 

The main tradeoff of using a Neural Network is overfitting as we increase the model capacity (by increasing the depth of the model) there is a high chance of overfitting. 
