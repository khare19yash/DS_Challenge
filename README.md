**Briefly describe the conceptual approach you chose! What are the tradeoffs?** 

To solve this classification problem, I implemented a four-layer feed forward neural network.  The network consists of three hidden layers of size 128,64 and 32 respectively and an output layer. After the output layer there is a SoftMax layer which predicts the probability of each class.

Activation Function – ReLU  

Loss Function – Cross Entropy Loss 

Optimizer – Adam Optimizer 

The main tradeoff of using a Neural Network is overfitting as we increase the model capacity (by increasing the depth of the model) there is a high chance of overfitting. 

For data preprocessing the following steps were taken  

   - As the data was imbalance so to handle it oversampling was used 
   - Missing values in numerical variables were replaced by mean 
   - Numerical values were normalized between 0 and 1 
   - Missing values in categorical variables were replaced by NA 
   - All categorical values were converted into one hot encoding 

**What is the model performance? What is the complexity? Where are the bottlenecks?** 

The model has a training set accuracy of 83% and validation set accuracy of 81%. 

As the forward pass and backward pass in a feed forward neural network involves matrix multiplication and the time complexity of matrix multiplication is O(n^3) so the time complexity of neural network is higher than that. In fact, in a neural network the learning phase (backpropagation) is slower than the inference phase (forward propagation). This is because the backward pass involves gradient descent which has to be repeated many times.  

As the neural network involves storing various weight matrices so the space complexity of a neural network is also higher than O(n^2). 

**If you had more time, what improvements would you make, and in what order of priority?**

The following improvements can be made in the given order of priority -  

   1. Explore other techniques to handle imbalance data (in this model I have used oversampling). 
   2. Probably train a regression model to replace missing values.
   3. Tuning various hyperparameters to make the model more robust. 
   4. Trian different models like Random forests, SVM and compare their accuracy 

## Important Notes
**As the test data is too big (892816 rows) it is better to divide it in small chunks (Probably 100000 rows or less) and then run the model otherwise out of memory error may appear**

**I have uploaded both the python script and the ipython notebook. To understand the model and all the data preprocessing steps use the ipyton notebook and to get the result run the python script.**
```
python model.py
```
