## Lesson 1: Data


# WHAT IT IS?
Data is a collection of examples, where each example has features (inputs) and often a target (what we want to predict).

# WHY IT MATTERS IN ML:
data is the basic need of ml because without data no training of the model

# Key facts:
1.data can be of any type int, float, str, boolean but for ml each type should be converted to numbers before training
2.the order of features should be consistent or remain same throught the dataset
3.rows are objects, entities or individual samples
4.columns are features of that particular object, entity or individual

# Interview Answer
- Data is the starting point of ML which consists of the info that require to train the models.
- ML models require numeric input; strings and booleans must be encoded first
- Garbage data in → garbage model out. Typos and wrong order corrupt silently.
- the performance of the model majorly depends on the quality of the dataset
- So data quality and consistency are as important as the algorithm


## Lesson 2 : Vector

# WHAT IT IS?
A vector is a ordered list of numbers 
for ex: list from python, array from numpy 
let cars = [[20000, 20, 5],
            [30000, 17, 6]]
    each list from cars is a vector 


# WHY IT MATTERS IN ML:
ML needs datasets for trainig the model, those datasets consists of ordered list of numbers which is a vector i.e A dataset consist of huge amounts of vectors 
So, vectors are essential for the efficient model training

# Key facts:
- Any ordered list of numbers can be a vector 
  it can be list or array 
- using list from python and performing operations on list is less efficient because in python the list contains pointers to individual python objects which has their own location, memory and overhead
- this speed can be improved using the array from numpy which uses compile c code for performin g operations which produces better speedup
- and also on list the element wise operations are done using loop which is inefficient because of slow speed
- but using numpy array saves time and performace approximately 25x times better comparatively

let a and b are arrays
- a + b → element-wise addition (returns a vector)
- a * 5 → scalar multiplication (returns a vector)
- a * b → element-wise multiplication (returns a vector)
- np.dot(a, b) or a @ b → dot product: multiply element-wise, then SUM (returns a SCALAR)
- The difference: a * b gives a vector; np.dot(a, b) gives a single number
- Dot product of [1,2,3] and [10,20,30] = 1×10 + 2×20 + 3×30 = 140

# Interview Answer
A vector is an ordered list of numbers. In ML, every example — every row of a dataset — is a vector of features. We use NumPy arrays rather than Python lists because NumPy does element-wise and matrix operations natively and 20–100x faster, using compiled C and contiguous memory. The dot product of two vectors is the sum of their element-wise products, and it's the core operation behind linear models and neural networks.

## Lesson 3: Linear model = Dot Product

# what it is:
prediction = np.dot(w, x) + b

# Why it matters:
This is the simplest ML model. Every linear regression,
logistic regression, and neural network layer is built on this

# Key facts:
- w = weights vector. Learned by the model
- x = features vector. One example
- b = bias. A single number. Baseline prediction when all features are 0.
- Each weight says"how much does this feature matter, and in which direction."
- Sign of weight = direction (+ pushes direction up, - pushes it down)
- Magnitude of weight = strength of effect
- Training = finding the w and b that make predictions match reality
- np.dot(w, x) is literally "multiply each feature by its weight, then sum"

# Interview answer:
A linear model computes prediction = w . x + b, which is a dot product of the weight vector and the feaature vector, plus a bias.
Each weight encodes how strongly a feature affects the prediction and in which direction.
Training means finding the weights and bias that minimize the difference between predictions and true values.


## Lesson 4: How training Actually Works (Gradient Descent)

# WHAT IT IS?
Training is the process of finding the weights and bias that make the model's predictions closest to the true values.
The algorithm that does this is called gradient descent.

# WHY IT MATTERS IN ML:
Every ML model - linear regression, logistic regression, neural networks, GPT - is trained using gradient descent.
It is the single most important algorithm in machine learning

# THE TRAINING LOOP:
1. Start with initial w and b (ususally 0 or random)
2. Repeat many times (epochs):
    a. Predict: y_pred = w * X + b
    b. Error:   error = y_pred - y_true
    c. Loss:    L = mean(error^2)
    d. Gradient:how does loss change if I nudge w? and b?
    e. Update: w -= lr * grad_w
               b -= lr * grad_b
3. After enough epochs, w and b converge to values that fit the data

# THE FORMULAS
- Prediction:     y_pred = w * X + b
- Loss (MSE):     L = mean((y_pred - y_true)^2)
- Gradient of w:  grad_w  = 2 * mean(error * X)
- Gradient of b:  grad_b = 2 * mean(error)
- Update rule:    w -= lr * grad_w
                  b -= lr * grad_b

# KEY FACTS
- Gradiet descent does NOT tru random weights and compare
  It COMPUTES the direction to move using the gradient
- The gradient is the slope of the loss. It tells you which way is downhill.
- Update sign: grad_w > 0 -> decreasing w reduce loss (the -=handles it).
- Learning rate (lr) is the step size:
  * too big -> Overshoots, weights blow up to nan (diverges)
  * too small -> Correct direction but too slow to converge
  * just right -> loss drops smoothly and converges
- Loss decreasing over epochs = the model is learning
- Loss plateauing = converged (or learning rate too small).
- With noise in the data: model recovers the "noisy best fit",
  not the exact generating paramenters. Loss has  floor > 0.
- Without noise: model converges to the exact true values. Loss -> 0
- np.polyfit uses the closed-form normal equation (one shot, exact).
  Gradient descent is iterative. 
  Both minimize MSE, so for linear regression they converge to the same answer. 
  Gradient descent is preferred in deep learning because it works when no closed form exists.

# EXPERIMENTS I RAN:
- lr = 0.01   -> loss 336 -> 3.22, w=2.92, b=2.42 (converged nicely)
- lr = 1.0    -> w=nan, b=nan (diverges - step size too big)
- lr = 0.0001 -> w=3.18, b=0.59 after 1000 epochs (too slow)
- Noise removed -> w =3.00, b=1.99, los=0.0000 (perfect fit)

# INTERVIEW ANSWER:
- Training a model means iteratively adjusting its parameters to minimize a loss function. For linear regression, we minimize mean squared error using gradient descent: compute predictions, compute the error, compute the gradient of the loss with respect to each weight, and take a small step in the opposite direction.
- The learning rate controls step size - too lrage diverges, too small converges too slowly.
- The gradient is the derivative of the loss: for MSE,  ∂L/∂w = 2·mean(error·x) and ∂L/∂b = 2·mean(error).
- The gradient decides the direction and steepness. The learning rate scales that step. The actual change is lr × gradient. Sign comes from the gradient, magnitude comes from lr × |gradient|.







