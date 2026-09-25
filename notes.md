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


## Lesson 4b: Extra Insights

1. Why does the gradient shrink near the bottom?
The gradient is the slope of the loss curve. At the minimum, slope=0.
So gradient descent naturally slows down as it approaches the answer.

2. Local minima, saddle points, plateaus
- Linear regrassion: one bowl -shaped loss, one minimum. Always converges.
- Neural Networks: many minima, saddle points, plateaus.
  Solutions: random init, SGD (batched gradients), momentum, Adam optimizer.

3. Do w and b need the same learning rate?
No. Modern Optimizers (Adam, RMSProp) track a per-parameter effective learning rate autumatically. For a simple linearr model, one lr is fine.

4. What is Noise?
- Everything that affects the target but isn't in your features.
- Examples: house location, buyer motivation, geentics, photoblur.
      let us assume that two houses has same features 1500 sqft, 3 bedrooms, 5 year old but they have different prices because of other thing like location, interior, facilities, seller communication skills etc.
      this is called noise data
- Noise is irreducible - no model can drive loss below the noise floor.
The only way to reduce it is better features (feature engineering).


## Lesson 5: Deriving the MSE Gradient

# WHAT IT IS
The mathematical derivation of the gradient formula used in gradient descent.
shows why grad_w = 2 * mean(error * X) and grad_b = 2 * mean(error).

# WHY IT MATTERS:
Every ML model is trained by minimizing a loss. To do that, we need the gradient: the derivative of the loss with respect to each parameter.
Understanding the derivation means you can derive gradients for ANY loss,not just the ones in tutorials. It's the difference between using a library and building a model.

# WHAT A DERIVATIVE IS:
A derivative is a rate of change. It answers: "if I change this thing a tiny bit, how much does that other thing change?"

ex: the derivative of distance with respect to time is speed.

The gradient is the derivative of loss with respect to a parameter.
It tells you: "If I nudge w a tiny bit, how much does the loss change?"

# THE DERIVATION

step-1: Loss function
        L = (1/n) . Σ (ŷ_i - y_i)²
        where ŷ_i = w·x_i + b -> predicted value
              y_i -> actual value

step-2: Substitute the prediction
        L = (1/n) . Σ (w·x_i + b - y_i)²

step-3: Let error of exaample i  be u_i
        u_i =  w·x_i + b - y_i

step-4: Apply the power rule and chain rule:
        d/du (u²) = 2u
 
        d/dw (u²) = 2u · du/dw
              = 2u · x_i           (because du/dw = x_i)
              = 2 · error_i · x_i
        
        d/db (u²) = 2u · du/db
              = 2u · 1             (because du/db = 1)
              = 2 · error_i

step-5: The loss is a mean, so its derivative is the mean of the
        individual derivatives
        ∂L/∂w = 2 · mean(error · x)
        ∂L/∂b = 2 · mean(error)


# THE TWO CALCULUS RULES USED:

Power rule:
    d/du (u²) = 2u
    "derivative of something squared = 2 times that something"

Chain rule:
    d/dw (u²) = 2u · du/dw
    "derivative of the outside × derivative of the inside"

# WHY THE "2" APPEARS:
The power rule. Differentiating u² always brings down a factor of 2.

# WHY "x" APPEARS IN grad_w BUT NOT grad_b:
- w multiplies x, so du/dw = x (chain rule brings x into the result)
- b is added, so du/db = 1 (chain rule contributes nothing extra)

# WHY WE SQUARE THE ERROR (MSE):
- Raw errors cancel: +2 and -2 average to 0, falsely looking perfect
- Unary minus does NOT fix this - it just flips signs, still cancels
- Absolute value fixes cancellation but has a sharp corner at 0
  -> not differentiable -> gradient descent can't take smooth steps
- Squaring fixes cancellation AND is smooth everywhere
- Bonus: heavily penalizes large errors (great if you want to avoid big mistakes, bad if your data has outliers)

Comparison of loss functions:
    Raw error   mean(error)         → cancels, useless
    MAE         mean(|error|)       → no cancel, but not smooth at 0
    MSE         mean(error²)        → no cancel, smooth, penalizes big errors
    Huber       hybrid              → good compromise for outliers

# YOU DON'T HAVE TO USE MSE:
MSE is the default for regression, not the only option. 
choosing a loss = deciding which mistakes matter most:
          All mistakes equally bad       → MAE
          Big mistakes especially bad    → MSE
          Balance between the two        → Huber
          Predicting probabilities       → Cross-entropy

MSE is taught first because:
    - Easiest gradient to derive (power rule + chain rule)
    - Default in sklearn / PyTorch / TensorFlow
    - Teaches the general pattern: write loss -> differentiation -> get gradient
  
# GRADIENT CHECKING - WHY IT MATTERS IN PRACTICE:

When you write a new loss function or a new neural network layer, you compute the gradient yourself. One typo in the derivation and the training silently fails - the loss doesn't decrease, or it explodes to nan, or it converges to the wrong answer.

Gradient checking catches this before it wastes a training run:

How it works:
    1.Compute the gradient ANALYTICALLY (from the formula you derived)
    2.Compute it NUMERICALLY (nudge the parameter by a tiny epsilon, measure the loss change, divide by epsilon)
    3.They must match to ~5 decimal places
    4.If they don't match, your derivation has a bug

Real-world use:
    - Deep learning frameworks have gradient checking built in
    - PyTorch has torch.autograd.gradcheck()
    - Used to debug neural networks when training silently fails

Key insight:
    The numerical gradient doesn't need calculus - just subtraction.
    It's independent of your derivation, so it's a trustworthy ground truth to check your analytical work. If both methods came from the same derivation, they'd agree on wrong answers too. Different paths agreeing = correctness.

# INTERVIEW ANSWER:
- To train a model, we minimize the loss function with gradient descent.
- The loss for linear regression with MSE is the mean of the squared errors.
- We differentiate this using the power rule and chain rule. Letting u be the
  per-example error, we get d/dw(u²) = 2·error·x and d/db(u²) = 2·error.
- Since the loss is an average, its gradient is the average of the individual  gradients: ∂L/∂w = 2·mean(error·x) and ∂L/∂b = 2·mean(error).
- The '2' comes from the power rule.
- The 'x' comes from the chain rule -> x is what multiplies w, so it enters the gradient.
- For b, the coefficient is 1, so no x appears.
- We verify this numerically using gradient checking - nudging the parameter and measuring the loss change - and the two methods must match.
- In practice, gradient checking it is how engineers debug new loss functions and neural network layers.



## Lesson 6: First Real ML Pipeline

# WHAT IT IS
Load data → split into train/test → fit → predict → evaluate on test.
The full cycle of training and measuring a model.

# WHY IT MATTERS
This is the standard ML workflow. Every model, every company, every day.
The pipeline is identical for linear regression and for GPT. Only the
model changes.

# PIPELINE — 5 STEPS

1. Load data
     X = features   (n_samples, n_features)
     y = target     (n_samples,)

2. Split into train and test
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42)
     Typical: 80% train, 20% test.

3. Create and fit
     model = LinearRegression()
     model.fit(X_train, y_train)

4. Predict
     y_train_pred = model.predict(X_train)
     y_test_pred  = model.predict(X_test)

5. Evaluate
     mean_squared_error(y_true, y_pred)   # TRUE first, prediction second

# THE SKLEARN API — 3 METHODS
     fit(X, y)       → train
     predict(X)      → predict
     score(X, y)     → evaluate (R² for regression)

Every sklearn model (LinearRegression, RandomForest, SVM, KNN, ...) uses
the same three methods with the same argument order. Learn once, use all.

# WHY TRAIN vs TEST
- Train set = practice problems
- Test set  = final exam (data the model has never seen)
- If you evaluate on training data, you're checking if the model
  MEMORIZED. Test score is the only honest measure.
- Training error is always optimistically low.

# THREE FIT STATES

  Underfitting:  high train error + high test error
                 → model too simple
                 → fix: more complex model, more features

  Good fit:      low train error + low test error   (small gap)
                 → the goal

  Overfitting:   low train error + high test error  (big gap)
                 → memorized training data, can't generalize
                 → fix: simpler model, regularization, more data

The GAP between train and test error is the signal — not the absolute
numbers.

# DIABETES RESULTS (what we got)
  X shape: (442, 10)
  train: (353, 10)   test: (89, 10)   353 + 89 = 442 ✓
  Training MSE: 2868.55
  Test MSE:     2900.19
  Gap ~1% → well-fit model. Linear regression is simple, hard to overfit.

# KEY DETAILS
- random_state=42 → reproducible split. Without it, different results
  every run and you can't compare experiments.
- Argument order (y_true, y_pred) matters for accuracy, precision,
  recall, f1, confusion_matrix. For MSE it's symmetric, but build the
  habit now.
- model.coef_ = weights (one per feature)
- model.intercept_ = bias
- Print coeffs paired with feature names:
    for name, coef in zip(data.feature_names, model.coef_):
        print(f"{name}: {coef:.4f}")

# WHAT AN API IS
A contract between two pieces of software. Defines:
  - what you can request
  - how to request it
  - what you get back
Hides the implementation. It does NOT "call" anything — it's the
interface through which calls are made.

Restaurant analogy:
  Menu = the API.  Waiter = implementation.  Kitchen = the server.

# THE SKLEARN API
The consistent interface all sklearn models share: fit / predict / score.
That consistency is the API. Without it, every model would have different
method names and argument orders.

# TYPES OF APIs
- Library API:   np.mean(), model.fit()
- Web API:       GET https://api.github.com/users/...
- OS API:        open("file.txt")
- Hardware API:  drivers talking to hardware

In interviews, "API" usually means Web API. "sklearn API" means the
library's consistent interface pattern.

# DATA STORAGE & MEMORY AT SCALE

  < 1 GB         → load all into NumPy / pandas
  1–50 GB        → load into RAM, batch during training
  50 GB – 1 TB   → memory-map, Parquet/Arrow, generators
  > 1 TB         → distributed (Spark/Dask) or streaming (Kafka)

Techniques:
- Batching: feed small chunks, not the whole dataset. This IS what
  stochastic gradient descent (SGD) means. How all deep learning trains.
- Generators: yield one batch at a time. One example in memory.
- np.memmap: treat a disk file as an array. OS pages it in on demand.
- Parquet/Arrow: columnar storage. Read only the columns you need.
- PyTorch DataLoader: prefetch next batch while current trains.
  num_workers=8 → parallel processes feed the GPU.

Rule of thumb:
  Fits in RAM → load once, batch during training.
  Doesn't    → stream it.

# THE GPU BOTTLENECK
GPUs compute very fast; data loading is slow. Often the GPU idles waiting
for data. Fixes: prefetch, parallel loaders, fast SSDs (NVMe), keep data
in RAM between epochs.

# INTERVIEW ANSWER
"The standard pipeline is load, split, fit, predict, evaluate. I split
into train and test BEFORE training — evaluating on training data only
tells you if the model memorized. If train error is low and test error
is high, the model is overfitting; if both are high, it's underfitting;
a small gap means a good fit. sklearn uses a consistent API — fit,
predict, score — so once you learn it, every model works the same way.
For large data, I batch during training (that's SGD), use memory-mapping
or Parquet when data exceeds RAM, and stream when it exceeds a machine."

# MY QUESTION
How is memory handled when training on huge datasets?


## Lesson 7: R² and cross-validation

# WHAT IT IS
R² = a scale-free metric (how much better than predicting the mean).
Cross-validation = averaging performance over k train/test splits.

# WHY IT MATTERS
MSE alone doesn't tell you if a model is good - R² does.
A single train/test split is unreliable - cross-val fixes that

# R² (Coefficient of Determination)
                R² = 1 - (SS_res / SS_tot)
SS_res = sum of squared errors of your model
SS_tot = sum of squared errors of the baseline (predicting the mean)

- 1.0 -> Perfect
- 0.0 -> no better than predicting the mean
- <0  -> WORSE than predicting the mean (happens)
- 0.45 -> model explains 45% of the variance in y
         the other 55% is noise or unmodeled features or non-linear effects.
         Not necessarily "noise."

# CROSS-VALIDATION (K-FOLD)
1. Split data into k folds (ususally k=5)
2. Train on k-1, test on the remaining 1
3. Repeat k times, each time a different fold is the test set
4. Average the k scores

        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        scores.mean()   # best estimate of performance
        scores.std()    # how stable the model is

# WHY CV IS BETTER THAN ONE SINGE SPLIT
A single split's score depends entirely on which points landed in the test set. That score could be lucky or unlucky. Cross-validation averages over k different splits -> more stable estimate. Every point is tested exactly once and trained on k-1 times.

# MEAN AND STD - HOW TO READ THEM
- Mean = your best estimate of true performace
- std = Uncertainty around that estimate
- Report as : R² = mean(scores) ± std(scores)
      
ex: R² = 0.45 ± 0.15

WRONG: "score is 0.45 + 0.15 = 0.65"
RIGHT: "score is 0.45, but individual folds range roughly 0.30-0.60"

# INTERPRETING STD
- small std (0.02-0.05) -> model is stable and consistent
- Large std (0.10-0.20) -> model is unstable; performance depends on which data it sees - a warning sign

# INTERVIEW ANSWER
R² measures how much of the variance in the target my model explains, on a scale where 1 is perfect and 0 means no better than predicting the mean. Because a single train/test split gives an unreliable estimate that depends on which points happened to land in the test set, I use k-fold cross-validation: train on k-1 folds, test on the held-out fold, average k folds. The mean is my best estimate of performance, and std tells me how stable the model is. A large std means the model's performance varies a lot depending on the data it sees.

# MY QUESTION
How do i get my each fold's score?

