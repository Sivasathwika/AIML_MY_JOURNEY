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

