#lesson 3: why dot product is core of every ML model

#Prediction by hand , not using any ML library. No training 
#using just a model with known weights, applied to a new house



import numpy as np

# Weights for a hypothetical house price model.
# Features (in this order): size_sqft, bedrooms, age_years
# Learned weights:
w = np.array([100, 5000, -1000])
#Bias:
b = 20000

#A house to predict:
#1500 sqft, 3 bedrooms, 5 years old
house = np.array([1500, 3, 5])

#1.compute the prediction
prediction = np.dot(w, house) + b
print("House price prediction:", prediction)

#2.Interpret the weights
# w[0] = 100 means each square foot adds $100 to the price
# w[1] = 5000 means each bedroom adds $5000 to the price
# w[2] = -1000 means each year of age reduces the price by $1000
# Bias b = 20000-> baseline price if all features were 0


#3.compute predictions for three more houses:
house_A = np.array([2500, 4, 2])
house_B = np.array([800, 1, 30])
house_C = np.array([1500, 3, 5])

prediction_A = np.dot(w, house_A) + b
prediction_B = np.dot(w, house_B) + b
prediction_C = np.dot(w, house_C) + b

print("House A price prediction:", prediction_A)
print("House B price prediction:", prediction_B)
print("House C price prediction:", prediction_C)


# 4.change the age of houseB from 30 to 5
house_B[2] = 5
prediction_B_new = np.dot(w, house_B) + b
print("House B new price prediction (age changed to 5):", prediction_B_new)

# When age dropped from 30 to 5 (25 years younger),
# price went up by 25 × 1000 = $25,000.
# This is w[2] = -1000 at work: each year of age reduces price by 1000.
