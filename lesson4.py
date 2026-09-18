# Lesson4: How training actually works

"""
1. Start with w=0, b=0   (or random)
2. Loop many times:
   a. Predict: y_pred = w * x + b          for every house
   b. Error:   error = y_pred - y_true
   c. Loss:    mean(error²)                one number = how wrong we are
   d. Gradient of w: how does loss change if I nudge w?
   e. Gradient of b: how does loss change if I nudge b?
   f. Update:  w = w - learning_rate * grad_w
               b = b - learning_rate * grad_b
3. After many loops, w and b are close to the true relationship.
"""

import numpy as np

#Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10      # 1000 houses, size 0-10
y = 3 * X + 2 + np.random.randn(100, 1) * 2 #true: w=3, b=2, with noise

# Initialize
w = 0.0
b = 0.0
lr = 0.01
epochs = 1000
n = len(X)

#Training loop
for epoch in range(epochs):
    #1.predictions: y_pred = w * X + b
    y_pred = w * X +b

    #2.Error: error = y_pred -y 
    error = y_pred - y

    #3.Loss (for tracking): loss = np.mean(error ** 2)
    loss = np.mean(error ** 2)

    #4.Gradients
    grad_w = 2 * np.mean(error * X)
    grad_b = 2 * np.mean(error)

    #5.Update
    w -= lr * grad_w

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: grad_w={grad_w:.4f}, lr={lr}, step size={lr*grad_w:.6f}, w={w:.4f}")

    b -= lr * grad_b

    #print loss every 100 epochs to watch it decrease
    if epoch %100 == 0:
        print(f"Epoch {epoch}: loss ={loss:.4f}, w={w:.4f}, b={b:.4f}")

#Final
print(f"\nFinal: w={w:.4f}, b={b:.4f}")
print(f"True: w=3.0, b=2.0")


#Compare with NumPy's own fit
w_truth, b_truth = np.polyfit(X.flatten(), y.flatten(), 1)
print(f"np.polyfit: w={w_truth:.4f}, b={b_truth:.4f}")