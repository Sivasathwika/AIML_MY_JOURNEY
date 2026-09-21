import numpy as np

#Tiny problem: 3 examples, 1 feature

X = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])    #true relation: y = 2x
w = 1.5     # deliberately wrong so gradient is not zero                     
b = 0.0
#------Step-1: predictions, error, loss -----------------
y_pred = w * X +b
error = y_pred - y
loss = np.mean(error ** 2)
print(f"loss = {loss:.6f}")

#------Step-2: analytical gradient -----------------
# Use the formula you derived
grad_w_analytical = 2 * np.mean(error * X)
grad_b_analytical = 2 * np.mean(error)

#------Step-3: numerical gradient -----------------
# Idea: nudge the parameter by a tiny amount,
#       see how much the loss changes
#       divide by the nudge size.
epsilon = 1e-6

# nudge w
w_new = w + epsilon
loss_w = np.mean((w_new * X + b - y) ** 2)
grad_w_numerical = (loss_w - loss) / epsilon

# nudge b
b_new = b+ epsilon
loss_b = np.mean((w * X + b_new - y) ** 2)
grad_b_numerical = (loss_b - loss)/ epsilon


#------Step-4: compare -----------------
print(f"grad_w analytical: {grad_w_analytical:.6f}")
print(f"grad_w numerical: {grad_w_numerical:.6f}")
print()
print(f"grad_b analytical: {grad_b_analytical:.6f}")
print(f"grad_b numerical: {grad_b_numerical:.6f}")






