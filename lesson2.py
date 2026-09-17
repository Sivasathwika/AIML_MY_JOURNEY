import numpy as np

# 1.create twwo vectors
a = np.array([1,2,3])
b = np.array([10, 20, 30])

#print their shapes
print(a.shape)
print(b.shape)

# 2.Vector addition 
c = a + b
print("Vector addition:", c)

# 3.Scalar multiplication
d = a * 5
print("Scalar multiplication:", d)

# 4.Dot Product
# 1st way
way1 = np.dot(a,b)

# 2nd way
way2 = a @ b

#no sum just multiplication
mul = a * b

# 4th way sum of the multiplication
way3 = np.sum(mul)

print("Dot product using dot function in numpy:", way1)
print("Dot product using @ symbol:", way2)
print("Element-wise multiplication (a * b):", mul)
print("Dot product (np.sum(a * b)):", way3)

# 5.Do the dot product by HAND first ( no code):
# 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140

#