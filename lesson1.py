# A tiny dataset: 4 people
# Features (orprintder): height_cm, weight_kg, age
people = [
    [170, 70, 30],
    [160, 55, 25],
    [180, 85, 40],
    [165, 60, 22]
]

# 1.print the whole dataset
print("Full dataset:")
print(people)
print()

# 2.Print just the first person
print("First person:", people[0])


# 3.print the height of the third person
print("Height of third person:", people[2][0])

# 4.Print just the ages of all four people
ages = [person[2] for person in people]
print("All ages:", ages)


# 5.Create a new person and add them to list
people.append([170, 68, 28])
print("number of people:", len(people))


print("Weight of 3rd:", people[2][1])

#New row person 1 with years_of_experience added as 4th feature
# [ 170, 70, 30 , 6]  height=170, weight=70, age=30, years_of_experience=6