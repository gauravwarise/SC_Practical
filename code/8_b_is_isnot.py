# Using membership operators
fruits = ['apple', 'banana', 'cherry']
print('banana' in fruits)  # True (banana is in the list)
print('orange' not in fruits)  # True (orange is not in the list)

# Using identity operators
x = [1, 2, 3]
y = x  # y refers to the same object as x
z = [1, 2, 3]  # z is a new object with the same content

print(x is y)  # True (same object in memory)
print(x is z)  # False (different objects, even if they have the same content)
print(x is not z)  # True (different objects)
