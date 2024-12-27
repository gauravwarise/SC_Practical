# List Example
fruits = ['apple', 'banana', 'cherry']
print('apple' in fruits)  # Output: True
print('orange' not in fruits)  # Output: True

# String Example
text = "Hello, world!"
print('Hello' in text)  # Output: True
print('hello' not in text)  # Output: True (case-sensitive)

# Dictionary Example (checking keys)
person = {'name': 'John', 'age': 25}
print('name' in person)  # Output: True
print('John' not in person)  # Output: True (checking value would return False)

# Combined Use Case
# Check if a fruit is in the list and a word is in the string, and if a key is in the dictionary
fruit_check = 'banana' in fruits  # True
word_check = 'world' in text  # True
key_check = 'age' in person  # True

# Output combined result
if fruit_check and word_check and key_check:
    print("All checks passed!")
else:
    print("Some checks failed.")
