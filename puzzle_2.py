
def count_frequency(xs):
    """A function that takes in a list of items and returns a dictionary counting each items frequency."""
    frequency_dictionary = {} 
    for x in xs:
        

    if *key* in frequency_dictionary:
        frequency_dictionary[*key*] += 1 
    else:
        frequency_dictionary[0] = 1
    return frequency_dictionary

# :) heyyy

# example use case:
xs = ["0", "0", "1", "2", "0", "4", "6", "2"]

# because "0" appears 3 times, "1" appears once, etc. 
expected = {"0": 3, "1": 1, "2": 2, "4": 1, "6": 1}

your_result = count_frequency(xs)
print("You made:", your_result)
# will throw an error if they are not the same 
assert your_result == expected

print("If you see this printed then you did it!!!!")