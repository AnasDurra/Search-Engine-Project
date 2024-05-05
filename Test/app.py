from dataset.antique_reader import AntiqueReader

# Create an instance of AntiqueReader with the file path
readerFile = AntiqueReader('C:\\Users\\fares\\Downloads\\antique-collection.txt')

# Call the load_as_dict() method to load the file contents into a dictionary
key_value_pairs = readerFile.load_as_dict()

# Print the dictionary containing key-value pairs
print(key_value_pairs)
