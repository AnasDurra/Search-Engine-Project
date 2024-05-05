from dataset.antique_reader import AntiqueReader

readerFile = AntiqueReader('C:\\Users\\fares\\Downloads\\antique-collection.txt')

key_value_pairs = readerFile.load_as_dict()

print(key_value_pairs)
