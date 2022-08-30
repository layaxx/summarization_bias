from Abstractiveness import calculate_abstractness
from Compression_score import calculate_compression_score

data_path = 'data/formatted'

# Calculate Abstractness
abstractness = calculate_abstractness(data_path)
print("Abstractness is: " + str(abstractness))

# Calculate Compression Score
compression = calculate_compression_score(data_path)
print("Compression Score is: " + str(compression))

# Calculate IDS
# TODO: calculate IDS
