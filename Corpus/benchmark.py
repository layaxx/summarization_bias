import sys
from Abstractiveness import calculate_abstractness
from Compression_score import calculate_compression_score
from IDS_Redundancy import calculate_ids

data_path = 'data/formatted'

# Calculate Abstractness
abstractness = calculate_abstractness(data_path)
print("Abstractness is: " + str(abstractness))

# Calculate Compression Score
compression = calculate_compression_score(data_path)
print("Compression Score is: " + str(compression))

# Calculate IDS
limit = 10
try:
    limit = int(sys.argv[1])
except Exception:
    pass  # ignored
redundancy, relevance = calculate_ids(data_path, limit=limit)
print("Redundancy is: " + str(redundancy))
print("Relevance is: " + str(relevance))
print("Used limit of " + str(limit))
