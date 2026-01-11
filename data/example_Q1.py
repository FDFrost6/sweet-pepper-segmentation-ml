import pickle

# For Question 1 you receive the data matrices for the red class (e.g. y=1) and not background, or not coloured, class (e.g. y=0). You also have the data for the extension for the yellow class (e.g. y=2). In short,
# * Q1_BG_dict.pkl has the data for y=0 (background or not coloured).
# * Q1_Red_dict.pkl has the data for y=1 (red).
# * Q1_Yellow_dict.pkl has the data for y=2 (yellow).
#
# Exmaple code for loading:
for fname in ['Q1_BG_dict.pkl',  'Q1_Red_dict.pkl',  'Q1_Yellow_dict.pkl']:
	print("For the file:", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The size of the data matrix X for each set is:", data['train'].shape, data['validation'].shape, data['evaluation'].shape)


