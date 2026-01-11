import pickle
# For Question 2 you receive the images which are all fixed to be 64x64x3, the 64x64 is the pre-defined height and width of the image and the 3 is for the RGB (colour) values. You receive this for the background/not sweet pepper (e.g. y=0) and the sweet pepper (e.g. y=1). In short,
# * Q2_BG_dict.pkl has the data for y=0 (background)
# * Q2_SP_dict.pkl has the data for y=1 (sweet pepper)
#
# Exmaple code for loading:

for fname in ['Q2_BG_dict.pkl',  'Q2_SP_dict.pkl']:
	print("For the file:", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The number of entries for each set is:", len(data['train']), len(data['validation']), len(data['evaluation']))
		print("The size of each entry is:", data['train'][0].shape, data['validation'][0].shape, data['evaluation'][0].shape)

