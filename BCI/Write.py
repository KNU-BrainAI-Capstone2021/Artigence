from matplotlib import pyplot as plt
import os
import numpy as np

# #saving result path
save_dir = 'C:/Users/PC/Desktop/test.txt'


def write(input) :
	for i in range(10000):
		print(i)
	print("write input len:", len(input))
	
	# multi_classification output plotting
	plt.plot(input)
	plt.show()