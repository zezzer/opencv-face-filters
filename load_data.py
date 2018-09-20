import numpy as np
import os
import cv2
import csv

def get_training_data():
	return read_data('data/training.csv')

def read_data(file, skip_lines=1):
	data = []
	labels = []
	with open(file) as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line_count = 0

	    for row in csv_reader:
	    	line_count += 1
	    	if line_count <= skip_lines or len(row) < 31 or '' in row:
	   			continue
	    	img = row[30].split(' ')
	    	label = [(float(x) if x else 0) for x in row[0:30]]
	    	data.append(img)
	    	labels.append(label)
	    	if line_count % 1000 == 0:
	    		print(line_count)

	return np.array(data), np.array(labels)