#! /usr/bin/env python
import argparse
from collections import defaultdict
import numpy as np
import json
from functools import reduce # The reduce(fun,seq) function is used to apply a particular function passed in its argument to all of the list elements mentioned in the sequence passed along
import os
import operator
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path


TEST_IMAGES = './data/images'


def cos_similarity(v1, v2):
	return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

def retrieve_similar_image(img, top_n=20):
	# retrieve the vocabulary for the image in analysis
	vocab = image_voc[img][0]
	img_to_check = []

	# For each vocabulary, retrieve the list of relevant images
	# from inverted index. This will reduce the number of images to be
	# processed and hence speeding it up. 
	for ix, v in enumerate(vocab):
		if int(v) > 0:
			img_to_check.append(inverted_index[str(ix)])
	img_to_check = np.unique(reduce(operator.add, img_to_check)) # summing all the lists associated to different inverted_indeces
	similarities = defaultdict()

	# For each image that is relevant, compute the similarity score
	for image in img_to_check:
		similarities[image] = cos_similarity(
			np.array(computed_tfidf[img]),  # image we choose
			np.array(computed_tfidf[image]))[0][0] 
		print(cos_similarity(
			np.array(computed_tfidf[img]), 
			np.array(computed_tfidf[image]))[0][0])

	# Return the top n similar images
	out = dict(sorted(similarities.items(), 
	                  key=lambda x:x[1], # Choosing the second item from x
					  reverse=True)[:top_n]).keys()
	scores = dict(sorted(similarities.items(), 
	                  key=lambda x:x[1], # Choosing the second item from x
					  reverse=True)[:top_n]).values()
	print(scores)

	
	folder = Path(TEST_IMAGES)
	image_list = [str(name).split("/")[-1] for name in folder.glob("*.jpg")]

	#print(image_list)
    
	count = 0
	for im in out:
		count += 1
		if im in image_list:
			img = cv2.imread('./data/images/'+im)
			cv2.imshow("Image "+str(count), img)
			cv2.waitKey(0)

	return out


if __name__ == '__main__':
	path = os.getcwd()
	bovw_path = path + "/src/pretrained/"
	img_path = path + "/src/video_node/images/"

	with open(os.path.join(bovw_path, "inverted_index.txt")) as f:
	    inverted_index = json.loads(f.readlines()[0])[0]

	with open(os.path.join(bovw_path, "tfidf.txt")) as f:
		computed_tfidf = f.readlines()
		computed_tfidf = json.loads(computed_tfidf[0])[0]

	with open(os.path.join(bovw_path, "image_vocabs.txt")) as f:
		image_voc = f.readlines()
		image_voc = json.loads(image_voc[0])[0]
	
	print(retrieve_similar_image("frame1.jpg"))
