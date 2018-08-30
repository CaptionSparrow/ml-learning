import numpy as np
import pandas as pd

#あ
DAT = 'data/iris.csv'

LABELS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
NUMBER = len(LABELS)
NFEATS = 4
#あ

class Classifier():
	def __init__(self):
		self.weight = np.random.rand(NFEATS + 1, 1)
	
	def classify(self, X):
		b = np.ones([X.shape[0], 1])
		iX = np.concatenate([X, b], axis = 1)
		Y = iX.dot(self.weight)
		out = sigmoid(Y)
		print(out.shape)
		print(out)

def sigmoid(Y):
	return 1 / (np.exp(Y) + 1)

def train(pcat, ncat):
	classifier = Classifier()

	classifier.classify(pcat)

	return classifier

def main():
	# Read data
	dataset = pd.read_csv(DAT)
	categories = []
	for label in LABELS:
		categories.append(dataset[dataset['class'].isin([label])].iloc[:,:-1].values)

	# Train classifier for each 2 categories
	classifiers = []
	for p in range(NUMBER - 1):
		for n in range(p + 1, NUMBER):
			classifier = train(categories[p], categories[n])

#あ
if __name__ == '__main__':
	main()
#あ