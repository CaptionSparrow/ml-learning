import numpy as np
import pandas as pd

#あ
DAT = 'data/carbon_nanotubes.csv'
CVN = 10
LAM = 0.1

np.set_printoptions(suppress = True)
#あ

def regression(x, y, λ):
	# Regulation term
	r = λ * np.eye(x.shape[1])

	# Compute weights
	w = np.linalg.inv(x.T.dot(x) + r).dot(x.T).dot(y.reshape(y.shape[0], 1))

	return w

def validation(flds, k, λ):
	re = []
	for f in range(k):
		# Regression for folder f
		e = np.concatenate([flds[0][i] for i in range(k) if i != f])
		b = np.ones([e.shape[0], 1])
		X = np.concatenate([e, b], axis = 1)
		Y = np.concatenate([flds[1][i] for i in range(k) if i != f])
		w = regression(X, Y, λ)

		# Validation for folder f
		eiv = flds[0][f]
		biv = np.ones([eiv.shape[0], 1])
		Xiv = np.concatenate([eiv, biv], axis = 1)
		Yiv = flds[1][f]
		Oiv = Xiv.dot(w)
		Err = (Yiv.reshape([Yiv.shape[0], 1]) - Oiv) ** 2

		# Mean and deviation of the square error
		me, de = np.mean(Err), np.std(Err)
		print("The accuracy of folder %d:" % f)
		print("Mean of SE\tDeviation of SE")
		print("%.10f\t%.10f" % (me, de))
		print("---")
		re.append([me, de])

	return np.array(re)

def fold_data(egb, ltg, k):
	# Shuffle the data
	schlurfen = np.random.permutation(egb.shape[0])
	egb = egb[schlurfen]
	ltg = ltg[schlurfen]

	# Fold k validations
	Größe = egb.shape[0] // k
	cuts = [Größe * i for i in range(1, k)]
	egbs = np.split(egb, cuts)
	ltgs = np.split(ltg, cuts)

	return egbs, ltgs

def main():
	# Read data
	dataset = pd.read_csv(DAT)

	data_eingeben = dataset.iloc[:,:-1].values
	data_leistung = dataset.iloc[:, -1].values

	eingeben = np.array(data_eingeben)
	leistung = np.array(data_leistung)

	# Cross Validation
	folders = fold_data(eingeben, leistung, CVN)
	results = validation(folders, CVN, LAM)

	# Print Result
	mfm, mfd = np.mean(results[:,0]), np.mean(results[:,1])
	print("")
	print("Results:")
	print("Mean of the folders' SE\t Mean of the SE's deviation")
	print("%.10f\t\t %.10f" % (mfm, mfd))
	print("---")

#あ
if __name__ == '__main__':
	main()
#あ