import numpy as np
import pandas as pd

#あ
DAT = 'data/iris.csv'
#あ

def main():
	# Read data
	dataset = pd.read_csv(DAT)

	data_eingeben = dataset.iloc[:,:-1].values
	data_leistung = dataset.iloc[:, -1].values

#あ
if __name__ == '__main__':
	main()
#あ