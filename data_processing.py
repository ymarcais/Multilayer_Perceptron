import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

@dataclass
class Data_processing:
	
	# split dataset with random (80% for training set and 20% for test set)
	def random_data_split(self, data_path):
		data = pd.read_csv(data_path)
		train_df=[]
		test_df=[]

		x = data.shape[0]
		y = data.shape[1]
		print("data shape :", x, "X", y)
		random_figures = np.random.choice([1, 0], size=x, p=[0.8, 0.2])
		for i in range(x):
			if random_figures[i] == 1:
				train_df.append(data.iloc[i])
			else:
				test_df.append(data.iloc[i])
		x = len(train_df)
		y = len(train_df[0])
		print("train_df shape :", x, "X", y)
		x = len(test_df)
		y = len(test_df[0])
		print("train_df shape :", x, "X", y)
		return train_df, test_df

	def save_to_csv(self, data_path, train_data_path, test_data_path):
		train_df, test_df = self.random_data_split(data_path)
		train_df = pd.DataFrame(train_df)
		test_df = pd.DataFrame(test_df)
		train_df.to_csv(train_data_path, index=False)
		test_df.to_csv(test_data_path, index=False)

def	main():
	dp = Data_processing()
	data_path = "data.csv"
	train_data_path = "train_data_path.csv"
	test_data_path = "test_data_path.csv"
	dp.save_to_csv(data_path, train_data_path, test_data_path)

if	__name__ == "__main__":
	main()

