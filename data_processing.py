import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
import sys
import os

lib_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib_py'))
sys.path.append(lib_py_path)
from stats import Statistician


@dataclass
class Data_processing:
	st: Statistician

	#get data
	def get_data(self, data_path):
		dataset = pd.read_csv(data_path, header=None)
		return dataset

	#count NaN distribution by column
	def distribution_NaN(self, dataset, column):
		count_NaN = pd.isna(dataset[column]).sum()
		return count_NaN
	
	#check if column is filled by digits or NaN
	def check_digits(self, dataset, column):
		is_only_digits = dataset[column].apply(lambda x: isinstance(x, (float, int)) or pd.isna(x))
		if is_only_digits.all():
			return True
			
	# Calculate median without NaN
	def get_median(self, dataset, column):
		median = 0
		count_NaN = self.distribution_NaN(dataset, column)
		if self.check_digits(dataset, column) == True:
			size_minus_NaN = dataset.shape[0] - count_NaN
			buff= sorted(dataset[column])
			buff_no_NaN = buff[:size_minus_NaN - 1]
			median = self.st.quartil(buff_no_NaN, 50)
		return median
		
	#replace NaN by median value
	def replace_nan_to_median(self, dataset):
		for column in dataset.columns:
			median = self.get_median(dataset, column)
			#dataset[column] = dataset[column].fillna(median)
			dataset[column].fillna(median, inplace=True)
		return dataset
	
	# split dataset with random (80% for training set and 20% for test set)
	def random_data_split(self, dataset):
		train_df=[]
		test_df=[]

		x = dataset.shape[0]
		y = dataset.shape[1]
		print("data shape :", x, "X", y)
		random_figures = np.random.choice([1, 0], size=x, p=[0.8, 0.2])
		for i in range(x):
			if random_figures[i] == 1:
				train_df.append(dataset.iloc[i])
			else:
				test_df.append(dataset.iloc[i])
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
	st = Statistician()
	dp = Data_processing(st)	
	data_path = "data.csv"
	train_data_path = "train_data_path.csv"
	test_data_path = "test_data_path.csv"
	dataset = dp.get_data(data_path)
	dataset = dp.replace_nan_to_median(dataset)
	print(dataset[:])
	#dp.save_to_csv(data_path, train_data_path, test_data_path)

if	__name__ == "__main__":
	main()

