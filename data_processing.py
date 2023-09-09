import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
		
	#replace NaN by median value, 'M':1 and 'B':0
	def replace_nan_to_median(self, dataset):
		diag = {'M': 1, 'B': 0}
		dataset_01 = pd.DataFrame()
		dataset_01 = dataset.iloc[:, 1:].copy()
		for column in dataset_01.columns:
			median = self.get_median(dataset_01, column)
			dataset_01[column].fillna(median, inplace=True)
		new_dataset = dataset_01.replace(diag)
					
		print("dataset_01", new_dataset)
		return new_dataset
	
	def normalizator(self, dataset):
		normalized_dataset = pd.DataFrame(dataset)
		bm_column = normalized_dataset.iloc[:, 0]
		
		normalized_dataset = normalized_dataset.iloc[:, 1:]
		scaler = StandardScaler()
		scaler.fit(normalized_dataset)

		normalized_data = scaler.transform(normalized_dataset)
		columns = normalized_dataset.columns  # Get the column names before transformation

		normalized_dataset = pd.DataFrame(data=normalized_data, columns=columns)
		normalized_dataset.insert(0, 0, bm_column)
		return normalized_dataset
	
	# split dataset with random (80% for training set and 20% for test set)
	def random_data_split(self, dataset):
		train_df=[]
		test_df=[]

		df = pd.DataFrame(dataset)
		#print("pandas df:", df)
		x = dataset.shape[0]
		y = dataset.shape[1]
		print("data shape :", x, "X", y)
		random_figures = np.random.choice([1, 0], size=x, p=[0.8, 0.2])
		for i in range(x):
			if random_figures[i] == 1:
				train_df.append(df.iloc[i])
			else:
				test_df.append(df.iloc[i])
		x = len(train_df)
		y = len(train_df[0])
		print("train_df shape :", x, "X", y)
		x = len(test_df)
		y = len(test_df[0])
		print("train_df shape :", x, "X", y)
		return train_df, test_df

	#save file
	def save_to_csv(self, normalized_dataset, train_data_path, test_data_path):
		train_df, test_df = self.random_data_split(normalized_dataset)
		train_df = pd.DataFrame(train_df)
		test_df = pd.DataFrame(test_df)
		train_df.to_csv(train_data_path, index=False)
		test_df.to_csv(test_data_path, index=False)

def	main():
	st = Statistician()
	dp = Data_processing(st)	
	data_path = "data.csv"
	train_data_path = "train_data.csv"
	test_data_path = "test_data.csv"
	dataset = dp.get_data(data_path)
	dataset = dp.replace_nan_to_median(dataset)
	normalized_dataset = dp.normalizator(dataset)
	dp.save_to_csv(normalized_dataset, train_data_path, test_data_path)

if	__name__ == "__main__":
	main()

