import numpy as np
import pandas as pd
import argparse

SEED = 7
np.random.seed(SEED)

args = {
      'dataset_dir': '../Data/hw1/'
}
args = argparse.Namespace(**args)

NEEDED_HEADERS = ['ID', 'school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3']

# Read data
df_data = pd.read_csv(f'{args.dataset_dir}train.csv')
df_test_no_G3 = pd.read_csv(f'{args.dataset_dir}test_no_G3.csv')

# Read numerical headers
num_headers = df_data.dtypes.loc[lambda x: x==np.int64].index.tolist()

# Pick data
df_data = df_data[NEEDED_HEADERS]
df_test_no_G3 = df_test_no_G3[[n for n in NEEDED_HEADERS if n!='G3']]

# Encode one-hot vector
df_data = pd.get_dummies(df_data)
df_test_no_G3 = pd.get_dummies(df_test_no_G3)

# Read whole headers and specify headers for x and y
headers = df_data.columns.tolist()
x_headers = [h for h in headers if h not in ['ID', 'G3']]
y_headers = 'G3'

# Normalize data
df_data[x_headers] = df_data[x_headers].apply(lambda x: (x-x.mean())/x.std())
df_test_no_G3[x_headers] = df_test_no_G3[x_headers].apply(lambda x: (x-x.mean())/x.std())

# Split train / test
shuffle = np.random.permutation(len(df_data))
split_idx = int(len(df_data)*0.8)
df_train = df_data.iloc[shuffle[:split_idx]]
df_test = df_data.iloc[shuffle[split_idx:]]

# Specify data
train_x = df_train[x_headers].values
train_y = df_train[y_headers].values
test_x = df_test[x_headers].values
test_y = df_test[y_headers].values
test_no_G3_x = df_test_no_G3[x_headers].values

# Save to .npy file
np.save(f'{args.dataset_dir}student_train_x.npy', train_x)
np.save(f'{args.dataset_dir}student_train_y.npy', train_y)
np.save(f'{args.dataset_dir}student_test_x.npy', test_x)
np.save(f'{args.dataset_dir}student_test_y.npy', test_y)
np.save(f'{args.dataset_dir}student_test_no_G3_x.npy', test_no_G3_x)
