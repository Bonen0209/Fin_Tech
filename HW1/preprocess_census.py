import numpy as np
import pandas as pd
import argparse

SEED = 7
np.random.seed(SEED)

args = {
      'dataset_dir': '../Data/hw1/'
}
args = argparse.Namespace(**args)

# Headers
HEADERS = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country', 'Income']
X_HEADERS = [h for h in HEADERS if h not in ['Income']]
Y_HEADERS = 'Income'

# Read data
df_train = pd.read_csv(f'{args.dataset_dir}adult.data', header=None, names=HEADERS)
df_test = pd.read_csv(f'{args.dataset_dir}adult.test', header=None, names=HEADERS)
df_data = pd.concat([df_train, df_test])

# Read numerical headers
num_headers = df_data.dtypes.loc[lambda x: x==np.int64].index.tolist()

# Encode one-hot vector
df_x = pd.get_dummies(df_data[X_HEADERS])
df_y = pd.get_dummies(df_data[Y_HEADERS])[' >50K']

# Normalize data
df_x = df_x.apply(lambda x: (x-x.mean())/x.std())

# Split train / test
shuffle = np.random.permutation(len(df_data))
split_idx = int(len(df_data)*0.8)

# Specify data
train_x = df_x.iloc[shuffle[:split_idx]].values
train_y = df_y.iloc[shuffle[:split_idx]].values
test_x = df_x.iloc[shuffle[split_idx:]].values
test_y = df_y.iloc[shuffle[split_idx:]].values

# Save to .npy file
np.save(f'{args.dataset_dir}census_train_x.npy', train_x)
np.save(f'{args.dataset_dir}census_train_y.npy', train_y)
np.save(f'{args.dataset_dir}census_test_x.npy', test_x)
np.save(f'{args.dataset_dir}census_test_y.npy', test_y)
