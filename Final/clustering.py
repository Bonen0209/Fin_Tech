# -*- coding: utf-8 -*-
"""anime-recommendations-project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V6WmRtsyNncjlT-pFphWVNhhlC5e9RhH

<h2 style='text-align:center;font-family:Comic Sans MS;font-size:40px;background-color:lightseagreen;border:20px;color:white'>Anime Recommendations<h2>

![](https://i.pinimg.com/originals/a8/be/b0/a8beb06c120be3358360ae2be20588fd.gif)
    
<h2 style='text-align:center;font-family:Comic Sans MS;font-size:30px;background-color:lightseagreen;border:30px;color:white'>table of contents<h2>

## 1. Introduction
## 2. Data Id
## 3. Libraries
## 4. Preprocessing and Data Analysis
## 5. Cosine Similarity Model
## 6. Conclusion

# Introduction

This data set contains information on user preference data from 73,516 users on 12,294 anime. Each user is able to add anime to their completed list and give it a rating and this data set is a compilation of those ratings. The data was scraped thanks to [myanimelist.net](https://myanimelist.net) API.

![](https://i.pinimg.com/originals/a0/ee/ab/a0eeabadf50400a7ebd09ca29efc97db.gif)

# Data Id 📋

## Anime Dataset

This dataset is named **anime**. The dataset contains a set of **12,294 records** under **7 attributes**:

| Column Name | Description                                                    |
|-------------|----------------------------------------------------------------|
| `anime_id`  | myanimelist.net's unique id identifying an anime.              |
| `name`      | full name of anime.                                            |
| `genre`     | comma separated list of genres for this anime.                 |
| `type`      | movie, TV, OVA, etc.                                           |
| `episodes`  | how many episodes in this show. (1 if movie).                  |
| `rating`    |  average rating out of 10 for this anime.                      |
| `members`   | number of community members that are in this anime's "group".  |
                                                


## Rating Dataset

This dataset is named **rating**. The dataset contains a set of **7,813,737 records** under **3 attributes**:

| Column Name | Description                                                                        |
|-------------|------------------------------------------------------------------------------------|
| `user_id`   | non identifiable randomly generated user id.                                       |
| `anime_id`  | the anime that this user has rated.                                                |
| `rating`    | rating out of 10 this user has assigned (-1 if the user watched without assigning) |


### Aim of the Notebook:
Building a better anime recommendation system based only on similiar anime. 

![](https://miro.medium.com/max/1080/1*nq3tr7RFPqyoij72F8dnAw.gif)

# Libraries 📙📘📗📕
"""

import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter
import scipy as sp #pivot egineering


#ML model
from sklearn.metrics.pairwise import cosine_similarity


#default theme and settings
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

"""# Preprocessing and Data Analysis 💻
## First look at the data
### File Paths 📂
"""

#list all files under the input directory
for dirname, _, filenames in os.walk('../Data/final/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

rating_path = "../Data/final/rating.csv"
anime_path = "../Data/final/anime.csv"

"""### First few lines"""

rating_df = pd.read_csv(rating_path)
rating_df.head()

anime_df = pd.read_csv(anime_path)
anime_df.head()

"""### Data shapes and info"""

print(f"anime set (row, col): {anime_df.shape}\n\nrating set (row, col): {rating_df.shape}")

print("Anime:\n")
print(anime_df.info())
print("\n","*"*50,"\nRating:\n")
print(rating_df.info())

"""## Handling missing values 🚫"""

print("Anime missing values (%):\n")
print(round(anime_df.isnull().sum().sort_values(ascending=False)/len(anime_df.index),4)*100) 
print("\n","*"*50,"\n\nRating missing values (%):\n")
print(round(rating_df.isnull().sum().sort_values(ascending=False)/len(rating_df.index),4)*100)

"""It seems only the anime dataset has missing values."""

print(anime_df['type'].mode())
print(anime_df['genre'].mode())

"""Weirdly enough the mode value of `genre` is `Hentai`, the mode value of `type` is `TV`.
![](https://media1.tenor.com/images/008c75ee5f61121073f591b008eecec8/tenor.gif?itemid=13249584)
"""

# deleting anime with 0 rating
anime_df=anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type
anime_df['genre'] = anime_df['genre'].fillna(
anime_df['genre'].dropna().mode().values[0])

anime_df['type'] = anime_df['type'].fillna(
anime_df['type'].dropna().mode().values[0])

#checking if all null values are filled
anime_df.isnull().sum()

"""## Feeture Engineering 🐱‍💻

### Filling Nan values

In general the value `-1` suggests the user did not register a raiting so we will foll with `Nan` values.

"""

rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x==-1 else x)
rating_df.head(20)

"""### Now we will engineer our Dataframe in the following steps:

1. We want to recomment anime series only so the the relevant `type` is `TV`
2. We make a new Dataframe combining both anime and rating on the `anime_id` column.
3. Leaving only `	user_id`, `name` and `rating` as the Df.
4. For computing purpose only we compute our Df based only on the first 7500 users.

"""

#step 1
anime_df = anime_df[anime_df['type']=='TV']

#step 2
rated_anime = rating_df.merge(anime_df, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])

#step 3
rated_anime =rated_anime[['user_id', 'name', 'rating']]

#step 4
rated_anime_7500= rated_anime[rated_anime.user_id <= 7500]
rated_anime_7500.head()

"""### Pivot Table for similarity

We will create a pivot table of users as rows and tv show names as columns. The pivot table will help us will be analized for the calcuations of similarity.
"""

pivot = rated_anime_7500.pivot_table(index=['user_id'], columns=['name'], values='rating')
pivot.head()

"""### Now we will engineer our pivot table in the following steps:

1. Value normalization.
2. Filling `Nan` values as `0`.
3. Transposing the pivot for the next step.
4. Dropping columns with the values of `0` (unrated).
5. Using `scipy` package to convert to sparse matrix format for the similarity computation.

"""

# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)

"""# Cosine Similarity Model

![](https://media3.giphy.com/headers/CosineDotRip/dLeMRat9wmuZ.gif)

**formula:**
![](https://cdn-images-1.medium.com/max/579/1*5hJibEtQPavnbgRxg8w2Fg.gif)

Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction (more on [sciencedirect](https://www.sciencedirect.com/topics/computer-science/cosine-similarity)).
"""

#model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)

#Df of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)

def anime_recommendation(ani_name):
    """
    This function will return the top 5 shows with the highest cosine similarity value and show match percent
    
    example:
    >>>Input: 
    
    anime_recommendation('Death Note')
    
    >>>Output: 
    
    Recommended because you watched Death Note:

                    #1: Code Geass: Hangyaku no Lelouch, 57.35% match
                    #2: Code Geass: Hangyaku no Lelouch R2, 54.81% match
                    #3: Fullmetal Alchemist, 51.07% match
                    #4: Shingeki no Kyojin, 48.68% match
                    #5: Fullmetal Alchemist: Brotherhood, 45.99% match 

               
    """
    
    number = 1
    print('Recommended because you watched {}:\n'.format(ani_name))
    for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
        print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
        number +=1

anime_recommendation('Dragon Ball Z')

"""# Conclusion ✔

In this notebook, a recommendation algorithm based on cosine similarity was created.
For further analysis i sugggest prediction based on genres, or a user-user approach (“people like you, like that” logic).

If you liked the notebook please upvote!

![](https://media.tenor.com/images/a5721ade2ad3e7a1a3b45e73b1cd7ed1/tenor.gif)

<h2 style='text-align:center;font-family:Comic Sans MS;font-size:30px;background-color:lightseagreen;border:30px;color:white'>The End<h2>
"""