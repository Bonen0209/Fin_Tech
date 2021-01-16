import operator
import pandas as pd
import numpy as np
import scipy as sp
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import recmetrics
from ml_metrics.average_precision import mapk
from data_loader.datasets import AnimeDataset

def top_k_animes(df_anime_sim, anime_name, k=10):
    count = 1
    print('\nSimilar shows to {} include:\n'.format(anime_name))
    for anime in df_anime_sim.sort_values(by=anime_name, ascending=False).index[1:1+k]:
        print('No. {}: {}'.format(count, anime))
        count +=1

def top_k_users(df_user_sim, piv_norm, user, k=10):
    if user not in piv_norm.columns:
        return('\nNo data available on user {}'.format(user))
    
    print('\nMost Similar Users with User #{0}:\n'.format(user))
    sim_values = df_user_sim.sort_values(by=user, ascending=False).loc[:, user].tolist()[1:1+k]
    sim_users = df_user_sim.sort_values(by=user, ascending=False).index[1:1+k]
    zipped = zip(sim_users, sim_values)
    for user, sim in zipped:
        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim))

def predicted_rating(df_user_sim, piv, anime_name, user):
    sim_users = df_user_sim.sort_values(by=user, ascending=False).index
    user_values = df_user_sim.sort_values(by=user, ascending=False).loc[:, user].tolist()
    rating_list = []
    weight_list = []
    for j, i in enumerate(sim_users):
        rating = piv.loc[i, anime_name]
        similarity = user_values[j]
        if np.isnan(rating):
            continue
        elif not np.isnan(rating):
            rating_list.append(rating*similarity)
            weight_list.append(similarity)

    return sum(rating_list)/sum(weight_list)

def similar_k_user_recs(df_user_anime, df_user_sim, piv_norm, user, k=10):
    if user not in piv_norm.columns:
        return('No data available on user {}'.format(user))
    
    sim_users = df_user_sim.sort_values(by=user, ascending=False).index[1:1+k]
    best = []
    most_common = {}
    
    for i in sim_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i]==max_score].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1

    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    
    sorted_list = [anime for anime, ratings in sorted_list]

    return sorted_list

    #watched_list = df_user_anime[df_user_anime['user_id'] == user]['name'].unique()
    #recommended_list = []
    #for anime, watches in sorted_list:
    #    if anime not in watched_list and watches > 1:
    #        recommended_list.append(anime)

    #return recommended_list

def HitRate(predicts, targets):
    hits = 0
    total = 0

    predicts_anime = {anime for anime, _ in predicts}
    targets_anime = set(targets)

    return len(predicts_anime & targets_anime) / len(targets)
    

def main():
    # Data directory
    data_dir = Path('../Data/final/')

    # Anime Dataset
    dataset = AnimeDataset(data_dir=data_dir)

    # Get anime and rating dataframes
    df_anime, df_rating = dataset.get_whole_dataframe()

    # Replace the rating -1 with nan
    df_rating['rating'].replace({-1: np.nan}, inplace=True)

    # Merge anime and rating dataframes based on anime
    df_merged = df_rating.merge(df_anime, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])
    df_merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)

    # Pick the wanted columns
    df_merged = df_merged[['user_id', 'name', 'user_rating']]

    # Restricted the user to ID < 10000
    df_merged_sub= df_merged[df_merged['user_id'] <= 40000]
    
    # Get the normalized pivot table
    piv = df_merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
    piv_norm = piv.apply(lambda x: (x-x.mean())/(x.max()-x.min()), axis='columns')

    # Fill the Nan
    piv_norm.fillna(0, inplace=True)
    piv_norm = piv_norm.T

    # Delete unrated users
    piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]

    # Create the sparse matrix
    piv_sparse = sp.sparse.csr_matrix(piv_norm.values)

    # Calculate the simularity and turn into dataframe
    anime_similarity = cosine_similarity(piv_sparse)
    user_similarity = cosine_similarity(piv_sparse.T)
    df_anime_sim = pd.DataFrame(anime_similarity, index = piv_norm.index, columns = piv_norm.index)
    df_user_sim = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)

    # Predicting
    top_k_animes(df_anime_sim, 'Hunter x Hunter')
    top_k_users(df_user_sim, piv_norm, 3)

    print('\nWatched list')
    for anime in df_merged_sub[df_merged_sub['user_id']==3]['name']:
        print(anime)

    print('\nRecommended list')
    #for anime, _ in similar_k_user_recs(df_merged_sub, df_user_sim, piv_norm, 3)[:10]:
    for anime in similar_k_user_recs(df_merged_sub, df_user_sim, piv_norm, 3):
        print(anime)
    #print(similar_k_user_recs(df_merged_sub, df_user_sim, piv_norm, 3))
    #print(predicted_rating(df_user_sim, piv, 'Cowboy Bebop', 3))

    hit_rates = []
    for user in piv_norm.columns.to_list()[:1000]:
        print(user)
        recommended_list = similar_k_user_recs(df_merged_sub, df_user_sim, piv_norm, user)[:10]
        watched_list = df_merged[df_merged['user_id'] == user]['name'].unique()

        # Hit rate
        hit_rate = HitRate(recommended_list, watched_list)
        print("\nHit Rate: ", hit_rate)
        mean_average_precision = mapk(watched_list, recommended_list)
        hit_rates.append(hit_rate)

    print("\nAverage Hit Rate: ", sum(hit_rates)/len(hit_rates) )


if __name__ == '__main__':
    main()
