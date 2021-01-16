import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from base import BaseDataset
from utils import read_csv


class AnimeDataset(BaseDataset):
    """
    Anime dataset
    """
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.anime_filename = self.root_dir / 'anime.csv'
        self.rating_filename = self.root_dir / 'rating.csv'

        self.sentence_bert = SentenceTransformer('paraphrase-distilroberta-base-v1')

        #self.df_combine, self.df_anime, self.df_rating = self._prepare_data()
        self.df_anime, self.df_rating = self._prepare_data()

        ## Indexes
        self.anime_indexes = ['name', 'genre', 'type', 'episodes', 'rating', 'members']
        self.rating_indexes = [col for col in self.df_rating.columns]
        #self.combine_indexes = [col for col in self.df_combine.columns]

        #print(self.df_anime)
        #print(self.df_rating)

    def _prepare_data(self):
        df_anime = read_csv(self.anime_filename, index_col='anime_id')
        df_rating = read_csv(self.rating_filename)

        ## Split genre string to list
        #df_anime['genre'] = df_anime['genre'].str.split(',')

        ## Embed the name
        #df_anime['name_embeddings'] = df_anime['name'].apply(self.sentence_bert.encode)

        # Fill Nan with 0 in rating column
        df_anime['rating'].fillna(0, inplace=True)

        # Fill Nan with Other in genre and type columns
        df_anime['genre'].fillna('Other', inplace=True)
        df_anime['type'].fillna('Other', inplace=True)

        # Handle duplicates
        df_rating = df_rating.drop_duplicates(subset=['anime_id', 'user_id'])

        ## Handle rating dataframe
        #df_rating = df_rating.pivot(index='user_id', columns='anime_id', values='rating')

        ## Replace -1 with mean
        #df_rating = df_rating.replace(-1, {col: df_anime.loc[col, 'rating'] for col in df_rating.columns if col in df_anime.index})
        #df_rating = df_rating.replace(-1, 0)

        ## Transpose rating dataframe
        #df_rating = df_rating.T

        ## Fill Nan with 0 in rating dataframe
        #df_rating.fillna(0, inplace=True)

        ## Join target
        #df_combine = pd.merge(df_anime, df_rating, left_index=True, right_index=True, how="outer")

        # Sort the indexes
        df_anime = df_anime.sort_index()

        ## Fill Nan with 0 in combine dataframe
        #df_combine['genre'].fillna('Other', inplace=True)
        #df_combine['type'].fillna('Other', inplace=True)
        #df_combine['name'].fillna('', inplace=True)
        #df_combine['episodes'].fillna(0, inplace=True)
        #df_combine['rating'].fillna(0, inplace=True)
        #df_combine['members'].fillna(0, inplace=True)
        #df_combine.fillna({col: 0 for col in df_rating.columns}, inplace=True)

        #print(df_anime)
        #print(df_rating)
        #print(df_combine)

        ## Save dataframe
        #df_anime.to_csv(self.root_dir/'anime_clean.csv')
        #df_rating.to_csv(self.root_dir/'rating_clean.csv')
        #df_combine.to_csv(self.root_dir/'combine_clean.csv')

        #return df_combine, df_anime, df_rating
        return df_anime, df_rating

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
        #data = self.df_hotel[self.df_hotel.columns.difference(['label'])].iloc[idx]
        #target = self.df_hotel['label'].iloc[idx]
        #return data.to_numpy(), target

    def get_whole_dataframe(self):
        return self.df_anime, self.df_rating
        #if self.training:
        #    data = self.df_hotel[self.df_hotel.columns.difference(['label'])]
        #    target = self.df_hotel['label']
        #    return data, target
        #else:
        #    data = self.df_hotel[self.df_hotel.columns.difference(['label'])]
        #    return data

