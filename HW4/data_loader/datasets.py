import talib
import pandas as pd
from base import BaseDataset
from utils import read_csv


class FinanceDataset(BaseDataset):
    """
    Finance dataset
    """
    def __init__(self, data_dir, filename):
        super().__init__(data_dir)
        self.filename = self.root_dir / filename
        self.df_finance = self._prepare_data()

    def _prepare_data(self):
        df = read_csv(self.filename, index_col='Date', parse_dates=True)

        # Moving Average
        df['SMA10'] = pd.DataFrame(talib.SMA(df['Close'].to_numpy(), 10), index=df.index)
        df['SMA30'] = pd.DataFrame(talib.SMA(df['Close'].to_numpy(), 30), index=df.index)

        # KD Line 
        df['K'], df['D'] = talib.STOCH(df['High'], df['Low'], df['Close'])

        # Specify features
        #data_index = df.columns.drop('Close')
        data_index = df.columns

        # Normalization
        normalized_df = (df[data_index] - df[data_index].min()) / (df[data_index].max() - df[data_index].min())
        #normalized_df['Close'] = df['Close']

        # Fill NA with 0
        normalized_df.fillna(0, inplace=True)

        return normalized_df.astype('float32')

    def __len__(self):
        return len(self.df_finance) - 30

    def __getitem__(self, idx):
        data = self.df_finance.iloc[idx:30+idx]
        target = self.df_finance['Close'].iloc[30+idx]
        return data.to_numpy(), target

    @classmethod
    def plot_data_characteristics(cls, df, start='2019-01-01', end='2019-12-31'):
        import matplotlib.pyplot as plt
        from mplfinance.original_flavor import candlestick2_ochl, volume_overlay

        # Pick data
        ohlcv = df.loc[start:end, ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA10', 'SMA30', 'K', 'D']]

        # Configure plots
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_axes([0.1,0.5,0.8,0.4])
        ax2 = fig.add_axes([0.1,0.4,0.8,0.1])
        ax3 = fig.add_axes([0.1,0.2,0.8,0.2])

        # Set title
        #fig.suptitle('Errorbar subsampling for better appearance')
        
        # Draw Candlestick and Moving Average
        candlestick2_ochl(ax,
            ohlcv['Open'],
            ohlcv['Close'],
            ohlcv['High'],
            ohlcv['Low'],
            width=0.6,
            colorup='r',
            colordown='g',
            alpha=0.75
        )
        ax.plot(ohlcv['SMA10'].to_numpy(), label='10')
        ax.plot(ohlcv['SMA30'].to_numpy(), label='30')
        
        # Draw KD Line
        ax2.plot(ohlcv['K'], label='K')
        ax2.plot(ohlcv['D'], label='D')
        
        # Draw Volume Bar
        volume_overlay(ax3,
            ohlcv['Open'],
            ohlcv['Close'],
            ohlcv['Volume'],
            colorup='r',
            colordown='g',
            width=0.7,
            alpha=0.6
        )

        # Set X labels
        ax3.set_xticks(range(0, len(ohlcv.index), 40))
        ax3.set_xticklabels(ohlcv.index[::40].strftime('%Y-%m-%d'))
        
        # Show legend
        ax.legend();
        ax2.legend();
        
        # Save figure
        plt.savefig('data_characteristics.png')

    @classmethod
    def plot_predict(cls, output, target, start='2018-01-01', end='2019-12-31'):
        pass
