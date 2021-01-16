import data_loader.datasets as module_data
from pathlib import Path

def main():
    data_dir = Path('../Data/final/')

    dataset = module_data.AnimeDataset(data_dir=data_dir)

    #train_data_loader = module_data.FinanceDataLoader(data_dir, batch_size=128, shuffle=True, validation_split=0.2, num_workers=8, training=True)
    #valid_data_loader = train_data_loader.split_validation()
    #for batch_idx, (data, target) in enumerate(train_data_loader):
    #    print(data.shape)
    #    print(target.shape)


if __name__ == '__main__':
    main()
