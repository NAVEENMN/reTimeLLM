import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class Dataset_ETT_hour(Dataset):
    def __init__(self, flag='train', size=None,
                 features='S', data_path='data/ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        global df_data
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in


def main():
    df_raw = pd.read_csv("ETTh1.csv")
    print(df_raw.head())

    """
    seq_len 512
    lable_len 48
    pred_len 96
    factor 3
    enc in 7
    dec in 7
    c_out 7
    batch size 24
    d_model=32
    d_ff=128
    """
    dataset = Dataset_ETT_hour()

    data_loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )

    # i, (seqs, labels, seq_m, label_m)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(data_loader)):
        print(f"batch {i + 1}:")
        print("\nbatch x: This is the input sequence for the models, containing the historical time series data points that the models will use to predict future values.")
        print("batch x:", batch_x.shape)
        print("\nbatch y: This is the target sequence for the models, containing the true future values that the models will try to predict.")
        print("batch y:", batch_y.shape)
        print("\nbatch x': This is the time encoding for the input sequence, which provides additional information about the time of day, day of the week, etc.")
        print("batch x':", batch_x_mark.shape)
        print("\nbatch y': This is the time encoding for the target sequence, which provides additional information about the time of day, day of the week, etc.")
        print("batch y':", batch_y_mark.shape)
        print("\n")

        if i == 2:
            break


if __name__ == "__main__":
    main()