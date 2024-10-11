from TimeLLM import TimeLLM
from data.dataloader import Dataset_ETT_hour
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def main():
    model = TimeLLM()
    dataset = Dataset_ETT_hour()
    data_loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(data_loader)):
        print(f"batch {i}")
        # print(torch.reshape(batch_x[0], (1, 384)))
        print(batch_x.shape)
        output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        # print(torch.reshape(output[0], (1, 384)))
        print(f"output shape {output.shape}")
        print(torch.reshape(output[0], (1, 96)))
        if i == 0:
            break


if __name__ == "__main__":
    main()