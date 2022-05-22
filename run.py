import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from model import RobustModel
import numpy as np

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=0)

class ImageDataset(Dataset):
    """ Image shape: 28x28x3 """

    def __init__(self, root_dir, fmt=':06d', extension='.png'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = plt.imread(img_path)
        img = rgb2gray(img)
        #img = img.transpose(2,0,1)
        data = torch.from_numpy(img)
        return data


def inference(data_loader, model):
    """ model inference """
    model = model.double()
    model.eval()
    preds = []

    with torch.no_grad():
        for X in data_loader:
            y_hat = model(X.double())
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #1')
    parser.add_argument('--load_model', default='model_final.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./test/', help='image dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='test loader batch size')

    args = parser.parse_args()
    
    # setting for DEVICE
    device = torch.device('cpu') 

    # instantiate model
    model = RobustModel()
    model.load_state_dict(torch.load(args.load_model, map_location=device))

    # load dataset in test image folder
    test_data = ImageDataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    preds = inference(test_loader, model)

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))
