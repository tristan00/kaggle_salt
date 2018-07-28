import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision
import cv2
from pathlib import Path
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude, morphology
import tqdm
import random
# from torchsample.callbacks import EarlyStopping


device = "cuda"
border = 5
directory = '/home/td/Documents/salt/'


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutions are from VGG11
        self.encoder = models.vgg11().features

        # "relu" layer is taken from VGG probably for generality, but it's not clear
        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1, )

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))


def unet11(**kwargs):
    model = UNet11(**kwargs)

    return model


def get_model():
    model = unet11()
    model.train()
    return model.to(device)


def load_image(path, mask=False, is_test = False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    # img = cv2.imread(str(path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = Image.open(str(path)).convert('LA')

    if not is_test:
        rotation = random.randint(0,3)
        transpose =  random.randint(0,1) == 1
        strech = False
    else:
        rotation = 0
        transpose = False
        strech = False

    if mask:
        # Convert mask to 0 and 1 format
        # img = cv2.imread(str(path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[:, :, 0:1] // 255
        img = Image.open(str(path)).convert('LA')
        if strech:
            img = img.resize((128, 128), Image.LANCZOS)

        img_np = np.array(img.getdata())[:, 0]
        img = img_np.reshape(img.size[1], img.size[0])
        if transpose:
            img = np.transpose(img)
        img = np.rot90(img, rotation)

        if not strech:
            img = np.pad(img, ((0, 27), (0, 27)), 'constant')
        # img = np.dstack((np.expand_dims(img, axis=2),
        #                  np.expand_dims(img, axis=2),
        #                  np.expand_dims(img, axis=2)))

        img = (img > 127).astype((int))

        # img = img//255
        return torch.from_numpy(img).float()
    else:
        img = Image.open(str(path)).convert('LA')
        if strech:
            img = img.resize((128, 128), Image.LANCZOS)
        img_np =  np.array(img.getdata())[:, 0]
        img = img_np.reshape(img.size[1], img.size[0])
        if transpose:
            img = np.transpose(img)
        img = np.rot90(img, rotation)
        if not strech:
            img = np.pad(img, ((0, 27), (0, 27)), 'constant')

        x_center_mean = img[border:-border, border:-border].mean()
        x_csum = (np.float32(img) - x_center_mean).cumsum(axis=0)
        x_csum -= x_csum[border:-border, border:-border].mean()
        x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

        g = gaussian_gradient_magnitude(img, sigma=.4)
        img = img / 255.0
        img = np.dstack((np.expand_dims(img, axis=2),
                              np.expand_dims(g, axis=2),
                              np.expand_dims(x_csum, axis=2)))


        return torch.from_numpy(img).float().permute([2, 0, 1])


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False, is_train = False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.len_multiplier = 1
        #
        # if not is_test and not is_val:
        #     self.len_multiplier = 8

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index//self.len_multiplier]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path, is_test=self.is_test)

        if self.is_test:
            return (image,)
        else:
            mask = load_image(mask_path, mask=True)
            return image, mask


def main():
    depths_df = pd.read_csv(os.path.join(directory, 'train.csv'))

    train_path = os.path.join(directory, 'train')
    file_list = list(depths_df['id'].values)
    # file_list = file_list[:400]
    file_list_train, file_list_val = train_test_split(file_list, test_size=.05)
    # file_list_val = file_list[::10]
    # file_list_train = [f for f in file_list if f not in file_list_val]
    dataset = TGSSaltDataset(train_path, file_list_train)
    dataset_val = TGSSaltDataset(train_path, file_list_val)
    model = get_model()

    learning_rate = 1e-5
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    prev_val_loss = 99999.0

    results = []
    patience = 200

    for e in range(10000):
        train_loss = []
        for image, mask in tqdm.tqdm(data.DataLoader(dataset, batch_size=32, shuffle=True)):
            image = image.type(torch.float).to(device)
            y_pred = model(image)
            # s_mask = mask.squeeze()
            # s_y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, mask.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss = []
        for image, mask in data.DataLoader(dataset_val, batch_size=1, shuffle=False):
            image = image.to(device)
            y_pred = model(image)
            s_mask = mask.squeeze()
            s_y_pred = y_pred.squeeze()

            loss = loss_fn(s_y_pred, s_mask.to(device))
            val_loss.append(loss.item())

        prev_val_loss = np.mean(val_loss)
        results.append(prev_val_loss)
        print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))
        print(results.index(min(results)), len(results))
        if np.mean(val_loss) >= min(results):
            torch.save(model, 'torchmodel')


        if (results.index(min(results)) + patience) <= len(results):
            break
    model = torch.load('torchmodel')
    import glob

    test_path = os.path.join(directory, 'test')
    test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

    print(len(test_file_list))
    # test_file_list = test_file_list[:500]
    test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)

    all_predictions = []
    for image in tqdm.tqdm(data.DataLoader(test_dataset, batch_size=30)):
        image = image[0].type(torch.float).to(device)
        y_pred = model(image).cpu().detach().numpy()
        all_predictions.append(y_pred)
    all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]

    # all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
    all_predictions_stacked = all_predictions_stacked[:,0:101,0:101]
    test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)

    val_predictions = []
    val_masks = []
    for image, mask in tqdm.tqdm(data.DataLoader(dataset_val, batch_size=30)):
        image = image.type(torch.float).to(device)
        y_pred = model(image).cpu().detach().numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask)

    print(np.vstack(val_predictions).shape, np.vstack(val_masks).shape)
    val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]

    val_masks_stacked = np.vstack(val_masks)[:, :, :]
    val_predictions_stacked = val_predictions_stacked[:,0:101,0:101]

    val_masks_stacked = val_masks_stacked[:,0:101,0:101]

    from sklearn.metrics import jaccard_similarity_score

    metric_by_threshold = []
    for threshold in np.linspace(0, 1, 101):
        val_binary_prediction = (val_predictions_stacked > threshold).astype(int)

        iou_values = []
        for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
            iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
            iou_values.append(iou)
        iou_values = np.array(iou_values)

        accuracies = [
            np.mean(iou_values > iou_threshold)
            for iou_threshold in np.linspace(0.5, 0.95, 10)
        ]
        print('Threshold: {0}, Metric: {1}'.format(threshold, np.mean(accuracies)))
        metric_by_threshold.append((np.mean(accuracies), threshold))

    best_metric, best_threshold = max(metric_by_threshold)
    print(best_metric, best_threshold)
    threshold = best_threshold
    binary_prediction = (all_predictions_stacked > threshold).astype(int)

    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    submit.to_csv('submit_baseline2.csv', index=False)

if __name__ == '__main__':
    main()