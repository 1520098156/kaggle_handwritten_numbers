import csv

import numpy as np
import torch

submission_path = 'submission.csv'


def read_train_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/train.csv'))
    next(csv_reader)
    for row in csv_reader:
        label = int(row[0])
        pic_in_row = [float(x) for x in row[1:]]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        temp = (pic, label)
        data.append(temp)
    return data


def read_test_data():
    data = []
    csv_reader = csv.reader(open('digit-recognizer/test.csv'))
    next(csv_reader)
    for row in csv_reader:
        pic_in_row = [float(x) for x in row]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        data.append(pic)
    return data


def write_test_prediction(prediction):
    with open(submission_path, 'w', newline='') as file:
        csv_write = csv.writer(file)
        header = ['ImageId', 'Label']
        csv_write.writerow(header)
        for i in range(28000):
            row = [i + 1, prediction[i]]
            csv_write.writerow(row)
