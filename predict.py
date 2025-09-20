from model import convATTnet, convLSTMnet
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os
import random

def main():
    argparser = argparse.ArgumentParser(
        description="NovaAMP model for predicting AMPs.")
    argparser.add_argument('-i', '--inputs', default='./',
                           type=str, help='input file')
    argparser.add_argument('-b', '--batch-size', default=1, type=int,
                           metavar='N')
    argparser.add_argument(
        '-o', '--output', default='AMP_predict.csv', type=str, help='output file')

    args = argparser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    predict(args)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')

codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20", 'X': "0", 'B': "0", 'U': "0"}
starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)


class MyDataset(Dataset):
    def __init__(self, sequence, labels):
        self._data = sequence
        self._label = labels

    def __getitem__(self, idx):
        sequence = self._data[idx]
        label = self._label[idx]
        return sequence, label

    def __len__(self):
        return len(self._data)


def format(predict_fasta):
    formatfasta = []
    recordid = []
    original_sequences = []  # 用于保存原始序列信息
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        length = len(record.seq)
        original_sequences.append(str(record.seq))  # 保存原始序列
        if length <= 256:
            fastalist = []
            for i in range(1, 256 - length + 1):
                fastalist.append(0)
            for a in record.seq:
                fastalist.append(int(codeadict[a]))
            formatfasta.append(fastalist)
            recordid.append(record.id)
        else:
            print(f"Warning: Sequence {record.id} is longer than 256 and will be skipped.")
    inputarray = np.array(formatfasta)
    idarray = np.array(recordid, dtype=object)
    return inputarray, idarray, original_sequences  # 返回原始序列信息


def predict(args):
    formatfasta = args.inputs  # 获取 formatfasta 数据   
    inputarray, proid, original_sequences = format(formatfasta)  # 获取原始序列信息
    profasta = torch.tensor(inputarray, dtype=torch.long)

    print(f"Processing {formatfasta}")
    print("Input tensor shape:", profasta.shape)

    data_ids = MyDataset(profasta, proid)
    data_loader = DataLoader(dataset=data_ids, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model = convLSTMnet()
    model.to(device)
    model.load_state_dict(torch.load('model.pt'), strict=True)
    print('warm up ...\n')


    pred_r = []  # 存储单个文件的预测结果

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, inputs_id = data
            inputs = inputs.to(device)
            outputs = model(inputs)

            if device == torch.device('cpu'):
                probability = outputs[0].item()
            else:
                probability = outputs.item()

            sequence_id = ''.join(inputs_id)
            original_sequence = original_sequences[i]  # 获取对应的原始序列

            if probability > 0.5:
                pred_r.append([sequence_id, probability, original_sequence, 'AMP'])
            else:
                pred_r.append([sequence_id, probability, original_sequence, 'non-AMP'])
    
    # generate outfile file
    df = pd.DataFrame(pred_r, columns=['protein', 'scores', 'sequence', 'predict result'])
    df.to_csv(args.output, sep='\t', header=True, index=True)


if __name__ == '__main__':
    main()
