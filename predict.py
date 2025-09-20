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
        description="DeepAlgPro Network for predicting AMPs.")
    argparser.add_argument('-i', '--inputs', default='./01_species',
                           type=str, help='input folder')
    argparser.add_argument('-b', '--batch-size', default=1, type=int,
                           metavar='N')
    argparser.add_argument(
        '-o', '--output', default='./01_species/01_results_0919',
        type=str, help='output folder')  # 输出文件夹路径

    args = argparser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    predict(args)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')

codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20", 'X': "0", 'B': "0", 'U': "0", '*': "0"}
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
    original_descriptions = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        length = len(record.seq)
        original_sequences.append(str(record.seq))  # 保存原始序列
        original_descriptions.append(str(record.description))
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
    return inputarray, idarray, original_sequences,original_descriptions  # 返回原始序列信息

def get_all_fasta_files(input_folder):
    fasta_files = []
    for root, dirs, files in os.walk(input_folder):  # 遍历整个目录
        for file in files:
            if file.endswith(".fa"):  # 只选择 .fa 文件
                fasta_files.append(os.path.join(root, file))
    return fasta_files

def predict(args):
    fasta_files = get_all_fasta_files(args.inputs)  # 获取所有 .fa 文件路径

    # 加载模型
    model = convLSTMnet()
    model.to(device)
    model.load_state_dict(torch.load('model.pt'), strict=True)
    print('warm up ...\n')

    for fasta_file in fasta_files:
        inputarray, proid, original_sequences,original_descriptions = format(fasta_file)  # 获取原始序列信息
        profasta = torch.tensor(inputarray, dtype=torch.long)

        print(f"Processing {fasta_file}")
        print("Input tensor shape:", profasta.shape)

        data_ids = MyDataset(profasta, proid)
        data_loader = DataLoader(dataset=data_ids, batch_size=args.batch_size, shuffle=False)

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
                original_description = original_descriptions[i]

                if probability > 0.5:
                    pred_r.append([sequence_id, probability, original_sequence, 'AMP',original_description])
                else:
                    pred_r.append([sequence_id, probability, original_sequence, 'non-AMP',original_description ])

        # 生成输出文件名
        file_name = os.path.basename(fasta_file)          # 获取带扩展名的文件名（如：example.fa）
        file_base = os.path.splitext(file_name)[0]        # 去掉扩展名（如：example）
        output_file = os.path.join(args.output, f"{file_base}_predict.xlsx")
        
        # 生成 DataFrame 并保存为 .xlsx 文件
        df = pd.DataFrame(pred_r, columns=['protein', 'scores', 'sequence', 'predict result','description'])
        df.to_csv(output_file, sep='\t', header=True, index=True)
        print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
