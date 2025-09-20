import torch
import torchmetrics
from model import convATTnet,convLSTMnet
from logger import Logger
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logger = Logger()


def log(str, log_out):
    print(str)
    logger.set_filename(log_out)
    logger.log(str + '\n')


codeadict = {'A': "1", 'C': "2", 'D': "3", 'E': "4", 'F': "5", 'G': "6", 'H': "7", 'I': "8", 'K': "9", 'L': "10",
             'M': "11", 'N': "12", 'P': "13", 'Q': "14", 'R': "15", 'S': "16", 'T': "17", 'V': "18", 'W': "19", 'Y': "20"}


def format(predict_fasta):
    formatfasta = []
    recordlabel = []
    for record in SeqIO.parse(predict_fasta, 'fasta'):
        fastalist = []
        length = len(record.seq)
        if length <= 256:
            for i in range(1, 256-length+1):
                fastalist.append(0)
            for a in record.seq:
                fastalist.append(int(codeadict[a]))
        formatfasta.append(fastalist)
        if record.id.startswith('AMP'):
            recordlabel.append(1)
        else:
            recordlabel.append(0)

    inputarray = np.array(formatfasta)
    print(inputarray.shape)
    labelarray = np.array(recordlabel)
    return(inputarray, labelarray)



def test(args):
    x = torch.tensor(format(args.inputs)[0], dtype=torch.long)
    y = torch.tensor(format(args.inputs)[1], dtype=torch.float)
    test_ids = TensorDataset(x, y)
    test_loader = DataLoader(
        dataset=test_ids, batch_size=args.batch_size, shuffle=True)
    model = convLSTMnet()
    model.to(device)
    model.load_state_dict(torch.load('./model.pt'), strict=True)
    model.eval()
    accuracy = torchmetrics.Accuracy(task="binary").to(device)
    recall = torchmetrics.Recall(task="binary").to(device)
    precision = torchmetrics.Precision(task="binary").to(device)
    auroc = torchmetrics.AUROC(num_classes=None, task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)
    finaloutputs = torch.tensor([]).to(device)
    finallabels = torch.tensor([], dtype=torch.long).to(device)
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            labels = torch.as_tensor(labels, dtype=torch.long)
            # finaloutputs = torch.cat([finaloutputs, outputs], 0)
            # finallabels = torch.cat([finallabels, labels], 0)
		    #         labels = torch.as_tensor(labels, dtype=torch.long)
            finaloutputs = torch.cat([finaloutputs, outputs], 0)
            finallabels = torch.cat([finallabels, labels], 0)
    # print(f'valid:{finallabels}')
        finaloutputlist = []
        finallabelslist = []
        for i in finaloutputs.cpu().type(torch.float32).tolist():
            for a in i:
                finaloutputlist.append(float(a))
        for i in finallabels.cpu().type(torch.float32).tolist():
            for a in i:
                finallabelslist.append(float(a))
        # print(finaloutputlist, finallabelslist)
        fpr, tpr, thresholds = metrics.roc_curve(
            finallabelslist, finaloutputlist)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw,
                 label="ROC curve (area = %0.3f)" % roc_auc,)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        #plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.savefig('./auroc.png')
        plt.show()

        pre, reca, thresholds = metrics.precision_recall_curve(
            finallabelslist, finaloutputlist)
        auc_prc = metrics.auc(reca, pre)
        # fprtpr = np.vstack((fpr, tpr))
        # prerecall = np.vstack((pre, reca))
        # fprtpr = fprtpr.transpose()
        # prerecall = prerecall.transpose()
        # print(fprtpr.shape)
        plt.figure()
        lw = 2
        plt.plot(reca, pre, color="darkorange", lw=lw,
                 label="ROC curve (area = %0.3f)" % roc_auc,)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        #plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.savefig(
            'auprc.png')
        plt.show()
        accuracy(finaloutputs, finallabels)
        recall(finaloutputs, finallabels)
        precision(finaloutputs, finallabels)
        auroc(finaloutputs, finallabels)
        f1(finaloutputs, finallabels)
        accuracy_value = accuracy.compute()
        recall_value = recall.compute()
        precision_value = precision.compute()
        auroc_value = auroc.compute()
        f1_value = f1.compute()
        accuracy.reset()
        recall.reset()
        precision.reset()
        auroc.reset()
        f1.reset()
        log('Test Result: F1: ' + str("%.5f" % f1_value.item()) + '\tAccurcay: ' + str("%.5f" % accuracy_value.item()) + '\tPrecision: ' + str("%.5f" %
            precision_value.item()) + '\tRecall: ' + str("%.5f" % recall_value.item())+'\tAUROC: ' + str("%.5f" % auroc_value.item()), './test.log')
