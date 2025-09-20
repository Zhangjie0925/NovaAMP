# NovaAMP: A Novel Deep Learning Model for Antimicrobial Peptide Identification
## Introduction
NovaAMP (Novel Antimicrobial Peptide Identifier) is a deep learning model designed to accurately and efficiently identify antimicrobial peptides (AMPs) from protein sequences. By integrating convolutional neural networks (CNNs) with Long Short-Term Memory (LSTM) layers, this model effectively captures local patterns and long-range dependencies within peptide sequences, enabling high-throughput virtual screening to facilitate the discovery of novel antimicrobial peptides.


## Features
1. Hybrid Architecture: Utilizes CNN for local feature extraction and LSTM for sequence modeling
2. High Performance: Achieves excellent accuracy in distinguishing AMPs from non-AMPs
3. Easy to Use: Simple command-line interface for prediction tasks
4. FASTA Compatible: Directly processes standard FASTA format files
5. Comprehensive Output: Generates detailed prediction results with confidence scores


## Installation
1. Clone the repository:
```bash
git clone https://github.com/Zhangjie0925/NovaAMP.git
cd NovaAMP
```
2. Create and activate a virtual environment (recommended):
```bash
conda create -n novaamp
conda activate novaamp
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install biopython pandas numpy
```

## Quick Start
Run prediction on a folder containing FASTA files:
```bash
python predict.py -i ./input_folder -o ./output_folder
This will process all .fa files in the input folder and save results to the output folder.
```
## Usage
1. Prediction
The main prediction script processes FASTA files and generates prediction results:
```bash
python predict.py -i [input_folder] -o [output_folder] -b [batch_size]
```
Parameters:
-i, --inputs: Path to the input folder containing FASTA files (default: ./01_species)
-o, --output: Path to the output folder for results (default: ./01_species/01_results_0919)
-b, --batch-size: Batch size for prediction (default: 1)

2. Training
To train the model:
```bash
python main.py --mode train --inputs [data_directory] --epochs 100 --batch-size 72 --lr 0.0001
```
3. Testing
To test the model:
```bash
python main.py --mode test --inputs [data_directory] --batch-size 72
```
## Model Architecture
NovaAMP uses a hybrid CNN-LSTM architecture:
1. Input Processing: Sequences are one-hot encoded (21 amino acid classes)
2. Convolutional Layer: 1D convolution with 64 filters of size 16
3. Pooling: Max pooling with kernel size 5 and stride 5
4. LSTM Layer: Single-layer LSTM with 100 hidden units
5. Classification: Fully connected layer with sigmoid activation for binary classification
The model is implemented in PyTorch and defined in model.py.

## Data Format
### Input Format
1. FASTA format files with protein sequences
2. Maximum sequence length: 256 amino acids
3. Supported amino acids: ACDEFGHIKLMNPQRSTVWY
4. Non-standard amino acids are converted to padding tokens

Example Input:
```text
>protein1
MKLFVALVISLAAVSSSASAS
>protein2
MKQSTIALALLPLLFTPVTKA
```
### Output Format
1. Predictions are saved as tab-separated files with the following columns:
2. protein: Protein identifier from FASTA header
3. scores: Prediction confidence score (0-1)
4. sequence: Original amino acid sequence
5. predict result: Classification result (AMP or non-AMP)
6. description: Full description from FASTA header

Example Output:
```text
  protein scores  sequence  predict result
0 protein1  0.9994520545005798  MKLFVALVISLAAVSSSASAS AMP
1 protein2  0.9992108345031738  MKQSTIALALLPLLFTPVTKA AMP
```

## Acknowledgements
Thanks to the developers of PyTorch and BioPython for providing excellent deep learning and bioinformatics libraries.This work was inspired by previous research in computational antimicrobial peptide discovery.

## Contact
Project Link: https://github.com/Zhangjie0925/NovaAMP

## Note: This tool is intended for research purposes only. Predictions should be validated by experimental assays before clinical or therapeutic applications.
