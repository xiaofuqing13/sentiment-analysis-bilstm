# sentiment_analysis.py
"""
一、安装了必要的库,推荐python3.7以上
pip install torch torchvision torchaudio
pip install torchtext
pip install scikit-learn
pip install matplotlib seaborn

二、 数据集
Large Movie Review 数据集下载链接：https://ai.stanford.edu/~amaas/data/sentiment/
"""
import os
import re
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm

# 设置字体路径（黑体路径）
font_path = "C:/Windows/Fonts/simhei.ttf"  # SimHei（黑体）字体路径
font_prop = fm.FontProperties(fname=font_path)

# 全局修改默认字体
rcParams['font.family'] = font_prop.get_name()

# -----------------------------
# 一、数据加载与预处理
# -----------------------------

def load_data(data_dir):
    """
    加载数据集中的文本和标签。

    Args:
        data_dir (str): 数据集目录路径。

    Returns:
        texts (list): 文本列表。
        labels (list): 标签列表（1表示正面，0表示负面）。
    """
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
                    labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

def preprocess_text(text):
    """
    对文本进行预处理，包括转小写、去除标点和非字母字符、分词。

    Args:
        text (str): 原始文本。

    Returns:
        tokens (list): 预处理后的词汇列表。
    """
    # 转小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = text.split()
    return tokens

# -----------------------------
# 二、构建词汇表与编码
# -----------------------------

class Vocabulary:
    """
    词汇表类，用于构建词汇表并将词汇转换为数值编码。
    """
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, tokenized_texts):
        """
        根据预处理后的词汇列表构建词汇表。

        Args:
            tokenized_texts (list of list): 预处理后的词汇列表。
        """
        frequencies = Counter()
        for tokens in tokenized_texts:
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if word not in self.stoi:
                    self.stoi[word] = len(self.itos)
                    self.itos[len(self.itos)] = word

    def numericalize(self, tokens):
        """
        将词汇列表转换为数值编码。

        Args:
            tokens (list): 词汇列表。

        Returns:
            numericalized (list): 数值编码列表。
        """
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

# -----------------------------
# 三、创建数据集与数据加载器
# -----------------------------

class IMDBDataset(Dataset):
    """
    自定义数据集类，用于加载文本序列和对应的标签。
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    """
    自定义的批处理函数，用于填充序列和处理批次数据。

    Args:
        batch (list of tuples): 每个元素是 (sequence, label)。

    Returns:
        padded_sequences (Tensor): 填充后的序列张量。
        labels (Tensor): 标签张量。
        lengths (Tensor): 原始序列长度张量。
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded_sequences, labels, lengths

# -----------------------------
# 四、模型构建
# -----------------------------

class SentimentRNN(nn.Module):
    """
    循环神经网络模型，用于情绪分类任务。
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        out = self.fc(hidden)
        return out

# -----------------------------
# 五、训练与评估
# -----------------------------

def train(model, loader, optimizer, criterion, device):
    """
    训练模型一个epoch。

    Args:
        model (nn.Module): 模型。
        loader (DataLoader): 训练数据加载器。
        optimizer (Optimizer): 优化器。
        criterion (Loss): 损失函数。
        device (torch.device): 设备（CPU或GPU）。

    Returns:
        avg_loss (float): 平均损失。
        all_preds (list): 所有预测结果。
        all_labels (list): 所有真实标签。
    """
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for texts, labels, lengths in loader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    return avg_loss, all_preds, all_labels

def evaluate(model, loader, criterion, device):
    """
    评估模型在验证集或测试集上的表现。

    Args:
        model (nn.Module): 模型。
        loader (DataLoader): 数据加载器。
        criterion (Loss): 损失函数。
        device (torch.device): 设备（CPU或GPU）。

    Returns:
        avg_loss (float): 平均损失。
        all_preds (list): 所有预测结果。
        all_labels (list): 所有真实标签。
    """
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths).squeeze(1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = epoch_loss / len(loader)
    return avg_loss, all_preds, all_labels

# -----------------------------
# 六、主函数
# -----------------------------

def main():
    # 设置随机种子（可选）
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据集路径（请根据实际路径修改）
    train_dir = 'aclImdb/train'
    test_dir = 'aclImdb/test'

    # 加载训练和测试数据
    print("加载数据...")
    train_texts, train_labels = load_data(train_dir)
    test_texts, test_labels = load_data(test_dir)

    # 预处理文本
    print("预处理文本...")
    train_tokens = [preprocess_text(text) for text in train_texts]
    test_tokens = [preprocess_text(text) for text in test_texts]

    # 构建词汇表
    print("构建词汇表...")
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_tokens)
    print(f"词汇表大小: {len(vocab.stoi)}")

    # 数值化文本
    print("数值化文本...")
    train_sequences = [vocab.numericalize(tokens) for tokens in train_tokens]
    test_sequences = [vocab.numericalize(tokens) for tokens in test_tokens]

    # 划分训练集和验证集
    print("划分训练集和验证集...")
    train_seq, val_seq, train_lbl, val_lbl = train_test_split(train_sequences, train_labels, test_size=0.2, random_state=42)

    # 创建数据集
    print("创建数据集...")
    train_dataset = IMDBDataset(train_seq, train_lbl)
    val_dataset = IMDBDataset(val_seq, val_lbl)
    test_dataset = IMDBDataset(test_sequences, test_labels)

    # 创建数据加载器
    batch_size = 64
    print("创建数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 定义模型参数
    vocab_size = len(vocab.stoi)
    embed_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    bidirectional = True
    dropout = 0.5

    # 实例化模型
    print("构建模型...")
    model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    n_epochs = 10
    best_valid_loss = float('inf')

    # 训练过程
    print("开始训练...")
    for epoch in range(n_epochs):
        train_loss, train_preds, train_labels_epoch = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_preds, valid_labels_epoch = evaluate(model, val_loader, criterion, device)

        # 计算准确率
        train_preds_bin = np.round(train_preds)
        valid_preds_bin = np.round(valid_preds)
        train_acc = accuracy_score(train_labels_epoch, train_preds_bin)
        valid_acc = accuracy_score(valid_labels_epoch, valid_preds_bin)

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')

        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # 加载最佳模型
    print("加载最佳模型并评估测试集...")
    model.load_state_dict(torch.load('best-model.pt'))

    # 在测试集上评估
    test_loss, test_preds, test_labels_epoch = evaluate(model, test_loader, criterion, device)
    test_preds_bin = np.round(test_preds)
    test_acc = accuracy_score(test_labels_epoch, test_preds_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels_epoch, test_preds_bin, average='binary')
    auroc = roc_auc_score(test_labels_epoch, test_preds)

    print(f'\n测试集结果:')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print(f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f} | AUROC: {auroc:.3f}')

    # 绘制混淆矩阵
    cm = confusion_matrix(test_labels_epoch, test_preds_bin)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵图像
    plt.show()

if __name__ == "__main__":
    main()
