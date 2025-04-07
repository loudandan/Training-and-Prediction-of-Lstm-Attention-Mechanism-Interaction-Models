# import os
# import re
# import pandas as pd
# import numpy as np
# from Bio import SeqIO
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm
# import itertools

# # ================== 配置参数 ==================
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 32
# EPOCHS = 25
# PROTEIN_MAX_LEN = 512
# RNA_MAX_LEN = 2048
# MODEL_SAVE_PATH = "best_model.pth"

# # ================== 词汇表定义 ==================
# RNA_VOCAB = {
#     'A':1, 'C':2, 'G':3, 'U':4, 'T':5, 'R':6, 'Y':7, 'S':8, 'W':9, 'K':10,
#     'M':11, 'B':12, 'D':13, 'H':14, 'V':15, 'N':16, '-':17, '.':18,
#     'm':19, 's':20, 'X':21, '<MASK>':22, '<START>':23, '<END>':24, '<PAD>':0
# }

# PROTEIN_VOCAB = {
#     'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
#     'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19,
#     'V':20, 'U':21, 'O':22, 'X':23, '<MASK>':24, '<PAD>':0
# }

# # ================== 数据预处理 ==================
# def load_fasta(file_path):
#     """加载FASTA文件"""
#     return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}

# def encode_sequence(seq, vocab, max_len=None):
#     """序列编码函数"""
#     encoded = [vocab.get(c.upper(), 0) for c in seq]  # 处理大小写
#     if max_len:
#         encoded = encoded[:max_len] + [0]*(max_len - len(encoded))
#     return encoded

# def species_mask(sequence):
#     """物种特异性特征掩码"""
#     # 高S/C序列段标记
#     high_sc = list(re.finditer(r'[SC]{4,}', sequence.upper()))
#     # 低R区域标记
#     r_count = sequence.upper().count('R')
#     seq_len = max(len(sequence), 1)  # 防止除零
#     low_r = (r_count / seq_len) < 0.03
#     return {'sc_regions': high_sc, 'low_r': low_r}

# # ================== 数据集类 ==================
# class InteractionDataset(Dataset):
#     def __init__(self, df):
#         self.df = df
        
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         protein_seq = row['protein_seq']
#         rna_seq = row['rna_seq']
        
#         # 编码序列
#         protein = torch.tensor(encode_sequence(protein_seq, PROTEIN_VOCAB, PROTEIN_MAX_LEN))
#         rna = torch.tensor(encode_sequence(rna_seq, RNA_VOCAB, RNA_MAX_LEN))
        
#         # 生成物种特征
#         mask_info = species_mask(protein_seq)
#         # SC区域掩码
#         sc_mask = torch.zeros(PROTEIN_MAX_LEN, dtype=torch.float)
#         for match in mask_info['sc_regions']:
#             start = min(match.start(), PROTEIN_MAX_LEN-1)
#             end = min(match.end(), PROTEIN_MAX_LEN)
#             sc_mask[start:end] = 1.0
#         # 低R标记
#         low_r = torch.tensor(mask_info['low_r'], dtype=torch.float)
#         # 自适应权重计算
#         s_count = protein_seq.upper().count('S')
#         c_count = protein_seq.upper().count('C')
#         sc_ratio = (s_count + c_count) / max(len(protein_seq), 1)
#         sc_weight = 1 - sc_ratio / 10
        
#         return protein, rna, sc_mask, low_r, torch.tensor(sc_weight), torch.tensor(row['label'], dtype=torch.float)

# def collate_fn(batch):
#     """动态填充函数"""
#     proteins, rnas, sc_masks, low_rs, sc_weights, labels = zip(*batch)
    
#     # 填充序列
#     proteins_padded = pad_sequence(proteins, batch_first=True, padding_value=0)
#     rnas_padded = pad_sequence(rnas, batch_first=True, padding_value=0)
    
#     # 处理特征
#     sc_masks = torch.stack(sc_masks)
#     low_rs = torch.stack(low_rs)
#     sc_weights = torch.stack(sc_weights)
    
#     return (
#         proteins_padded.long(),
#         rnas_padded.long(),
#         sc_masks.float(),
#         low_rs.float(),
#         sc_weights.float(),
#         (proteins_padded != 0).float(),
#         (rnas_padded != 0).float(),
#         torch.stack(labels)
#     )

# # ================== 模型定义 ==================
# class PenalizedAttention(nn.Module):
#     """带惩罚项的注意力机制"""
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.penalty_heads = [2, 3]  # 对第3、4头施加惩罚
        
#     def forward(self, query, key, value, key_padding_mask=None):
#         attn_output, attn_weights = self.attn(
#             query, key, value,
#             key_padding_mask=key_padding_mask
#         )
        
#         # 计算惩罚项
#         penalty = 0
#         for head in self.penalty_heads:
#             if head < attn_weights.shape[0]:
#                 penalty += torch.mean(attn_weights[head]**2)
#         self.penalty = 0.1 * penalty  # 惩罚系数
        
#         return attn_output, attn_weights

# class InteractionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 蛋白质特征提取
#         self.protein_emb = nn.Embedding(len(PROTEIN_VOCAB), 64, padding_idx=0)
#         self.protein_lstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        
#         # RNA特征提取
#         self.rna_emb = nn.Embedding(len(RNA_VOCAB), 32, padding_idx=0)
#         self.rna_lstm = nn.LSTM(32, 64, bidirectional=True, batch_first=True)
        
#         # 注意力机制
#         self.attention = PenalizedAttention(embed_dim=128, num_heads=4)
        
#         # 特征融合层
#         self.fc = nn.Sequential(
#             nn.Linear(384 + 2, 128),  # 新增两个特征维度
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
        
#         # 分类器
#         self.classifier = nn.Linear(128, 1)

#     def forward(self, protein, rna, sc_mask, low_r, protein_mask, rna_mask):
#         # 特征提取
#         p_emb = self.protein_emb(protein)
#         p_out, _ = self.protein_lstm(p_emb)
        
#         r_emb = self.rna_emb(rna)
#         r_out, _ = self.rna_lstm(r_emb)
        
#         # 注意力机制
#         attn_output, _ = self.attention(
#             p_out.transpose(0, 1),
#             r_out.transpose(0, 1),
#             r_out.transpose(0, 1),
#             key_padding_mask=(rna_mask == 0).bool()
#         )
#         attn_output = attn_output.transpose(0, 1)
        
#         # 特征融合
#         basic_features = torch.cat([
#             p_out.mean(dim=1),
#             r_out.mean(dim=1),
#             attn_output.mean(dim=1)], 
#             dim=1
#         )
        
#         # 添加物种特征
#         sc_feature = sc_mask.mean(dim=1, keepdim=True)  # [bs, 1]
#         species_features = torch.cat([sc_feature, low_r.unsqueeze(1)], dim=1)
        
#         # 全连接层
#         combined = torch.cat([basic_features, species_features], dim=1)
#         fused = self.fc(combined)
        
#         return self.classifier(fused), self.attention.penalty

# # ================== 训练流程 ==================
# def evaluate(model, loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Evaluating"):
#             proteins, rnas, sc_masks, low_rs, sc_weights, p_mask, r_mask, labels = batch
#             proteins = proteins.to(DEVICE)
#             rnas = rnas.to(DEVICE)
#             labels = labels.to(DEVICE)
            
#             outputs, _ = model(proteins, rnas, sc_masks, low_rs, p_mask, r_mask)
#             preds = (torch.sigmoid(outputs.squeeze()) > 0.5).int().cpu().numpy()
            
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds)
    
#     report = classification_report(all_labels, all_preds, 
#                                  target_names=['Negative', 'Positive'],
#                                  output_dict=True)
#     pd.DataFrame(report).transpose().to_csv("evaluation_report.csv", float_format="%.4f")
#     return report

# def test(model, loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     results = []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Evaluating"):
#             proteins, rnas, sc_masks, low_rs, sc_weights, p_mask, r_mask, labels = batch
#             proteins = proteins.to(DEVICE)
#             rnas = rnas.to(DEVICE)
#             labels = labels.to(DEVICE)
            
#             outputs, _ = model(proteins, rnas, sc_masks, low_rs, p_mask, r_mask)
#             preds = (torch.sigmoid(outputs.squeeze()) > 0.5).int().cpu().numpy()
            
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds)

#             outputs = torch.sigmoid(outputs)
#             for i in range(len(outputs)):
#                 results.append((
#                     int(outputs[i] > 0.5),
#                     float(outputs[i])
#                 ))
#     # 保存结果时添加注释
#     result_df = pd.DataFrame(results, columns=[ "prediction", "probability"])
#     result_df.to_csv(
#         "test.tsv", 
#         sep='\t', 
#         index=False,
#         float_format="%.4f",
#         header=['#Prediction', 'Confidence']  # 添加注释说明
#     )
#     print("指定配对预测完成！结果保存至 val.tsv")

#     report = classification_report(all_labels, all_preds, 
#                                  target_names=['Negative', 'Positive'],
#                                  output_dict=True)
#     pd.DataFrame(report).transpose().to_csv("evaluation_report.csv", float_format="%.4f")
#     return report['accuracy']

# def main():
#     # 加载数据
#     protein_seqs = load_fasta("train_protein.fa")
#     rna_seqs = load_fasta("train_lncRNA.fa")
#     labels_df = pd.read_csv("train_labels.csv")

#     # 构建数据集
#     valid_pairs = []
#     for _, row in labels_df.iterrows():
#         if row['protein'] in protein_seqs and row['lncRNA'] in rna_seqs:
#             valid_pairs.append({
#                 'protein_seq': protein_seqs[row['protein']],
#                 'rna_seq': rna_seqs[row['lncRNA']],
#                 'label': row['label']
#             })
#     data = pd.DataFrame(valid_pairs)
#     train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'])

#     # 创建DataLoader
#     train_loader = DataLoader(InteractionDataset(train_df), batch_size=BATCH_SIZE, 
#                             collate_fn=collate_fn, shuffle=True)
#     test_loader = DataLoader(InteractionDataset(test_df), batch_size=BATCH_SIZE,
#                            collate_fn=collate_fn)

#     # 初始化模型
#     model = InteractionModel().to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     criterion = nn.BCEWithLogitsLoss(reduction='none')

#     best_acc = 0.0
#     for epoch in range(EPOCHS):
#         model.train()
#         progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
#         for batch in progress:
#             proteins, rnas, sc_masks, low_rs, sc_weights, p_mask, r_mask, labels = batch
#             # 数据转移
#             proteins = proteins.to(DEVICE)
#             rnas = rnas.to(DEVICE)
#             sc_masks = sc_masks.to(DEVICE)
#             low_rs = low_rs.to(DEVICE)
#             p_mask = p_mask.to(DEVICE)
#             r_mask = r_mask.to(DEVICE)
#             labels = labels.to(DEVICE)
#             sc_weights = sc_weights.to(DEVICE)
            
#             # 前向传播
#             optimizer.zero_grad()
#             outputs, penalty = model(proteins, rnas, sc_masks, low_rs, p_mask, r_mask)
            
#             # 损失计算
#             base_loss = criterion(outputs.squeeze(), labels)
#             weighted_loss = (base_loss * sc_weights).mean()
#             total_loss = weighted_loss + penalty
            
#             # 反向传播
#             total_loss.backward()
#             optimizer.step()
            
#             progress.set_postfix({'loss': f"{total_loss.item():.4f}"})
        
#         # 验证阶段
#         # val_acc = evaluate(model, test_loader)
#         report = evaluate(model, test_loader)
#         val_acc = report['accuracy']
        
#         # 保存最佳模型
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), MODEL_SAVE_PATH)
#             pd.DataFrame(report).transpose().to_csv("Bestt_evaluation_report.csv", float_format="%.4f")
#             print(f"New best model saved with acc: {best_acc:.4f}")

#     # ================== 预测部分 ==================
#     class FilteredPredictionDataset(Dataset):
#         def __init__(self, protein_seqs, rna_seqs, pair_df):
#             self.pairs = []
#             for _, row in pair_df.iterrows():
#                 pid = row['protein']
#                 rid = row['lncRNA']
#                 if pid in protein_seqs and rid in rna_seqs:
#                     self.pairs.append((pid, rid))
            
#             self.protein_seqs = protein_seqs
#             self.rna_seqs = rna_seqs
        
#         def __len__(self):
#             return len(self.pairs)
        
#         def __getitem__(self, idx):
#             pid, rid = self.pairs[idx]
#             return {
#                 'protein_id': pid,
#                 'rna_id': rid,
#                 'protein_seq': self.protein_seqs[pid],
#                 'rna_seq': self.rna_seqs[rid]
#             }

#     def predict_collate_fn(batch):
#         proteins = [torch.tensor(encode_sequence(x['protein_seq'], PROTEIN_VOCAB, PROTEIN_MAX_LEN)) for x in batch]
#         rnas = [torch.tensor(encode_sequence(x['rna_seq'], RNA_VOCAB, RNA_MAX_LEN)) for x in batch]
        
#         proteins_padded = pad_sequence(proteins, batch_first=True, padding_value=0)
#         rnas_padded = pad_sequence(rnas, batch_first=True, padding_value=0)
        
#         return {
#             'protein_ids': [x['protein_id'] for x in batch],
#             'rna_ids': [x['rna_id'] for x in batch],
#             'proteins': proteins_padded.long(),
#             'rnas': rnas_padded.long(),
#             'protein_mask': (proteins_padded != 0).float(),
#             'rna_mask': (rnas_padded != 0).float()
#         }

#     # 执行预测
#     model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
#     model.eval()

#     predict_protein = load_fasta("predict_pro_512.fa")
#     predict_rna = load_fasta("predict_lnc_2048.fa")
#     predict_pairs = pd.read_csv("predict_id.csv")

#     predict_dataset = FilteredPredictionDataset(
#         protein_seqs=predict_protein,
#         rna_seqs=predict_rna,
#         pair_df=predict_pairs
#     )

#     predict_loader = DataLoader(
#         predict_dataset,
#         batch_size=256,
#         collate_fn=predict_collate_fn,
#         shuffle=False
#     )

#     results = []
#     with torch.no_grad():
#         for batch in tqdm(predict_loader, desc="Predicting"):
#             proteins = batch['proteins'].to(DEVICE)
#             rnas = batch['rnas'].to(DEVICE)
#             p_mask = batch['protein_mask'].to(DEVICE)
#             r_mask = batch['rna_mask'].to(DEVICE)
            
#             # 生成临时特征（预测时无标签）
#             dummy_sc = torch.zeros(proteins.size(0), PROTEIN_MAX_LEN).to(DEVICE)
#             dummy_low_r = torch.zeros(proteins.size(0)).to(DEVICE)
            
#             outputs, _ = model(proteins, rnas, dummy_sc, dummy_low_r, p_mask, r_mask)
#             probs = torch.sigmoid(outputs.squeeze())
            
#             for i in range(len(probs)):
#                 results.append((
#                     batch['protein_ids'][i],
#                     batch['rna_ids'][i],
#                     int(probs[i] > 0.5),
#                     float(probs[i])
#                 ))

#     # 保存结果
#     result_df = pd.DataFrame(results, columns=["protein", "lncRNA", "prediction", "probability"])
#     result_df.to_csv(
#         "predictions.tsv", 
#         sep='\t', 
#         index=False,
#         float_format="%.4f",
#         header=['#Protein', 'lncRNA', 'Prediction', 'Confidence']
#     )
#     print("预测完成！结果保存至 predictions.tsv")

# if __name__ == "__main__":
#     main()




import os
import re
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import itertools
# 添加必要的库
from collections import OrderedDict

# ================== 配置参数 ==================
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
BATCH_SIZE = 32
EPOCHS = 25
PROTEIN_MAX_LEN = 512
RNA_MAX_LEN = 2048
MODEL_SAVE_PATH = "best_model.pth"

# ================== 数据预处理 ==================
# 自定义词汇表
RNA_VOCAB = {
    'A':1, 'C':2, 'G':3, 'U':4, 'T':5, 'R':6, 'Y':7, 'S':8, 'W':9, 'K':10,
    'M':11, 'B':12, 'D':13, 'H':14, 'V':15, 'N':16, '-':17, '.':18,
    'm':19, 's':20, 'X':21, '<MASK>':22, '<START>':23, '<END>':24, '<PAD>':0
}

PROTEIN_VOCAB = {
    'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
    'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19,
    'V':20, 'U':21, 'O':22, 'X':23, '<MASK>':24, '<PAD>':0
}

def species_mask(seq):
    """物种特异性特征掩码"""
    sc_mask = [0]*len(seq)
    # 检测连续4个及以上S/C的区域
    for match in re.finditer(r'[SC]{4,}', seq):
        for i in range(match.start(), match.end()):
            sc_mask[i] = 1
    # 检测低R含量
    r_content = seq.count('R') / len(seq) if len(seq) > 0 else 0
    low_r = 1 if r_content < 0.03 else 0
    return sc_mask, low_r

def load_fasta(file_path):
    """加载FASTA文件"""
    return {record.id: str(record.seq) for record in SeqIO.parse(file_path, "fasta")}

def encode_sequence(seq, vocab, max_len=None):
    """增强序列编码函数"""
    sc_mask, low_r = species_mask(seq)
    encoded = [vocab.get(c, 0) for c in seq]
    sc_encoded = sc_mask + [0]*(max_len - len(sc_mask)) if max_len else sc_mask
    
    if max_len:
        encoded = encoded[:max_len] + [0]*(max_len - len(encoded))
        sc_encoded = sc_encoded[:max_len]
    
    return encoded, sc_encoded, low_r

# ================== 数据加载 ==================
# 加载训练数据
protein_seqs = load_fasta("train_protein.fa")
rna_seqs = load_fasta("train_lncRNA.fa")
labels_df = pd.read_csv("train_labels.csv")

# 构建有效数据对
valid_pairs = []
for _, row in labels_df.iterrows():
    if row['protein'] in protein_seqs and row['lncRNA'] in rna_seqs:
        protein_seq = protein_seqs[row['protein']]
        rna_seq = rna_seqs[row['lncRNA']]
        valid_pairs.append({
            'protein_seq': protein_seq,
            'rna_seq': rna_seq,
            'label': row['label']
        })
data = pd.DataFrame(valid_pairs)
train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['label'])

# ================== 数据集类 ==================
class InteractionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 编码蛋白质序列和特征
        protein, sc_mask, low_r = encode_sequence(
            row['protein_seq'], PROTEIN_VOCAB, PROTEIN_MAX_LEN)
        
        # 编码RNA序列
        rna, _, _ = encode_sequence(row['rna_seq'], RNA_VOCAB, RNA_MAX_LEN)
        
        return (
            torch.tensor(protein),
            torch.tensor(sc_mask).float(),
            torch.tensor([low_r]).float(),
            torch.tensor(rna),
            torch.tensor(row['label'], dtype=torch.float)
        )

def collate_fn(batch):
    """动态填充函数"""
    proteins, sc_masks, low_rs, rnas, labels = zip(*batch)
    
    proteins_padded = pad_sequence(proteins, batch_first=True, padding_value=0)
    sc_masks_padded = pad_sequence(sc_masks, batch_first=True, padding_value=0)
    low_rs = torch.stack(low_rs)
    rnas_padded = pad_sequence(rnas, batch_first=True, padding_value=0)
    
    return (
        proteins_padded.long(),
        sc_masks_padded.float(),
        low_rs.float(),
        rnas_padded.long(),
        (proteins_padded != 0).float(),
        (rnas_padded != 0).float(),
        torch.stack(labels)
    )


# 创建DataLoader
train_loader = DataLoader(InteractionDataset(train_df), batch_size=BATCH_SIZE, 
                         collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(InteractionDataset(test_df), batch_size=BATCH_SIZE,
                        collate_fn=collate_fn)
# ================== 模型定义 ==================
class PenalizedAttention(nn.Module):
    """带L2惩罚的注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.sc_penalty = 0.0

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.attn(
            query, key, value, 
            key_padding_mask=key_padding_mask
        )
        # 对第3、4头施加L2惩罚
        if attn_weights is not None:
            self.sc_penalty = torch.sum(attn_weights[:, :, 2:4]**2).mean() * 0.1
        return attn_output

class InteractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 蛋白质特征提取
        self.protein_emb = nn.Embedding(len(PROTEIN_VOCAB), 64, padding_idx=0)
        self.sc_emb = nn.Linear(1, 16)  # 特征掩码嵌入
        self.lowr_emb = nn.Linear(1, 16)  # 低R特征嵌入
        
        self.protein_lstm = nn.LSTM(64+16+16, 64, bidirectional=True, batch_first=True)
        
        # RNA特征提取
        self.rna_emb = nn.Embedding(len(RNA_VOCAB), 32, padding_idx=0)
        self.rna_lstm = nn.LSTM(32, 64, bidirectional=True, batch_first=True)
        
        # 注意力机制
        self.attention = PenalizedAttention(embed_dim=128, num_heads=4)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, protein, sc_mask, low_r, rna, protein_mask, rna_mask):
        # 蛋白质特征融合
        p_emb = self.protein_emb(protein)
        sc_feat = self.sc_emb(sc_mask.unsqueeze(-1))
        lowr_feat = self.lowr_emb(low_r.unsqueeze(-1).expand(-1, p_emb.size(1), -1))
        
        p_combined = torch.cat([p_emb, sc_feat, lowr_feat], dim=-1)
        p_out, _ = self.protein_lstm(p_combined)
        
        # RNA特征
        r_emb = self.rna_emb(rna)
        r_out, _ = self.rna_lstm(r_emb)
        
        # 注意力
        attn_output = self.attention(
            p_out.transpose(0, 1),
            r_out.transpose(0, 1),
            r_out.transpose(0, 1),
            key_padding_mask=(rna_mask == 0)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # 特征融合
        combined = torch.cat([
            p_out.mean(dim=1),
            r_out.mean(dim=1),
            attn_output.mean(dim=1)], dim=1)
        
        return self.classifier(combined), self.attention.sc_penalty

# ================== 训练与评估 ==================
class AdaptiveLoss(nn.Module):
    """自适应损失函数"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, sc_content):
        base_loss = self.bce(inputs, targets)
        # 根据S/C含量调整权重
        weights = 1 - sc_content/10
        return (base_loss * weights).mean()

model = InteractionModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = AdaptiveLoss()

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            proteins, sc_mask, low_r, rnas, p_mask, r_mask, labels = batch
            batch = [t.to(DEVICE) for t in batch]
            
            outputs, penalty = model(*batch[:-1])

            
            sc_mask =sc_mask.to(DEVICE)
            # print(outputs.device)
            # print(batch[-1].device)
            # print(sc_mask.device)
            loss = criterion(outputs.squeeze(), batch[-1], sc_mask.mean(dim=1)) + penalty
            
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            all_labels.extend(batch[-1].cpu().numpy())
            all_preds.extend(preds.numpy())
            total_loss += loss.item()
    
    report = classification_report(all_labels, all_preds, 
                                 target_names=['Negative', 'Positive'],
                                 output_dict=True)
    return report, total_loss/len(loader)

def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    results = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            proteins, sc_mask, low_r, rnas, p_mask, r_mask, labels = batch
            batch = [t.to(DEVICE) for t in batch]
            
            outputs, penalty = model(*batch[:-1])
            sc_mask =sc_mask.to(DEVICE)
            loss = criterion(outputs.squeeze(), batch[-1], sc_mask.mean(dim=1)) + penalty
            
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            all_labels.extend(batch[-1].cpu().numpy())
            all_preds.extend(preds.numpy())
            total_loss += loss.item()
            outputs = torch.sigmoid(outputs)
            for i in range(len(outputs)):
                results.append((
                    int(outputs[i] > 0.5),
                    float(outputs[i])
                ))
    # 保存结果时添加注释
    result_df = pd.DataFrame(results, columns=[ "prediction", "probability"])
    result_df.to_csv(
        "test.tsv", 
        sep='\t', 
        index=False,
        float_format="%.4f",
        header=['#Prediction', 'Confidence']  # 添加注释说明
    )
    print("指定配对预测完成！结果保存至 val.tsv")
    report = classification_report(all_labels, all_preds, 
                                 target_names=['Negative', 'Positive'],
                                 output_dict=True)
    return report['accuracy']
best_acc = 0.0
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress:
        batch = [t.to(DEVICE) for t in batch]
        proteins, sc_mask, low_r, rnas, p_mask, r_mask, labels = batch
        
        optimizer.zero_grad()
        outputs, penalty = model(proteins, sc_mask, low_r, rnas, p_mask, r_mask)
        loss = criterion(outputs.squeeze(), labels, sc_mask.mean(dim=1)) + penalty
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # 验证阶段
    val_report, val_loss = evaluate(model, test_loader)
    val_acc = val_report['accuracy']
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        pd.DataFrame(val_report).transpose().to_csv("Bestt_evaluation_report.csv", float_format="%.4f")
        print(f"New best model saved with acc: {best_acc:.4f}")

# ================== 新增测试集部分 ==================
# 加载验证数据
val_protein_seqs = load_fasta("test_pro_512.fa")
val_rna_seqs = load_fasta("test_lnc_2048.fa")
val_labels_df = pd.read_csv("test_data_all.csv")

# 构建验证数据对
val_pairs = []
for _, row in val_labels_df.iterrows():
    if row['protein'] in val_protein_seqs and row['lncRNA'] in val_rna_seqs:
        val_pairs.append({
            'protein_seq': val_protein_seqs[row['protein']],
            'rna_seq': val_rna_seqs[row['lncRNA']],
            'label': row['label']
        })
val_data = pd.DataFrame(val_pairs)

# 创建验证集DataLoader
val_dataset = InteractionDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
# ================== 添加验证集评估 ==================
print("\n正在评估验证集...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
val_acc = test(model, val_loader)
print(f"验证集准确率: {val_acc:.4f}")

# ================== 预测部分 ==================
# （保持原有预测代码结构，增加特征处理）
# 注意：预测时需同样处理物种特征掩码

# 添加必要的库
from collections import OrderedDict
def predict_collate_fn(batch):
    """预测数据整理函数"""
    proteins = [torch.tensor(x['protein_encoded']) for x in batch]
    sc_masks = [torch.tensor(x['sc_mask']) for x in batch]
    low_rs = [torch.tensor([x['low_r']]) for x in batch]
    rnas = [torch.tensor(x['rna_encoded']) for x in batch]
    
    # 填充序列
    proteins_padded = pad_sequence(proteins, batch_first=True, padding_value=0)
    sc_masks_padded = pad_sequence(sc_masks, batch_first=True, padding_value=0)
    low_rs = torch.stack(low_rs)
    rnas_padded = pad_sequence(rnas, batch_first=True, padding_value=0)
    
    return {
        'protein_ids': [x['protein_id'] for x in batch],
        'rna_ids': [x['rna_id'] for x in batch],
        'proteins': proteins_padded.long(),
        'sc_masks': sc_masks_padded.float(),
        'low_rs': low_rs.float(),
        'rnas': rnas_padded.long(),
        'protein_mask': (proteins_padded != 0).float(),
        'rna_mask': (rnas_padded != 0).float()
    }

class CrossPredictionDataset(Dataset):
    """生成所有有效蛋白质和RNA的笛卡尔积数据集"""
    def __init__(self, protein_seqs, rna_seqs, protein_ids, rna_ids):
        self.pairs = list(itertools.product(protein_ids, rna_ids))
        self.protein_seqs = protein_seqs
        self.rna_seqs = rna_seqs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        protein_id, rna_id = self.pairs[idx]
        protein_seq = self.protein_seqs[protein_id]
        rna_seq = self.rna_seqs[rna_id]
        
        # 编码蛋白质序列并获取特征
        protein_encoded, sc_mask, low_r = encode_sequence(
            protein_seq, PROTEIN_VOCAB, PROTEIN_MAX_LEN)
        
        # 编码RNA序列
        rna_encoded, _, _ = encode_sequence(rna_seq, RNA_VOCAB, RNA_MAX_LEN)
        
        return {
            'protein_id': protein_id,
            'rna_id': rna_id,
            'protein_encoded': protein_encoded,
            'sc_mask': sc_mask,
            'low_r': low_r,
            'rna_encoded': rna_encoded
        }
# 加载训练好的模型

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# 加载预测数据
predict_protein = load_fasta("predict_pro_512.fa")
predict_rna = load_fasta("predict_lnc_2048.fa")
predict_pairs_df = pd.read_csv("predict_id.csv")

# 筛选有效序列
valid_proteins = [p for p in predict_pairs_df['protein'].unique() if p in predict_protein]
valid_lncRNAs = [r for r in predict_pairs_df['lncRNA'].unique() if r in predict_rna]

# 创建预测数据集
predict_dataset = CrossPredictionDataset(
    protein_seqs=predict_protein,
    rna_seqs=predict_rna,
    protein_ids=valid_proteins,
    rna_ids=valid_lncRNAs
)

# 创建数据加载器
predict_loader = DataLoader(
    predict_dataset,
    batch_size=512,
    collate_fn=predict_collate_fn,
    num_workers=8,
    pin_memory=True
)

# 执行预测
results = []
with torch.no_grad():
    for batch in tqdm(predict_loader, desc="交叉预测"):
        # 准备输入数据
        inputs = {
            'protein': batch['proteins'].to(DEVICE),
            'sc_mask': batch['sc_masks'].to(DEVICE),
            'low_r': batch['low_rs'].to(DEVICE),
            'rna': batch['rnas'].to(DEVICE),
            'protein_mask': batch['protein_mask'].to(DEVICE),
            'rna_mask': batch['rna_mask'].to(DEVICE)
        }
        
        # 模型推理
        outputs, _ = model(
            protein=inputs['protein'],
            sc_mask=inputs['sc_mask'],
            low_r=inputs['low_r'],
            rna=inputs['rna'],
            protein_mask=inputs['protein_mask'],
            rna_mask=inputs['rna_mask']
        )
        
        # 获取预测结果
        probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
        predictions = (probs > 0.5).astype(int)
        
        # 组装结果
        batch_results = zip(
            batch['protein_ids'],
            batch['rna_ids'],
            predictions.tolist(),
            probs.round(4).tolist()
        )
        results.extend(batch_results)

# 优化大文件保存
chunk_size = 1000000
for i in range(0, len(results), chunk_size):
    df_chunk = pd.DataFrame(
        results[i:i+chunk_size],
        columns=["protein", "lncRNA", "prediction", "probability"]
    )
    # 添加文件头注释
    header = [
        '# Protein-RNA Interaction Predictions',
        '# Model Version: 2.0',
        '# Features: Species-specific masking, Attention penalty'
    ]
    
    # 写入文件
    with open(f"predictions_part_{i//chunk_size}.tsv", 'w') as f:
        f.write('\n'.join(header) + '\n')
        df_chunk.to_csv(
            f,
            sep='\t',
            index=False,
            mode='a',
            header=True
        )

print("预测完成！结果分块保存至 predictions_part_*.tsv")