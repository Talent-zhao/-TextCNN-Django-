# -*- coding: utf-8 -*-
"""
字符级 TextCNN / TextRCNN 简易训练（CPU），输出与 algorithm.torch_nlp 推理格式兼容。

用法:
  python scripts/train_char_torch.py --csv datasets/depression_nlp/zh/oesd_keyword_binary/splits/train.csv --arch textcnn
  python scripts/train_char_torch.py --csv datasets/depression_nlp/en/depression_posts_binary/splits/train.csv --arch textrcnn

依赖: torch, pandas, sklearn（标签编码）
"""
from __future__ import print_function

import os
import sys

# 线程与 BLAS：须在任意 numpy/pandas/sklearn 导入之前设置。
# 关键：numpy/sklearn 会抢先链接 MKL/OpenMP，若先于 torch 导入，再 import torch 时两套运行时并存，
# Windows 上即使用纯 Embedding+Linear 也可能随机 0xC0000005。因此必须先 import torch，再 import numpy。
if sys.platform == 'win32':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse
import json
import random
import re
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError:
    print('ERROR: torch not installed. pip install torch', flush=True)
    sys.exit(1)

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    if sys.platform == 'win32':
        torch.backends.mkldnn.enabled = False
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from algorithm.model_paths import NUM2NAME_REL
from algorithm.model_training_registry import csv_arg_to_rel, record_training_session
from algorithm.training_summary_console import print_experiment_form_footer


def _experiment_sample_token_from_args(max_samples):
    try:
        n = int(max_samples)
    except (TypeError, ValueError):
        return 'full'
    if n <= 0:
        return 'full'
    mapping = {8000: '8000', 15000: '15000', 30000: '30000'}
    return mapping.get(n, 'full')


def build_vocab(texts, max_chars=5000):
    chars = set()
    for t in texts:
        for c in re.sub(r'\s+', '', str(t)):
            chars.add(c)
            if len(chars) >= max_chars:
                break
        if len(chars) >= max_chars:
            break
    # 保证顺序稳定
    clist = sorted(chars)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, c in enumerate(clist, start=2):
        vocab[c] = i
    return vocab


def encode(text, vocab, max_len):
    toks = [ch for ch in re.sub(r'\s+', '', str(text))]
    unk = vocab.get('<UNK>', 1)
    pad = vocab.get('<PAD>', 0)
    ids = [vocab.get(ch, unk) for ch in toks[:max_len]]
    while len(ids) < max_len:
        ids.append(pad)
    return ids


class CharDataset(Dataset):
    """
    惰性按条 encode，避免对大 CSV 预计算 self.x（3 万+ 条时内存暴涨易被 OOM 杀进程且无报错）。
    """

    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.y = np.asarray(labels, dtype=np.int64)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = encode(self.texts[i], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(int(self.y[i]), dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes, num_channels, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, (k, embedding_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class CharBagMLP(nn.Module):
    """
    字符 Embedding + 掩码均值池化 + 全连接，无 Conv2d / RNN。
    Windows 上部分 PyTorch+MKL 在 Conv2d 仍会 0xC0000005，用此结构训练最稳。
    """

    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.5, padding_idx=0):
        super(CharBagMLP, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.dropout(emb)
        mask = (x != self.padding_idx).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (emb * mask).sum(dim=1) / denom
        return self.fc(pooled)


class TextRCNN(nn.Module):
    """字符级 RCNN 变体；Windows 上 BiLSTM 易触发 0xC0000005，可改用 BiGRU（结构兼容，权重不互通）。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5, rnn_type='lstm'):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_type = (rnn_type or 'lstm').lower()
        half = hidden_dim // 2
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, half, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.LSTM(embed_dim, half, batch_first=True, bidirectional=True)
        self.rnn_type = rnn_type
        self.fc_cat = nn.Linear(embed_dim + hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        cat = torch.cat([out, emb], dim=-1)
        cat = torch.tanh(self.fc_cat(cat))
        cat = cat.transpose(1, 2)
        pooled = F.max_pool1d(cat, cat.size(2)).squeeze(2)
        pooled = self.dropout(pooled)
        return self.fc_out(pooled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--arch', required=True, choices=('textcnn', 'textrcnn'))
    parser.add_argument(
        '--out-dir',
        default='',
        help='默认 model/<arch>/（textcnn 或 textrcnn）；留空则按架构写入子目录',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=-1,
        help='训练轮数；-1 按样本量自动（大集减小轮数以控制耗时）',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='>0 时随机子采样该条数再训练（调试或极速试跑）',
    )
    parser.add_argument(
        '--registry-train-increment',
        type=int,
        default=1,
        help='成功写盘后注册表 train_count 累加步长；0 不增加但仍更新召回等指标',
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=150,
        help='每多少个 batch 打印一次进度（0 关闭）',
    )
    parser.add_argument(
        '--rcnn-hidden',
        type=int,
        default=0,
        help='TextRCNN RNN hidden 总维（双向各一半）；0=自动，Windows 默认 128，其它系统 256',
    )
    parser.add_argument(
        '--rcnn-rnn',
        choices=('auto', 'lstm', 'gru'),
        default='auto',
        help='非 Windows 上 TextRCNN 的循环层类型（Windows 固定用 CharBagMLP，忽略此项）',
    )
    args = parser.parse_args()
    if not (args.out_dir or '').strip():
        args.out_dir = os.path.join(ROOT, 'model', args.arch)

    if sys.platform == 'win32' and args.batch_size >= 32:
        args.batch_size = 16
        print('[0/3] win32 batch_size=16', flush=True)

    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    if 'text' not in df.columns or 'label' not in df.columns:
        print('ERROR: CSV must have columns: text, label', flush=True)
        sys.exit(1)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print('[0/3] subsampled to n=%d (--max-samples)' % len(df), flush=True)

    n_all = len(df)
    if args.epochs < 0:
        if n_all <= 8000:
            args.epochs = 8
        elif n_all <= 20000:
            args.epochs = 5
        elif n_all <= 40000:
            args.epochs = 3
        else:
            args.epochs = 2
        print('[0/3] auto epochs=%d n_samples=%d' % (args.epochs, n_all), flush=True)

    print('[1/3] rows=%d' % len(df), flush=True)
    texts = df['text'].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    num_classes = len(le.classes_)
    label_names = [str(x) for x in le.classes_]
    vocab = build_vocab(texts)
    vocab_size = max(vocab.values()) + 1

    ds = CharDataset(texts, y, vocab, args.max_len)
    rng = random.Random(42)
    order = list(range(len(ds)))
    rng.shuffle(order)
    n_val = max(1, int(0.15 * len(ds)))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print('[2/3] split train=%d val=%d (lazy CharDataset, low RAM)' % (len(train_idx), len(val_idx)), flush=True)

    print('[3a] build model vocab_size=%d num_classes=%d' % (vocab_size, num_classes), flush=True)
    textrcnn_hid = 256
    textrcnn_rnn = 'lstm'
    textrcnn_backend = None
    textcnn_torch_backend = None

    if args.arch == 'textcnn':
        out_w = os.path.join(args.out_dir, 'textcnn.pt')
        if sys.platform == 'win32':
            print('[3a] Windows: CharBagMLP (no Conv2d/RNN)', flush=True)
            model = CharBagMLP(vocab_size, args.embed, num_classes, dropout=0.5)
            textcnn_torch_backend = 'char_bag_mlp'
        else:
            model = TextCNN(
                vocab_size, args.embed, num_classes,
                kernel_sizes=[3, 4, 5], num_channels=100, dropout=0.5,
            )
            textcnn_torch_backend = 'textcnn_conv'
    else:
        out_w = os.path.join(args.out_dir, 'textrcnn.pt')
        if sys.platform == 'win32':
            print('[3a] TextRCNN Windows: CharBagMLP (no Conv2d/RNN, textrcnn.pt)', flush=True)
            model = CharBagMLP(vocab_size, args.embed, num_classes, dropout=0.5)
            textrcnn_backend = 'char_bag_mlp'
            textrcnn_rnn = 'none'
            textrcnn_hid = 0
        else:
            if args.rcnn_hidden > 0:
                textrcnn_hid = args.rcnn_hidden
            if args.rcnn_rnn == 'auto':
                textrcnn_rnn = 'lstm'
            else:
                textrcnn_rnn = args.rcnn_rnn
            print(
                '[3a] TextRCNN hidden_dim=%d rnn=%s' % (textrcnn_hid, textrcnn_rnn),
                flush=True,
            )
            model = TextRCNN(
                vocab_size,
                args.embed,
                textrcnn_hid,
                num_classes,
                dropout=0.5,
                rnn_type=textrcnn_rnn,
            )
            textrcnn_backend = 'rnn'

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')
    model = model.to(device)

    if sys.platform == 'win32':
        try:
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
        except TypeError:
            opt = torch.optim.SGD(model.parameters(), lr=min(args.lr * 5, 0.5), momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(
        '[3b] device=%s (import order: torch before numpy/pandas/sklearn)' % device,
        flush=True,
    )
    print(
        '[3/3] training epochs=%d batch_size=%d (first batches may be slow; progress every %d batches)'
        % (args.epochs, args.batch_size, args.log_every or 0),
        flush=True,
    )
    best_macro_recall = None
    last_train_loss = None
    rep_dict_last = None
    all_y_last, all_p_last = [], []
    val_proba_last = None
    train_acc_for_ofit = None
    val_acc_for_ofit = None
    t_train0 = time.perf_counter()
    for ep in range(args.epochs):
        print('  epoch %d/%d: train phase start' % (ep + 1, args.epochs), flush=True)
        model.train()
        total, n = 0.0, 0
        nb = 0
        for bx, by in train_dl:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            total += loss.item() * bx.size(0)
            n += bx.size(0)
            nb += 1
            if args.log_every and nb % args.log_every == 0:
                print(
                    '  epoch', ep + 1, 'batch', nb, 'running_loss=%.4f' % (total / max(n, 1)),
                    flush=True,
                )
        train_loss = total / max(n, 1)

        model.eval()
        all_y, all_p = [], []
        all_prob_rows = []
        with torch.no_grad():
            for bx, by in val_dl:
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                prob = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                all_y.extend(by.cpu().tolist())
                all_p.extend(pred.cpu().tolist())
                all_prob_rows.append(prob.cpu().numpy())
        val_acc = accuracy_score(all_y, all_p)
        print(
            '--- epoch %d/%d train_loss=%.4f val_acc=%.4f'
            % (ep + 1, args.epochs, train_loss, val_acc),
            flush=True,
        )
        rep_dict = classification_report(
            all_y,
            all_p,
            labels=list(range(num_classes)),
            target_names=label_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        mr = rep_dict.get('macro avg', {}).get('recall')
        try:
            mr = float(mr) if mr is not None else None
        except (TypeError, ValueError):
            mr = None
        if mr is not None:
            best_macro_recall = mr if best_macro_recall is None else max(best_macro_recall, mr)
        last_train_loss = train_loss
        rep = classification_report(
            all_y,
            all_p,
            labels=list(range(num_classes)),
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
        print('[val] classification report (precision/recall/f1):', flush=True)
        print(rep, flush=True)
        rep_dict_last = rep_dict
        all_y_last = list(all_y)
        all_p_last = list(all_p)
        if all_prob_rows:
            val_proba_last = np.concatenate(all_prob_rows, axis=0)

        if ep == args.epochs - 1:
            val_acc_for_ofit = float(val_acc)
            tc, tt = 0, 0
            model.eval()
            with torch.no_grad():
                for bx, by in train_dl:
                    bx = bx.to(device)
                    by = by.to(device)
                    logits = model(bx)
                    pred = logits.argmax(dim=1)
                    tc += int((pred == by).sum().item())
                    tt += int(by.size(0))
            train_acc_for_ofit = float(tc) / float(max(tt, 1))

    train_wall_sec = time.perf_counter() - t_train0

    print('[3/3] final full-data epoch then save ...', flush=True)
    model.train()
    full_dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=False
    )
    for bx, by in full_dl:
        bx = bx.to(device)
        by = by.to(device)
        opt.zero_grad()
        logits = model(bx)
        loss = loss_fn(logits, by)
        loss.backward()
        opt.step()

    ckpt_out = {'state_dict': model.cpu().state_dict(), 'arch': args.arch}
    if args.arch == 'textcnn':
        ckpt_out['torch_backend'] = textcnn_torch_backend
        if textcnn_torch_backend == 'char_bag_mlp':
            ckpt_out['char_bag_embed'] = args.embed
            ckpt_out['char_bag_dropout'] = 0.5
    if args.arch == 'textrcnn':
        ckpt_out['textrcnn_backend'] = textrcnn_backend
        ckpt_out['textrcnn_hidden_dim'] = textrcnn_hid
        ckpt_out['textrcnn_rnn'] = textrcnn_rnn
        if textrcnn_backend == 'char_bag_mlp':
            ckpt_out['char_bag_embed'] = args.embed
            ckpt_out['char_bag_dropout'] = 0.5
    torch.save(ckpt_out, out_w)
    vpath = os.path.join(args.out_dir, 'textcnn_vocab.json')
    with open(vpath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    num2name = {str(i): str(le.classes_[i]) for i in range(len(le.classes_))}
    os.makedirs(os.path.join(ROOT, 'model'), exist_ok=True)
    n2path = os.path.join(ROOT, *NUM2NAME_REL.split('/'))
    with open(n2path, 'w', encoding='utf-8') as f:
        json.dump(num2name, f, ensure_ascii=False, indent=2)

    rel_model = os.path.relpath(out_w, ROOT).replace('\\', '/')
    vocab_rel = os.path.relpath(vpath, ROOT).replace('\\', '/')
    n2_rel = NUM2NAME_REL
    record_training_session(
        ROOT,
        [rel_model, vocab_rel, n2_rel],
        csv_arg_to_rel(args.csv, ROOT),
        args.arch,
        best_macro_recall,
        last_train_loss,
        train_count_increment=getattr(args, 'registry_train_increment', 1),
    )

    footer_acc = footer_p = footer_r = footer_f1 = footer_auc = None
    if rep_dict_last is not None and all_y_last:
        try:
            from app.training_evaluation import (
                infer_overfitting_from_train_val_acc,
                metrics_from_classification_report_dict,
                roc_auc_from_proba,
                save_evaluation_from_training,
            )

            acc, p, r, f1v = metrics_from_classification_report_dict(rep_dict_last, all_y_last, all_p_last)
            auc_val = roc_auc_from_proba(all_y_last, val_proba_last) if val_proba_last is not None else None
            footer_acc, footer_p, footer_r, footer_f1, footer_auc = acc, p, r, f1v, auc_val
            if auc_val is not None:
                print('[eval] 验证集 ROC-AUC (OvR macro 或二分类): {:.4f}'.format(auc_val), flush=True)
            ofit = None
            if train_acc_for_ofit is not None and val_acc_for_ofit is not None:
                ofit = infer_overfitting_from_train_val_acc(train_acc_for_ofit, val_acc_for_ofit)
                if ofit is not None:
                    print(
                        '[eval] 训练准确率 {:.4f} vs 验证 {:.4f}，过拟合(训练-验证>6%): {}'.format(
                            train_acc_for_ofit,
                            val_acc_for_ofit,
                            '是' if ofit else '否',
                        ),
                        flush=True,
                    )
            csv_rel_ex = csv_arg_to_rel(args.csv, ROOT)
            hp_ex = {
                'arch': args.arch,
                'epochs': int(args.epochs),
                'batch_size': int(args.batch_size),
                'max_len': int(args.max_len),
                'embed': int(args.embed),
                'lr': float(args.lr),
                'max_samples': int(args.max_samples) if getattr(args, 'max_samples', 0) else None,
                'model_path': rel_model,
                'vocab_path': vocab_rel,
            }
            if args.arch == 'textcnn' and textcnn_torch_backend:
                hp_ex['torch_backend'] = textcnn_torch_backend
            if args.arch == 'textrcnn':
                if textrcnn_backend:
                    hp_ex['textrcnn_backend'] = textrcnn_backend
                if textrcnn_hid:
                    hp_ex['textrcnn_hidden_dim'] = int(textrcnn_hid)
                hp_ex['textrcnn_rnn'] = str(textrcnn_rnn)
            save_evaluation_from_training(
                args.arch,
                acc,
                p,
                r,
                f1v,
                auc=auc_val,
                training_time_sec=train_wall_sec,
                is_overfitting=ofit,
                file_path=rel_model,
                vectorizer_path=vocab_rel,
                experiment_csv_rel=csv_rel_ex,
                experiment_train_sample_scale=_experiment_sample_token_from_args(args.max_samples),
                experiment_hyperparams_extra=hp_ex,
            )
        except Exception as ex:
            print('[eval_persist] 自动保存评估异常（可忽略）:', ex, flush=True)

    print('OK:', args.arch, flush=True)
    print('  weights:', out_w, flush=True)
    print('  vocab:', vpath, flush=True)
    print('  num2name updated', flush=True)

    hp_footer = {
        'arch': args.arch,
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'max_len': int(args.max_len),
        'embed': int(args.embed),
        'lr': float(args.lr),
        'max_samples': int(args.max_samples) if getattr(args, 'max_samples', 0) else None,
        'csv': os.path.basename(str(args.csv)),
    }
    if args.arch == 'textcnn' and textcnn_torch_backend:
        hp_footer['torch_backend'] = textcnn_torch_backend
    if args.arch == 'textrcnn':
        if textrcnn_backend:
            hp_footer['textrcnn_backend'] = textrcnn_backend
        if textrcnn_hid:
            hp_footer['textrcnn_hidden_dim'] = int(textrcnn_hid)
        hp_footer['textrcnn_rnn'] = str(textrcnn_rnn)
    print_experiment_form_footer(
        hp_footer,
        train_wall_sec,
        footer_auc,
        footer_acc,
        footer_p,
        footer_r,
        footer_f1,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print('ERROR:', repr(e), flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
