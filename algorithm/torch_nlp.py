# -*- coding: utf-8 -*-
import inspect
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

from algorithm.text_utils import load_num2name, text_to_char_ids


def _load_torch_checkpoint(path, map_location='cpu'):
    """
    加载本项目的 .pt 检查点。PyTorch 2.6+ 默认 weights_only=True，无法加载含元数据的训练快照，故显式 False。
    """
    if torch is None:
        raise RuntimeError('torch unavailable')
    kw = {'map_location': map_location}
    try:
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kw['weights_only'] = False
    except (TypeError, ValueError):
        pass
    return torch.load(path, **kw)


def _torch_predict_thread_limit():
    if torch is None:
        return
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _num_classes_from_num2name(num2name_path, cfg_fallback_key, cfg):
    """与训练时 num2name.json 类别数对齐，避免硬编码 4 类导致权重与结构不一致。"""
    m = load_num2name(num2name_path or '')
    if m:
        try:
            return len(m)
        except TypeError:
            pass
    return int(cfg.get(cfg_fallback_key, 4))


TextCNN = None
TextRCNN = None

if nn is not None:
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
        """与 train_char_torch 一致：无 Conv/RNN，Windows 训练常用。"""

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
        """
        字符级 TextRCNN：BiLSTM/BiGRU 上下文 + 词向量拼接；checkpoint 可含 textrcnn_rnn='gru'（Windows 训练常用）。
        """

        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0, dropout=0.5, rnn_type='lstm'):
            super(TextRCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
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

else:
    def TextCNN(*args, **kwargs):
        raise RuntimeError('torch unavailable')

    def CharBagMLP(*args, **kwargs):
        raise RuntimeError('torch unavailable')

    def TextRCNN(*args, **kwargs):
        raise RuntimeError('torch unavailable')


def predict_textcnn(text, weight_path, vocab_path, num2name_path, cfg):
    if not cfg.get('enable_textcnn', True):
        return {'enabled': False, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textcnn disabled'}
    if torch is None or nn is None:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'torch unavailable'}
    if not weight_path or not vocab_path or not os.path.exists(weight_path) or not os.path.exists(vocab_path):
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textcnn vocab/weight missing'}

    try:
        _torch_predict_thread_limit()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict):
            raise ValueError('invalid vocab format')

        max_len = int(cfg.get('textcnn_max_len', 128))
        embedding_dim = int(cfg.get('textcnn_embedding_dim', 128))
        num_classes = _num_classes_from_num2name(
            num2name_path, 'textcnn_num_classes', cfg
        )
        kernel_sizes = [int(i) for i in str(cfg.get('textcnn_kernel_sizes', '3,4,5')).split(',') if i.strip()]
        num_channels = int(cfg.get('textcnn_num_channels', 100))
        dropout = float(cfg.get('textcnn_dropout', 0.5))

        ckpt = _load_torch_checkpoint(weight_path, map_location='cpu')
        state_dict = ckpt.get('state_dict') if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        tb = ckpt.get('torch_backend') if isinstance(ckpt, dict) else None
        if tb == 'char_bag_mlp':
            emb = int(ckpt.get('char_bag_embed', embedding_dim))
            dr = float(ckpt.get('char_bag_dropout', dropout))
            model = CharBagMLP(
                vocab_size=max(vocab.values()) + 1,
                embed_dim=emb,
                num_classes=num_classes,
                dropout=dr,
            )
        else:
            model = TextCNN(
                vocab_size=max(vocab.values()) + 1,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                kernel_sizes=kernel_sizes,
                num_channels=num_channels,
                dropout=dropout,
            )
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        ids = text_to_char_ids(text, vocab, max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_prob = float(torch.max(probs).item())

        num2name = load_num2name(num2name_path or '')
        pred_name = num2name.get(str(pred_idx), str(pred_idx))
        return {'enabled': True, 'available': True, 'label': pred_name, 'prob': pred_prob, 'error': ''}
    except Exception as e:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': str(e)}


def predict_textrcnn(text, weight_path, vocab_path, num2name_path, cfg):
    if not cfg.get('enable_textrcnn', True):
        return {'enabled': False, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textrcnn disabled'}
    if torch is None or nn is None:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'torch unavailable'}
    if not weight_path or not vocab_path or not os.path.exists(weight_path) or not os.path.exists(vocab_path):
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textrcnn vocab/weight missing'}

    try:
        _torch_predict_thread_limit()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if not isinstance(vocab, dict):
            raise ValueError('invalid vocab format')

        max_len = int(cfg.get('textrcnn_max_len', cfg.get('textcnn_max_len', 128)))
        embed_dim = int(cfg.get('textrcnn_embedding_dim', cfg.get('textcnn_embedding_dim', 128)))
        hidden_dim = int(cfg.get('textrcnn_hidden_dim', 256))
        num_classes = _num_classes_from_num2name(
            num2name_path, 'textrcnn_num_classes', cfg
        )
        dropout = float(cfg.get('textrcnn_dropout', 0.5))

        ckpt = _load_torch_checkpoint(weight_path, map_location='cpu')
        state_dict = ckpt.get('state_dict') if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        backend = None
        if isinstance(ckpt, dict):
            backend = ckpt.get('textrcnn_backend')

        if backend == 'char_bag_mlp':
            emb = int(ckpt.get('char_bag_embed', embed_dim))
            dr = float(ckpt.get('char_bag_dropout', dropout))
            model = CharBagMLP(
                vocab_size=max(vocab.values()) + 1,
                embed_dim=emb,
                num_classes=num_classes,
                dropout=dr,
            )
            model.load_state_dict(state_dict, strict=False)
        elif backend == 'textcnn_compat':
            emb = int(ckpt.get('textcnn_embedding_dim', embed_dim))
            kernels = ckpt.get('textcnn_kernel_sizes') or [3, 4, 5]
            if isinstance(kernels, str):
                kernels = [int(x.strip()) for x in kernels.split(',') if x.strip()]
            nchan = int(ckpt.get('textcnn_num_channels', 100))
            dr = float(ckpt.get('textcnn_dropout', dropout))
            model = TextCNN(
                vocab_size=max(vocab.values()) + 1,
                embedding_dim=emb,
                num_classes=num_classes,
                kernel_sizes=kernels,
                num_channels=nchan,
                dropout=dr,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            hidden_dim = int(cfg.get('textrcnn_hidden_dim', 256))
            if isinstance(ckpt, dict) and ckpt.get('textrcnn_hidden_dim') is not None:
                hidden_dim = int(ckpt['textrcnn_hidden_dim'])
            rnn_type = 'lstm'
            if isinstance(ckpt, dict) and ckpt.get('textrcnn_rnn'):
                r = str(ckpt['textrcnn_rnn']).lower()
                if r not in ('none', ''):
                    rnn_type = r
            model = TextRCNN(
                vocab_size=max(vocab.values()) + 1,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                pad_idx=0,
                dropout=dropout,
                rnn_type=rnn_type,
            )
            model.load_state_dict(state_dict, strict=False)
        model.eval()

        ids = text_to_char_ids(text, vocab, max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_prob = float(torch.max(probs).item())
            pred_idx = int(torch.argmax(probs).item())

        num2name = load_num2name(num2name_path or '')
        pred_name = num2name.get(str(pred_idx), str(pred_idx))
        return {'enabled': True, 'available': True, 'label': pred_name, 'prob': pred_prob, 'error': ''}
    except Exception as e:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': str(e)}
