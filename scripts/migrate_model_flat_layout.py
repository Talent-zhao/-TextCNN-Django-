# -*- coding: utf-8 -*-
"""
Move flat legacy files under model/ into model/<algo>/ subfolders.

Also rewrites model/model_training_registry.json keys to the new relative paths.

Usage (project root):
  python scripts/migrate_model_flat_layout.py
  python scripts/migrate_model_flat_layout.py --dry-run

Options:
  --dedup
      If destination exists:
        - same size: delete src
        - different size: move src into destination folder as legacy_<basename>
  --cleanup-root
      Move remaining files directly under model/ into model/legacy/.
      Keeps: num2name.json, model_training_registry.json, README.txt
"""
from __future__ import print_function

import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithm.model_training_registry import load_registry, registry_abs_path, save_registry


MOVES = [
    ('model/model_svm.pkl', 'model/svm/model_svm.pkl'),
    ('model/model_knn.pkl', 'model/knn/model_knn.pkl'),
    ('model/model_rf.pkl', 'model/rf/model_rf.pkl'),
    ('model/model_dt.pkl', 'model/dt/model_dt.pkl'),
    ('model/model_lr.pkl', 'model/lr/model_lr.pkl'),
    ('model/tfidfVectorizer_svm.pkl', 'model/svm/tfidfVectorizer_svm.pkl'),
    ('model/tfidfVectorizer_knn.pkl', 'model/knn/tfidfVectorizer_knn.pkl'),
    ('model/tfidfVectorizer_rf.pkl', 'model/rf/tfidfVectorizer_rf.pkl'),
    ('model/tfidfVectorizer_dt.pkl', 'model/dt/tfidfVectorizer_dt.pkl'),
    ('model/tfidfVectorizer_lr.pkl', 'model/lr/tfidfVectorizer_lr.pkl'),
    ('model/tfidfVectorizer.pkl', 'model/svm/tfidfVectorizer.pkl'),
    ('model/textcnn.pt', 'model/textcnn/textcnn.pt'),
    ('model/textcnn_vocab.json', 'model/textcnn/textcnn_vocab.json'),
    ('model/textrcnn.pt', 'model/textrcnn/textrcnn.pt'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的操作')
    parser.add_argument(
        '--dedup',
        action='store_true',
        help='当 dst 已存在时：同大小删除 src；不同大小改名为 legacy_... 再移动。',
    )
    parser.add_argument(
        '--cleanup-root',
        action='store_true',
        help='将 model/ 根目录的剩余文件移动到 model/legacy/（不影响预测所需的必需文件）。',
    )
    args = parser.parse_args()

    for rel_old, rel_new in MOVES:
        src = os.path.join(ROOT, *rel_old.split('/'))
        dst = os.path.join(ROOT, *rel_new.split('/'))
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(dst):
            if not args.dedup:
                print('跳过（目标已存在）:', rel_new, flush=True)
                continue
            src_sz = os.path.getsize(src)
            dst_sz = os.path.getsize(dst)
            if src_sz == dst_sz:
                if args.dry_run:
                    print('去重删除（同大小）:', rel_old, '-> skip src', flush=True)
                else:
                    os.remove(src)
                    print('已删除重复文件（同大小）:', rel_old, flush=True)
                continue

            legacy_name = 'legacy_' + os.path.basename(src)
            legacy_dst = os.path.join(os.path.dirname(dst), legacy_name)
            # 若 legacy_dst 已存在，就继续改名（避免覆盖）
            if os.path.isfile(legacy_dst):
                base, ext = os.path.splitext(legacy_name)
                i = 2
                while os.path.isfile(legacy_dst):
                    legacy_dst = os.path.join(os.path.dirname(dst), '{}_{}{}'.format(base, i, ext))
                    i += 1
            if args.dry_run:
                print('去重保留并移动（不同大小）:', rel_old, '->', os.path.relpath(legacy_dst, ROOT).replace("\\", "/"), flush=True)
            else:
                shutil.move(src, legacy_dst)
                print(
                    '已移动为 legacy（不同大小）:',
                    rel_old,
                    '->',
                    os.path.relpath(legacy_dst, ROOT).replace("\\", "/"),
                    flush=True,
                )
            continue
        if args.dry_run:
            print('将移动', rel_old, '->', rel_new, flush=True)
        else:
            shutil.move(src, dst)
            print('已移动', rel_old, '->', rel_new, flush=True)

    tcx = os.path.join(ROOT, 'model', 'textcnn', 'textcnn_vocab.json')
    trx = os.path.join(ROOT, 'model', 'textrcnn', 'textcnn_vocab.json')
    if os.path.isfile(tcx) and not os.path.isfile(trx):
        if args.dry_run:
            print('将复制', tcx, '->', trx, '（TextRCNN 默认词表）', flush=True)
        else:
            os.makedirs(os.path.dirname(trx), exist_ok=True)
            shutil.copy2(tcx, trx)
            print('已复制词表到 textrcnn/（TextRCNN 默认路径）', flush=True)

    reg = load_registry(ROOT)
    if not reg:
        print('注册表为空或不存在，结束。', flush=True)
        return

    new_reg = {}
    for k, meta in reg.items():
        nk = k
        for o, n in MOVES:
            if k.replace('\\', '/') == o:
                nk = n
                break
        if nk in new_reg and isinstance(new_reg[nk], dict) and isinstance(meta, dict):
            prev_tc = int(new_reg[nk].get('train_count') or 0)
            cur_tc = int(meta.get('train_count') or 0)
            merged = dict(meta)
            merged['train_count'] = max(prev_tc, cur_tc)
            new_reg[nk] = merged
        else:
            new_reg[nk] = meta

    if args.dry_run:
        print('[dry-run] 将重写', registry_abs_path(ROOT), flush=True)
        return

    save_registry(ROOT, new_reg)
    print('已更新注册表键名。', flush=True)

    if args.cleanup_root:
        legacy_root = os.path.join(ROOT, 'model', 'legacy')
        keep_names = {'model_training_registry.json', 'num2name.json', 'README.txt'}
        # 若 cleanup 触发，应该先确认必需子目录已存在（不会覆盖）
        os.makedirs(legacy_root, exist_ok=True)
        for name in os.listdir(os.path.join(ROOT, 'model')):
            if name in keep_names:
                continue
            p = os.path.join(ROOT, 'model', name)
            if not os.path.isfile(p):
                continue
            # 避免误操作：如果它正好是我们 MOVES 中的 dst（则无需移动）
            rel = 'model/' + name
            should_skip = False
            # rel 可能是 dst 的 basename，但 dst 在子目录里，所以这里只排除 root 层面的必需文件
            #（我们只移动 root 直系文件）
            _ = rel
            if should_skip:
                continue
            dst_name = name
            dst_path = os.path.join(legacy_root, dst_name)
            if os.path.isfile(dst_path):
                base, ext = os.path.splitext(dst_name)
                i = 2
                while os.path.isfile(dst_path):
                    dst_name = '{}_{}{}'.format(base, i, ext)
                    dst_path = os.path.join(legacy_root, dst_name)
                    i += 1
            if args.dry_run:
                print('cleanup-root:', rel, '->', os.path.relpath(dst_path, ROOT).replace("\\", "/"), flush=True)
            else:
                shutil.move(p, dst_path)
                print('cleanup-root 已移动:', rel, '->', os.path.relpath(dst_path, ROOT).replace("\\", "/"), flush=True)


if __name__ == '__main__':
    main()
