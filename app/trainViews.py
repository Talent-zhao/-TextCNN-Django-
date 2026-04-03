# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import threading

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.utils.http import url_has_allowed_host_and_scheme
from django.views import View

from app.userViews import check_admin_access, get_admin_panel_user

# 训练数据目录（相对项目根）：depression_nlp/en 英文，depression_nlp/zh 中文
REL_SEG_DATASETS_DEPRESSION_NLP = ('datasets', 'depression_nlp')
REL_SEG_DEFAULT_TRAIN_CSV = REL_SEG_DATASETS_DEPRESSION_NLP + (
    'zh',
    'oesd_keyword_binary',
    'splits',
    'train.csv',
)
# 相对项目根的训练数据默认路径（训练页下拉默认选中此项，若存在）
DEFAULT_TRAIN_CSV = os.path.join(*REL_SEG_DEFAULT_TRAIN_CSV)

SKLEARN_ALGOS = ('svm', 'knn', 'rf', 'dt', 'lr')
TORCH_ALGOS = ('textcnn', 'textrcnn')

# 网页训练可选参数白名单（仅允许下列 token，禁止任意 CLI 注入）
TRAIN_SAMPLE_SCALE = {'full': 0, '8000': 8000, '15000': 15000, '30000': 30000}
TRAIN_SKLEARN_MAX_FEATURES = {'4000': 4000, '8000': 8000, '12000': 12000}
TRAIN_TORCH_BATCH = {'auto': None, '8': 8, '16': 16}
# 深度学习训练轮数：-1 自动；否则 1～64（由页面数字框传入，服务端强校验）
TORCH_EPOCHS_MIN = -1
TORCH_EPOCHS_MAX = 64

TRAIN_HUB_SAMPLE_CHOICES = (
    ('full', '全量（数据多则更慢，通常效果更好）'),
    ('8000', '随机子采样 8,000 条（快速试跑）'),
    ('15000', '随机子采样 15,000 条'),
    ('30000', '随机子采样 30,000 条'),
)
TRAIN_HUB_SKLEARN_MF_CHOICES = (
    ('4000', '4,000（词表较小，省内存）'),
    ('8000', '8,000（默认，平衡）'),
    ('12000', '12,000（词表更大，更慢）'),
)
TRAIN_HUB_TORCH_BATCH_CHOICES = (
    ('auto', '自动（脚本默认；Windows 可能降为 16）'),
    ('8', 'batch 8（更稳，略慢）'),
    ('16', 'batch 16'),
)

# Windows 子进程异常退出：3221225477 = 0xC0000005 STATUS_ACCESS_VIOLATION（非超时）
_WIN_ACCESS_VIOLATION_UNSIGNED = 3221225477
_WIN_ACCESS_VIOLATION_SIGNED = -1073741819


def _train_windows_crash_hint(code):
    if sys.platform != 'win32':
        return None
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c == _WIN_ACCESS_VIOLATION_UNSIGNED or c == _WIN_ACCESS_VIOLATION_SIGNED:
        return (
            '[TRAIN_HINT] 0xC0000005: train script must import torch BEFORE numpy (see train_char_torch.py top). '
            'Also try: pip install -U torch --index-url https://download.pytorch.org/whl/cpu\n'
        )
    return None


def _train_mirror_console(line):
    """将训练子进程日志原样写到 Django 进程的控制台（runserver 终端），便于失败/超时时排查。"""
    if not line:
        return
    try:
        sys.stderr.write(line if line.endswith('\n') else line + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


def _is_under_project(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    return path == root or path.startswith(root + os.sep)


def _datasets_depression_nlp_root():
    return os.path.join(settings.BASE_DIR, 'datasets', 'depression_nlp')


def _is_holdout_split_csv(rel_path):
    """
    是否为验证/测试划分文件：训练页下拉不展示，与 train 划分分开。
    规则：文件名为 val/test/validation/dev.csv（大小写不敏感），或路径中任一级目录名为
    val、test、validation、dev（避免误选整夹为测试集的数据）。
    """
    rel = (rel_path or '').replace('\\', '/').strip().lower()
    if not rel:
        return False
    parts = [p for p in rel.split('/') if p]
    if not parts:
        return False
    base = parts[-1]
    if base in ('val.csv', 'test.csv', 'validation.csv', 'dev.csv'):
        return True
    for p in parts[:-1]:
        if p in ('val', 'test', 'validation', 'dev'):
            return True
    return False


def _csv_train_list_sort_key(rel):
    """下拉列表排序：默认中文训练集最前，其次 zh/、en/splits/train、en 其它。"""
    rel_norm = rel.replace('\\', '/')
    default_norm = DEFAULT_TRAIN_CSV.replace('\\', '/')
    rlow = rel_norm.lower()
    dlow = default_norm.lower()
    if rlow == dlow:
        group = 0
    elif '/zh/' in rlow:
        group = 1
    elif '/en/' in rlow and '/splits/' in rlow and rlow.endswith('train.csv'):
        group = 2
    elif '/en/' in rlow:
        group = 3
    else:
        group = 4
    return (group, rel_norm.count('/'), rlow)


def _list_datasets_depression_nlp_csvs():
    """
    递归列出 datasets/depression_nlp 下（含 en/、zh/）可用于训练的 .csv，
    排除 val/test 等划分文件。
    """
    base = settings.BASE_DIR
    root = _datasets_depression_nlp_root()
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            if not fn.lower().endswith('.csv'):
                continue
            full = os.path.join(dirpath, fn)
            if not os.path.isfile(full):
                continue
            if not _is_under_project(full, base):
                continue
            rel = os.path.relpath(full, base).replace(os.sep, '/')
            if _is_holdout_split_csv(rel):
                continue
            out.append(rel)
    return sorted(out, key=_csv_train_list_sort_key)


def _locale_token_for_csv_rel(rel):
    """训练页数据区域：zh、en，或未放在 zh/en 子目录下的 other。"""
    r = (rel or '').replace('\\', '/').lower()
    if '/zh/' in r:
        return 'zh'
    if '/en/' in r:
        return 'en'
    return 'other'


def _infer_default_train_data_locale(default_csv, csv_choices):
    """
    「数据语言」下拉初始值：与 default_csv 所在区域一致；
    否则优先有 zh 时选 zh，其次 en，否则全部。
    """
    t = _locale_token_for_csv_rel(default_csv)
    if t in ('zh', 'en'):
        return t
    choices = csv_choices or []
    if any(_locale_token_for_csv_rel(c) == 'zh' for c in choices):
        return 'zh'
    if any(_locale_token_for_csv_rel(c) == 'en' for c in choices):
        return 'en'
    return 'all'


def _open_local_folder(path):
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


def _parse_torch_epochs_value(raw):
    """训练轮数：-1 表示脚本自动；1～64 为固定轮数。兼容旧字段字符串 auto。"""
    if raw is None or raw == '':
        return -1
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == 'auto' or s == '':
            return -1
        try:
            raw = int(s, 10)
        except ValueError:
            raise ValueError('训练轮数须为整数')
    try:
        ep = int(raw)
    except (TypeError, ValueError):
        raise ValueError('训练轮数须为整数')
    if ep == TORCH_EPOCHS_MIN:
        return -1
    if ep == 0:
        raise ValueError('训练轮数不能为 0，请填 -1（自动）或 1～{}'.format(TORCH_EPOCHS_MAX))
    if ep < TORCH_EPOCHS_MIN or ep > TORCH_EPOCHS_MAX:
        raise ValueError('训练轮数须在 -1（自动）或 1～{} 之间'.format(TORCH_EPOCHS_MAX))
    return ep


def _training_extra_args_from_body(body, algo):
    """
    从 JSON body 解析训练附加参数，仅接受白名单 token。
    返回 (sklearn_extra_list | None, torch_extra_list | None)；二者择一使用。
    """
    body = body or {}

    def _token_scale(key, default='full'):
        t = (body.get(key) or default)
        if not isinstance(t, str):
            t = str(t)
        t = t.strip().lower()
        if t not in TRAIN_SAMPLE_SCALE:
            raise ValueError('无效的样本规模选项')
        return TRAIN_SAMPLE_SCALE[t]

    if algo in SKLEARN_ALGOS:
        ms = _token_scale('train_sample_scale', 'full')
        kf = body.get('sklearn_max_features') or '8000'
        if not isinstance(kf, str):
            kf = str(kf)
        kf = kf.strip().lower()
        if kf not in TRAIN_SKLEARN_MAX_FEATURES:
            raise ValueError('无效的 TF-IDF 特征上限选项')
        out = ['--max-features', str(TRAIN_SKLEARN_MAX_FEATURES[kf])]
        if ms > 0:
            out.extend(['--max-samples', str(ms)])
        return out, None

    if algo in TORCH_ALGOS:
        ms = _token_scale('train_sample_scale', 'full')
        ep = _parse_torch_epochs_value(body.get('torch_epochs'))
        bs_tok = body.get('torch_batch') or 'auto'
        if not isinstance(bs_tok, str):
            bs_tok = str(bs_tok)
        bs_tok = bs_tok.strip().lower()
        if bs_tok not in TRAIN_TORCH_BATCH:
            raise ValueError('无效的批次大小选项')
        out = ['--epochs', str(ep)]
        bs = TRAIN_TORCH_BATCH[bs_tok]
        if bs is not None:
            out.extend(['--batch-size', str(bs)])
        if ms > 0:
            out = ['--max-samples', str(ms)] + out
        return None, out

    return None, None


def _safe_csv_path(rel_or_abs):
    """限制在项目根目录内。"""
    root = settings.BASE_DIR
    if not rel_or_abs or not str(rel_or_abs).strip():
        rel_or_abs = DEFAULT_TRAIN_CSV
    raw = str(rel_or_abs).strip()
    if os.path.isabs(raw):
        path = os.path.abspath(raw)
    else:
        path = os.path.abspath(os.path.join(root, raw))
    if not _is_under_project(path, root):
        raise ValueError('训练文件必须在项目目录内')
    if not os.path.isfile(path):
        raise ValueError('文件不存在: {}'.format(path))
    if not path.lower().endswith('.csv'):
        raise ValueError('仅支持 CSV 文件')
    return path


def _training_command(script_name, extra_args):
    root = settings.BASE_DIR
    script = os.path.join(root, 'scripts', script_name)
    if not os.path.isfile(script):
        return None, '脚本缺失: {}'.format(script)
    cmd = [sys.executable, script] + extra_args
    return cmd, None


def _stream_training_lines(cmd, root, timeout_sec):
    """逐行产出子进程合并后的 stdout/stderr，便于页面实时刷新；同时镜像到本进程 stderr（控制台）。"""
    _train_mirror_console('[train] cwd={}'.format(root))
    _train_mirror_console('[train] cmd={}'.format(' '.join(cmd)))
    _train_mirror_console('[train] timeout_sec={}'.format(timeout_sec))

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # Windows 控制台默认非 UTF-8，子进程中文日志会乱码；强制 UTF-8 与 Django 流式解码一致
    env['PYTHONIOENCODING'] = 'utf-8'
    env.setdefault('PYTHONUTF8', '1')
    if sys.platform == 'win32':
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')
        env.setdefault('OPENBLAS_NUM_THREADS', '1')
        env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        env.setdefault('CUDA_VISIBLE_DEVICES', '')
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
    except Exception as e:
        err_line = '[TRAIN_ERROR]{}\n'.format(str(e))
        _train_mirror_console(err_line)
        yield err_line
        ex_line = '[TRAIN_EXIT]1\n'
        _train_mirror_console(ex_line)
        yield ex_line
        return

    timer = None
    timed_out = [False]

    def _timeout_kill():
        if proc.poll() is None:
            timed_out[0] = True
            _train_mirror_console('[train] TIMEOUT: terminating child pid={}'.format(proc.pid))
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()

    if timeout_sec and timeout_sec > 0:
        timer = threading.Timer(timeout_sec, _timeout_kill)
        timer.start()
    try:
        while True:
            line = proc.stdout.readline()
            if line == '' and proc.poll() is not None:
                break
            if line:
                _train_mirror_console(line)
                yield line
        # 读取管道中剩余内容（极少数情况下 readline 后仍有残留）
        tail = proc.stdout.read()
        if tail:
            _train_mirror_console(tail)
            yield tail
        code = proc.wait() if proc.returncode is None else proc.returncode
        if code is None:
            code = 0
        if timed_out[0]:
            tline = '[TRAIN_TIMEOUT]\n'
            _train_mirror_console(tline)
            yield tline
            code = 1
        hint = _train_windows_crash_hint(code)
        if hint:
            _train_mirror_console(hint)
            yield hint
        ex_line = '[TRAIN_EXIT]{}\n'.format(int(code))
        _train_mirror_console(ex_line)
        yield ex_line
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        err_line = '[TRAIN_ERROR]{}\n'.format(str(e))
        _train_mirror_console(err_line)
        yield err_line
        ex_line = '[TRAIN_EXIT]1\n'
        _train_mirror_console(ex_line)
        yield ex_line
    finally:
        if timer is not None:
            timer.cancel()


@method_decorator(check_admin_access, name='dispatch')
class ModelTrainingHubView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        default_csv = DEFAULT_TRAIN_CSV.replace('\\', '/')
        exists = os.path.isfile(os.path.join(settings.BASE_DIR, *default_csv.split('/')))
        csv_choices = _list_datasets_depression_nlp_csvs()
        nlp_root = _datasets_depression_nlp_root()
        return render(
            request,
            'model/training_hub.html',
            {
                'userinfo': userinfo,
                'default_csv': default_csv,
                'default_csv_exists': exists,
                'csv_choices': csv_choices,
                'datasets_nlp_rel': 'datasets/depression_nlp',
                'datasets_nlp_abs': nlp_root,
                'datasets_zh_rel': 'datasets/depression_nlp/zh',
                'datasets_en_rel': 'datasets/depression_nlp/en',
                'train_data_locale_default': _infer_default_train_data_locale(
                    default_csv, csv_choices
                ),
                'train_sample_choices': TRAIN_HUB_SAMPLE_CHOICES,
                'sklearn_mf_choices': TRAIN_HUB_SKLEARN_MF_CHOICES,
                'torch_batch_choices': TRAIN_HUB_TORCH_BATCH_CHOICES,
            },
        )


def _safe_redirect_after_open_folder(request, default_url_name):
    next_url = request.GET.get('next')
    if next_url and url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return redirect(next_url)
    return redirect(default_url_name)


@method_decorator(check_admin_access, name='dispatch')
class OpenProjectFolderView(View):
    """
    在系统文件管理器中打开项目子目录（开发机/服务器本机）：
    - datasets_nlp: datasets/depression_nlp（可加 ?subset=zh|en 打开子目录）
    - model: model/（可加 ?algo=svm 等打开算法子目录）
    """

    def get(self, request, folder_kind=None):
        try:
            _get_login_userinfo(request)
        except Exception:
            pass
        resolved = folder_kind or getattr(self, 'folder_kind', None) or 'datasets_nlp'
        fk = (resolved if isinstance(resolved, str) else str(resolved)).lower()
        if fk == 'model':
            algo = (request.GET.get('algo') or '').strip().lower()
            known_algos = ('svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn', 'legacy')
            if algo in known_algos:
                root = os.path.join(settings.BASE_DIR, 'model', algo)
                msg = '已尝试打开 model 子目录：{}。'.format(root)
            else:
                root = os.path.join(settings.BASE_DIR, 'model')
                msg = '已尝试打开 model 文件夹：{}。可将权重、pkl、词表等放入此目录后在「融合设置」中填写相对路径。'.format(root)
            default_next = 'model_list'
        elif fk in ('datasets_nlp', 'datasets_depression_nlp'):
            base_root = _datasets_depression_nlp_root()
            subset = (request.GET.get('subset') or '').strip().lower()
            if subset == 'zh':
                root = os.path.join(base_root, 'zh')
                msg = '已尝试打开中文训练数据目录：{}。'.format(root)
            elif subset == 'en':
                root = os.path.join(base_root, 'en')
                msg = '已尝试打开英文训练数据目录：{}。'.format(root)
            else:
                root = base_root
                msg = '已尝试打开训练数据根目录：{}（含 en/、zh/）。添加或替换 CSV 后请刷新训练页。'.format(root)
            default_next = 'model_training'
        else:
            messages.error(request, '未知目录类型')
            return redirect('user_mgmt_list')
        try:
            _open_local_folder(root)
            messages.info(request, msg)
        except Exception as e:
            messages.error(request, '无法打开文件夹：{}'.format(e))
        return _safe_redirect_after_open_folder(request, default_next)


# 兼容仍引用旧类名的 urls.py（未传 folder_kind 时与 datasets_nlp 一致）
ModelTrainingOpenDatasetFolderView = OpenProjectFolderView


@method_decorator(check_admin_access, name='dispatch')
class ModelTrainingRunView(View):
    def post(self, request):
        try:
            _get_login_userinfo(request)
        except Exception:
            return JsonResponse({'ok': False, 'msg': '未登录'}, status=401)

        try:
            body = json.loads(request.body.decode('utf-8') or '{}')
        except Exception:
            body = {}
        algo = (body.get('algo') or '').strip().lower()
        csv_rel = body.get('csv_path', '').strip()

        try:
            csv_path = _safe_csv_path(csv_rel or None)
        except ValueError as e:
            _train_mirror_console('[train] reject: {}'.format(e))
            return JsonResponse({'ok': False, 'msg': str(e)})

        try:
            sk_ex, th_ex = _training_extra_args_from_body(body, algo)
        except ValueError as e:
            _train_mirror_console('[train] bad options: {}'.format(e))
            return JsonResponse({'ok': False, 'msg': str(e)}, status=400)

        # 注册表训练次数固定每次成功 +1，不在页面暴露
        inc_args = ['--registry-train-increment', '1']
        if algo in SKLEARN_ALGOS:
            cmd, err = _training_command(
                'train_single_sklearn.py',
                ['--csv', csv_path, '--algo', algo] + inc_args + (sk_ex or []),
            )
            timeout_sec = 600
        elif algo in TORCH_ALGOS:
            cmd, err = _training_command(
                'train_char_torch.py',
                ['--csv', csv_path, '--arch', algo] + inc_args + (th_ex or []),
            )
            # 大 CSV + CPU 训练常超过 15 分钟，避免误判为失败
            timeout_sec = 7200
        else:
            _train_mirror_console('[train] unknown algo: {}'.format(algo))
            return JsonResponse({'ok': False, 'msg': '未知算法: {}'.format(algo)})

        if err:
            _train_mirror_console('[train] {}'.format(err))
            return JsonResponse({'ok': False, 'msg': err})

        root = settings.BASE_DIR
        stream = _stream_training_lines(cmd, root, timeout_sec)
        resp = StreamingHttpResponse(stream, content_type='text/plain; charset=utf-8')
        resp['Cache-Control'] = 'no-cache, no-store'
        resp['X-Accel-Buffering'] = 'no'
        return resp
