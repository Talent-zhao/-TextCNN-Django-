import csv
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views import View
from django.utils.decorators import method_decorator

from app.models import CleanText, ExportLog, ModelInfo, PredictionResult, RawText, PredictionUsageLog, User
from app.userViews import check_admin_access, get_admin_panel_user

logger = logging.getLogger(__name__)


def _format_predict_error_message(exc):
    base = str(exc) if str(exc) else repr(exc)
    tb = traceback.format_exc()
    if len(tb) > 5000:
        tb = tb[:5000] + '\n…(堆栈已截断)'
    return '{}\n\n{}'.format(base, tb)


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


def _parse_post_raw_id(post):
    raw = post.get('raw_id')
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError('缺少 raw_id')
    return int(float(str(raw).strip()))


def _parse_post_model_id(post):
    raw = (post.get('model_id') or '').strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise ValueError('模型参数非法')
    return int(raw)


def _run_predict_in_subprocess(ids, model_info_id=None):
    """
    在独立 Python 进程中执行预测，避免 torch/transformers 等使主 runserver 进程崩溃后出现 ERR_CONNECTION_REFUSED。
    """
    if not ids:
        return {'result': [], 'success_count': 0, 'fail_count': 0}
    manage_py = os.path.join(settings.BASE_DIR, 'manage.py')
    if not os.path.isfile(manage_py):
        raise RuntimeError('未找到 manage.py: {}'.format(manage_py))
    cmd = [
        sys.executable,
        manage_py,
        'predict_rawtext',
        '--ids',
        ','.join(str(i) for i in ids),
    ]
    if model_info_id is not None:
        cmd.extend(['--model-id', str(model_info_id)])
    env = os.environ.copy()
    env.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoWeb.settings')
    try:
        completed = subprocess.run(
            cmd,
            cwd=settings.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError('预测超时（已超过 10 分钟），子进程已终止。') from None

    # Windows 0xC0000005（3221225477）常见于 torch/HF 原生崩溃；自动降级重试一次。
    if completed.returncode == 3221225477 and not env.get('DISABLE_TORCH_MODELS'):
        retry_env = env.copy()
        retry_env['DISABLE_TORCH_MODELS'] = '1'
        retry_env['DISABLE_HF_MODELS'] = '1'
        retry = subprocess.run(
            cmd,
            cwd=settings.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600,
            env=retry_env,
        )
        if retry.returncode == 0:
            completed = retry
        else:
            completed = retry

    out = (completed.stdout or '').strip()
    err = (completed.stderr or '').strip()

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    payload = None
    for ln in reversed(lines):
        if ln.startswith('{'):
            try:
                payload = json.loads(ln)
                break
            except json.JSONDecodeError:
                continue

    if completed.returncode != 0:
        hint = ''
        if not out and not err:
            hint = (
                '（子进程无输出即退出，常见于原生库崩溃。可尝试：减小批量、CUDA_VISIBLE_DEVICES= 仅用 CPU、'
                'runserver --noreload）'
            )
        if completed.returncode == 3221225477:
            hint += '\n已自动尝试“禁用 torch/HF 模型并回退传统模型”重试，但仍失败。'
        if payload is None:
            tail = (err + '\n' + out)[-4000:] if (err or out) else ''
            raise RuntimeError(
                '预测子进程失败，退出码 {}\n{}{}'.format(completed.returncode, tail, hint)
            )
        if payload.get('error') == '无有效 id':
            raise RuntimeError('无有效 RawText id')
        tail = (err + '\n' + out)[-3500:] if (err or out) else ''
        raise RuntimeError(
            '预测子进程失败，退出码 {}\n{}'.format(completed.returncode, tail)
        )

    if payload is None:
        combined = (completed.stdout or '') + '\n' + (completed.stderr or '')
        raise RuntimeError(
            '无法解析子进程输出（应含一行 JSON）：\n{}'.format(combined[-4000:])
        )
    return payload


def _run_predict_in_chunks(ids, model_info_id=None, chunk_size=200):
    """
    大批量时按块执行子进程预测，避免命令行参数过长或单次执行时间过久。
    """
    ids = [int(i) for i in ids if str(i).isdigit()]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {'result': [], 'success_count': 0, 'fail_count': 0}

    all_result = []
    success_count = 0
    fail_count = 0
    for i in range(0, len(ids), max(1, int(chunk_size))):
        batch_ids = ids[i:i + max(1, int(chunk_size))]
        payload = _run_predict_in_subprocess(batch_ids, model_info_id=model_info_id)
        result = payload.get('result') or []
        sc = payload.get('success_count')
        if sc is None:
            sc = len([r for r in result if r.get('ok')])
        fc = payload.get('fail_count')
        if fc is None:
            fc = len(result) - int(sc)
        success_count += int(sc)
        fail_count += int(fc)
        all_result.extend(result)
    return {'result': all_result, 'success_count': success_count, 'fail_count': fail_count}


def _build_filter_json(request):
    params = dict(request.GET)
    if 'export' in params:
        params.pop('export')
    normalized = {k: v[0] if isinstance(v, list) and v else v for k, v in params.items()}
    return json.dumps(normalized, ensure_ascii=False)


@method_decorator(check_admin_access, name='dispatch')
class PendingPredictListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        page = int(request.GET.get('page', 1))
        rows = RawText.objects.filter(status=RawText.STATUS_PENDING_PREDICT).order_by('-updated_at')
        page_obj = Paginator(rows, 20).get_page(page)
        model_options = ModelInfo.objects.filter(status='ready').order_by('-is_active', 'name', 'version')
        active_model = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
        return render(
            request,
            'risk/pending_predict_list.html',
            {
                'userinfo': userinfo,
                'page_obj': page_obj,
                'model_options': model_options,
                'default_model_id': active_model.id if active_model else '',
            },
        )

    def post(self, request):
        ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
        action = request.POST.get('action')

        try:
            model_info_id = _parse_post_model_id(request.POST)
            if action == 'single_predict':
                rid = _parse_post_raw_id(request.POST)
                payload = _run_predict_in_subprocess([rid], model_info_id=model_info_id)
                result = payload.get('result') or []
                first = result[0] if result else {}
                if not first.get('ok'):
                    raise RuntimeError(first.get('error', '预测失败'))
                pred_id = first.get('prediction_id')
                if ajax:
                    return JsonResponse(
                        {
                            'status': 1,
                            'msg': '预测完成',
                            'prediction_id': pred_id,
                            'alert_id': first.get('alert_id'),
                        }
                    )
                messages.success(request, '预测完成（结果 ID {}）'.format(pred_id))
                return redirect('pending_predict_list')

            if action == 'batch_predict':
                ids = [int(i) for i in request.POST.get('raw_ids', '').split(',') if i.strip().isdigit()]
                ids = list(dict.fromkeys(ids))
                if not ids:
                    if ajax:
                        return JsonResponse({'status': 0, 'msg': '请先勾选待预测数据'})
                    messages.warning(request, '请先勾选待预测数据')
                    return redirect('pending_predict_list')
                payload = _run_predict_in_chunks(ids, model_info_id=model_info_id)
                result = payload.get('result') or []
                success_count = payload.get('success_count')
                if success_count is None:
                    success_count = len([i for i in result if i.get('ok')])
                fail_count = payload.get('fail_count')
                if fail_count is None:
                    fail_count = len(result) - success_count
                if ajax:
                    return JsonResponse(
                        {
                            'status': 1,
                            'msg': '批量执行完成，成功{}条，失败{}条'.format(success_count, fail_count),
                            'result': result,
                        }
                    )
                messages.success(
                    request,
                    '批量预测完成：成功 {} 条，失败 {} 条'.format(success_count, fail_count),
                )
                if fail_count:
                    errs = [i.get('error', '') for i in result if not i.get('ok') and i.get('error')]
                    if errs:
                        fe = errs[0]
                        if len(fe) > 6000:
                            fe = fe[:6000] + '\n…(已截断)'
                        messages.warning(request, fe)
                return redirect('pending_predict_list')

            if action == 'batch_predict_all':
                ids = list(
                    RawText.objects.filter(status=RawText.STATUS_PENDING_PREDICT)
                    .order_by('id')
                    .values_list('id', flat=True)
                )
                if not ids:
                    if ajax:
                        return JsonResponse({'status': 0, 'msg': '当前没有待预测数据'})
                    messages.info(request, '当前没有待预测数据')
                    return redirect('pending_predict_list')
                payload = _run_predict_in_chunks(ids, model_info_id=model_info_id)
                result = payload.get('result') or []
                success_count = payload.get('success_count')
                if success_count is None:
                    success_count = len([i for i in result if i.get('ok')])
                fail_count = payload.get('fail_count')
                if fail_count is None:
                    fail_count = len(result) - success_count
                if ajax:
                    return JsonResponse(
                        {
                            'status': 1,
                            'msg': '全部待预测数据执行完成，成功{}条，失败{}条'.format(success_count, fail_count),
                            'result': result,
                        }
                    )
                messages.success(
                    request,
                    '全部待预测数据预测完成：成功 {} 条，失败 {} 条'.format(success_count, fail_count),
                )
                if fail_count:
                    errs = [i.get('error', '') for i in result if not i.get('ok') and i.get('error')]
                    if errs:
                        fe = errs[0]
                        if len(fe) > 6000:
                            fe = fe[:6000] + '\n…(已截断)'
                        messages.warning(request, fe)
                return redirect('pending_predict_list')

        except (TypeError, ValueError) as e:
            if ajax:
                return JsonResponse({'status': 0, 'msg': '参数错误：{}'.format(e)}, status=400)
            messages.error(request, '参数错误：{}'.format(e))
            return redirect('pending_predict_list')
        except Exception as e:
            logger.exception('pending_predict failed action=%s', action)
            detail = _format_predict_error_message(e)
            if ajax:
                return JsonResponse({'status': 0, 'msg': detail}, status=500)
            messages.error(request, detail)
            return redirect('pending_predict_list')

        if ajax:
            return JsonResponse({'status': 0, 'msg': '不支持的操作'})
        messages.error(request, '不支持的操作')
        return redirect('pending_predict_list')


@method_decorator(check_admin_access, name='dispatch')
class PredictionResultListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        model_version = request.GET.get('model_version', '').strip()
        risk_level = request.GET.get('risk_level', '').strip()
        is_alert = request.GET.get('is_alert', '').strip()
        start_time = request.GET.get('start_time', '').strip()
        end_time = request.GET.get('end_time', '').strip()
        export = request.GET.get('export', '').strip()
        page = int(request.GET.get('page', 1))
        rows = PredictionResult.objects.select_related('raw_text', 'model_info', 'predicted_label').all().order_by('-created_at')
        if model_version:
            rows = rows.filter(model_info__version=model_version)
        if risk_level.isdigit():
            rows = rows.filter(risk_level=int(risk_level))
        if is_alert in ('0', '1'):
            rows = rows.filter(is_alert_triggered=(is_alert == '1'))
        if start_time:
            try:
                rows = rows.filter(created_at__gte=datetime.strptime(start_time, '%Y-%m-%d'))
            except Exception:
                pass
        if end_time:
            try:
                rows = rows.filter(created_at__lte=datetime.strptime(end_time + ' 23:59:59', '%Y-%m-%d %H:%M:%S'))
            except Exception:
                pass
        if export in ('csv', 'json'):
            return self._export_predictions(request, rows, export)
        page_obj = Paginator(rows, 20).get_page(page)
        pred_vers = set(
            PredictionResult.objects.exclude(model_info__version__isnull=True)
            .exclude(model_info__version='')
            .values_list('model_info__version', flat=True)
            .distinct()
        )
        model_version_labels = []
        for m in ModelInfo.objects.filter(status='ready').order_by('name'):
            if m.version:
                model_version_labels.append((m.version, '{} — {}'.format(m.name, m.version)))
        for v in sorted(pred_vers - {p[0] for p in model_version_labels}):
            model_version_labels.append((v, v))
        model_version_labels.sort(key=lambda x: x[1])
        return render(
            request,
            'risk/prediction_result_list.html',
            {
                'userinfo': userinfo,
                'page_obj': page_obj,
                'model_version_labels': model_version_labels,
                'model_version': model_version,
                'risk_level': risk_level,
                'is_alert': is_alert,
                'start_time': start_time,
                'end_time': end_time
            }
        )

    def _export_predictions(self, request, rows, export):
        userinfo = _get_login_userinfo(request)
        total_count = rows.count()
        ExportLog.objects.create(
            export_type='prediction_result',
            export_format=export,
            exporter=userinfo,
            filter_json=_build_filter_json(request),
            export_count=total_count,
        )
        if export == 'json':
            data = [
                {
                    'id': r.id,
                    'raw_text_id': r.raw_text_id,
                    'model_version': r.model_info.version if r.model_info else '',
                    'risk_level': r.risk_level,
                    'is_alert_triggered': r.is_alert_triggered,
                    'probability_score': r.probability_score,
                    'rule_score': r.rule_score,
                    'final_risk_score': r.final_risk_score,
                    'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                } for r in rows
            ]
            response = HttpResponse(json.dumps(data, ensure_ascii=False, indent=2), content_type='application/json')
            response['Content-Disposition'] = 'attachment; filename="prediction_results.json"'
            return response

        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        response['Content-Disposition'] = 'attachment; filename="prediction_results.csv"'
        writer = csv.writer(response)
        writer.writerow(['ID', '文本ID', '模型版本', '风险等级', '触发预警', '模型概率分', '规则分', '最终风险分', '创建时间'])
        for r in rows:
            writer.writerow([
                r.id, r.raw_text_id, r.model_info.version if r.model_info else '', r.risk_level,
                '是' if r.is_alert_triggered else '否', r.probability_score, r.rule_score,
                r.final_risk_score, r.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        return response


@method_decorator(check_admin_access, name='dispatch')
class PredictionResultDetailView(View):
    def get(self, request, prediction_id):
        userinfo = _get_login_userinfo(request)
        pred = get_object_or_404(PredictionResult.objects.select_related('raw_text', 'model_info', 'predicted_label'), id=prediction_id)
        clean_obj = CleanText.objects.filter(raw_text=pred.raw_text).first()
        detail = {}
        if pred.detail_json:
            try:
                detail = json.loads(pred.detail_json)
            except Exception:
                detail = {'raw_detail': pred.detail_json}
        return render(
            request,
            'risk/prediction_result_detail.html',
            {'userinfo': userinfo, 'pred': pred, 'clean_obj': clean_obj, 'detail': detail}
        )


@method_decorator(check_admin_access, name='dispatch')
class PredictionUsageLogListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        user_id = (request.GET.get('user_id') or '').strip()
        model_id = (request.GET.get('model_id') or '').strip()
        success = (request.GET.get('success') or '').strip()
        start_time = (request.GET.get('start_time') or '').strip()
        end_time = (request.GET.get('end_time') or '').strip()
        page = int(request.GET.get('page', 1))

        rows = PredictionUsageLog.objects.select_related('user', 'model_info').all().order_by('-created_at')
        if user_id.isdigit():
            rows = rows.filter(user_id=int(user_id))
        if model_id.isdigit():
            rows = rows.filter(model_info_id=int(model_id))
        if success in ('0', '1'):
            rows = rows.filter(success=(success == '1'))
        if start_time:
            try:
                rows = rows.filter(created_at__gte=datetime.strptime(start_time, '%Y-%m-%d'))
            except Exception:
                pass
        if end_time:
            try:
                rows = rows.filter(created_at__lte=datetime.strptime(end_time + ' 23:59:59', '%Y-%m-%d %H:%M:%S'))
            except Exception:
                pass

        page_obj = Paginator(rows, 30).get_page(page)
        users = User.objects.filter(is_active=True).order_by('username', 'id')
        models = ModelInfo.objects.all().order_by('name', 'version')
        return render(
            request,
            'risk/usage_log_list.html',
            {
                'userinfo': userinfo,
                'page_obj': page_obj,
                'users': users,
                'models': models,
                'filter_user_id': user_id,
                'filter_model_id': model_id,
                'filter_success': success,
                'start_time': start_time,
                'end_time': end_time,
            }
        )
