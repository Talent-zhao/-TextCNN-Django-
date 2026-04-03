import csv
import hashlib
import json
import re
from datetime import datetime

from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views import View
from django.utils.decorators import method_decorator

from app.models import RawText, TextSource, User, CleanText
from app.userViews import check_admin_access, get_admin_panel_user


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


def _build_hash(content):
    return hashlib.sha256((content or '').strip().encode('utf-8')).hexdigest()


def _parse_publish_time(raw):
    if not raw:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(raw).strip(), fmt)
        except ValueError:
            continue
    return None


def _simple_clean_text(content):
    text = re.sub(r'<[^>]+>', '', content or '')
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return ' '.join(text.split()).strip()


def _clean_and_transit_to_pending_predict(raw_obj):
    """
    规则清洗写入 CleanText，再沿状态机进入「待预测」。
    不经人工标注、不创建 AnnotationRecord（自动走过 已清洗→待标注→已标注）。
    调用前 raw_obj 须为「未清洗」。
    """
    if raw_obj.status != RawText.STATUS_PENDING_CLEAN:
        return
    cleaned = _simple_clean_text(raw_obj.content)
    CleanText.objects.update_or_create(
        raw_text=raw_obj,
        defaults={
            'cleaned_content': cleaned,
            'tokenized_content': '',
            'removed_special_chars': True,
            'removed_stopwords': False,
        },
    )
    raw_obj.transit_to(RawText.STATUS_CLEANED)
    raw_obj.transit_to(RawText.STATUS_PENDING_LABEL)
    raw_obj.transit_to(RawText.STATUS_LABELED)
    raw_obj.transit_to(RawText.STATUS_PENDING_PREDICT)


def rawtext_import_redirect(request):
    """旧「数据导入」独立页已合并到数据管理列表，此处仅做兼容跳转。"""
    return redirect(reverse('rawtext_list') + '?focus=import')


def _decode_upload_bytes(raw: bytes) -> str:
    if not raw:
        return ''
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def import_rawtext_from_upload(request):
    """
    从上传文件导入 RawText（CSV/JSON）。
    CSV 正文列支持：content、text（与训练集 text,label 一致）；可选 id/external_id、name/author、time/publish_time。
    """
    upload = request.FILES.get('import_file')
    if not upload:
        messages.error(request, '请先选择要导入的文件')
        return redirect(reverse('rawtext_list') + '?focus=import')

    filename = (upload.name or '').lower()
    imported_count = 0
    skipped_count = 0

    direct_predict = request.POST.get('import_direct_predict') == '1'

    if filename.endswith('.csv'):
        source_type = 'csv'
        raw_bytes = upload.read()
        decoded = _decode_upload_bytes(raw_bytes).splitlines()
        reader = csv.DictReader(decoded)
        rows = list(reader)
    elif filename.endswith('.json'):
        source_type = 'json'
        raw_bytes = upload.read()
        text = _decode_upload_bytes(raw_bytes)
        try:
            rows = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            messages.error(request, 'JSON 解析失败，请检查文件编码与格式')
            return redirect(reverse('rawtext_list') + '?focus=import')
        if isinstance(rows, dict):
            rows = rows.get('data', [])
        if not isinstance(rows, list):
            messages.error(request, 'JSON 须为对象数组，或 {"data": [...]}')
            return redirect(reverse('rawtext_list') + '?focus=import')
    else:
        messages.error(request, '仅支持 CSV/JSON 文件')
        return redirect(reverse('rawtext_list') + '?focus=import')

    source, _ = TextSource.objects.get_or_create(
        name='{}导入'.format(source_type.upper()),
        defaults={'source_type': source_type},
    )

    for row in rows:
        if not isinstance(row, dict):
            skipped_count += 1
            continue
        content = (
            row.get('content')
            or row.get('text')
            or ''
        )
        content = (content or '').strip()
        if not content:
            skipped_count += 1
            continue
        dedup_hash = _build_hash(content)
        if RawText.objects.filter(dedup_hash=dedup_hash).exists():
            skipped_count += 1
            continue

        raw_obj = RawText.objects.create(
            source=source,
            external_id=str(row.get('id') or row.get('external_id') or ''),
            author_name=str(row.get('name') or row.get('author') or ''),
            content=content,
            publish_time=_parse_publish_time(row.get('time') or row.get('publish_time')),
            status=RawText.STATUS_PENDING_CLEAN,
            dedup_hash=dedup_hash,
        )
        imported_count += 1
        if direct_predict:
            _clean_and_transit_to_pending_predict(raw_obj)

    extra = ''
    if direct_predict and imported_count:
        extra = '；本次新增 {} 条已规则清洗并进入「待预测」（跳过人工标注，可直接去「待预测」页跑模型）'.format(
            imported_count
        )
    messages.info(
        request,
        '导入完成：新增 {} 条，跳过 {} 条{}'.format(imported_count, skipped_count, extra),
    )
    return redirect(reverse('rawtext_list'))


def build_admin_head_tail_pagination(paginator, page_obj, head_size=4, tail_size=4):
    """
    管理端列表：首页 + 前 N 页 + 跳转 + 后 N 页 + 末页。
    页数较少、头尾重叠时合并为连续页码。
    """
    num_pages = paginator.num_pages
    current = page_obj.number
    if num_pages <= 1:
        return {
            'pagination_show': True,
            'pagination_merged': True,
            'page_all': [1] if num_pages >= 1 else [],
            'page_head': [],
            'page_tail': [],
            'num_pages': max(1, num_pages),
            'current': current,
        }

    head = list(range(1, min(head_size, num_pages) + 1))
    tail_start = max(1, num_pages - tail_size + 1)
    tail = list(range(tail_start, num_pages + 1))
    merged = head[-1] >= tail[0]

    if merged:
        return {
            'pagination_show': True,
            'pagination_merged': True,
            'page_all': list(range(1, num_pages + 1)),
            'page_head': [],
            'page_tail': [],
            'num_pages': num_pages,
            'current': current,
        }

    return {
        'pagination_show': True,
        'pagination_merged': False,
        'page_head': head,
        'page_tail': tail,
        'page_all': [],
        'num_pages': num_pages,
        'current': current,
    }


@method_decorator(check_admin_access, name='dispatch')
class RawTextListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        keyword = request.GET.get('keyword', '').strip()
        status = request.GET.get('status', '').strip()
        page = int(request.GET.get('page', 1))
        focus_import = request.GET.get('focus') == 'import'

        queryset = RawText.objects.select_related('source').all().order_by('-created_at')
        status_choices = [c for c in RawText.STATUS_CHOICES if c[0] != RawText.STATUS_CLEANED]
        allowed_statuses = {c[0] for c in status_choices}
        if status and status not in allowed_statuses:
            status = ''
        if keyword:
            queryset = queryset.filter(content__icontains=keyword)
        if status:
            queryset = queryset.filter(status=status)

        paginator = Paginator(queryset, 20)
        page_obj = paginator.get_page(page)

        context = {
            'userinfo': userinfo,
            'page_obj': page_obj,
            'status_choices': status_choices,
            'keyword': keyword,
            'status': status,
            'focus_import': focus_import,
        }
        context.update(build_admin_head_tail_pagination(paginator, page_obj))
        return render(request, 'data/rawtext_list.html', context)

    def post(self, request):
        action = request.POST.get('action')
        if action == 'import':
            return import_rawtext_from_upload(request)
        if action == 'batch_clean':
            return self._batch_clean(request)
        if action == 'dedup':
            return self._deduplicate(request)
        if action == 'batch_delete':
            return self._batch_delete_selected(request)
        if action == 'delete_all_rawtext':
            return self._delete_all_rawtext(request)
        return JsonResponse({'status': 0, 'msg': '不支持的操作'})

    def _batch_clean(self, request):
        """
        将所有「未清洗」记录：规则清洗写入 CleanText 后，按状态机进入「待标注」
        （实际流转：未清洗 → 已清洗 → 待标注，表示清洗已完成，等待标注）。
        """
        n = 0
        qs = RawText.objects.filter(status=RawText.STATUS_PENDING_CLEAN).order_by('id')
        for raw_obj in qs.iterator():
            cleaned = _simple_clean_text(raw_obj.content)
            CleanText.objects.update_or_create(
                raw_text=raw_obj,
                defaults={
                    'cleaned_content': cleaned,
                    'tokenized_content': '',
                    'removed_special_chars': True,
                    'removed_stopwords': False,
                },
            )
            raw_obj.transit_to(RawText.STATUS_CLEANED)
            raw_obj.transit_to(RawText.STATUS_PENDING_LABEL)
            n += 1
        return JsonResponse(
            {
                'status': 1,
                'msg': '批量清洗完成：共处理 {} 条（未清洗 → 待标注，已写入清洗结果）'.format(n),
            }
        )

    def _batch_delete_selected(self, request):
        ids = (request.POST.get('raw_ids') or '').strip()
        raw_ids = [int(i) for i in ids.split(',') if i.strip().isdigit()]
        if not raw_ids:
            return JsonResponse({'status': 0, 'msg': '请先勾选要删除的记录'})
        qs = RawText.objects.filter(id__in=raw_ids)
        n = qs.count()
        qs.delete()
        return JsonResponse(
            {
                'status': 1,
                'msg': '已删除 {} 条 RawText（及关联清洗/标注/预测等）'.format(n),
            }
        )

    def _delete_all_rawtext(self, request):
        n = RawText.objects.count()
        RawText.objects.all().delete()
        return JsonResponse(
            {
                'status': 1,
                'msg': '已清空全部 RawText，共 {} 条（关联数据已由数据库级联删除）'.format(n),
            }
        )

    def _deduplicate(self, request):
        deleted = 0
        keep_ids = set()
        all_rows = RawText.objects.all().order_by('id')
        for row in all_rows:
            if not row.dedup_hash:
                row.dedup_hash = _build_hash(row.content)
                row.save(update_fields=['dedup_hash'])
            if row.dedup_hash in keep_ids:
                row.delete()
                deleted += 1
            else:
                keep_ids.add(row.dedup_hash)
        return JsonResponse({'status': 1, 'msg': f'去重完成，删除 {deleted} 条重复记录'})


@method_decorator(check_admin_access, name='dispatch')
class RawTextDetailView(View):
    def get(self, request, raw_id):
        userinfo = _get_login_userinfo(request)
        raw_obj = get_object_or_404(
            RawText.objects.select_related('source').prefetch_related('annotations', 'predictions'),
            id=raw_id
        )
        clean_obj = CleanText.objects.filter(raw_text=raw_obj).first()
        context = {
            'userinfo': userinfo,
            'raw_obj': raw_obj,
            'clean_obj': clean_obj,
            'can_clean_here': raw_obj.can_transit_to(RawText.STATUS_CLEANED),
        }
        return render(request, 'data/rawtext_detail.html', context)

    def post(self, request, raw_id):
        if request.POST.get('action') != 'clean':
            messages.error(request, '不支持的操作')
            return redirect('rawtext_detail', raw_id=raw_id)
        raw_obj = get_object_or_404(RawText, id=raw_id)
        if not raw_obj.can_transit_to(RawText.STATUS_CLEANED):
            messages.warning(request, '当前状态不可清洗（仅「未清洗」可执行；其它阶段请用样本标注/预测流程推进）')
            return redirect('rawtext_detail', raw_id=raw_id)
        cleaned = _simple_clean_text(raw_obj.content)
        CleanText.objects.update_or_create(
            raw_text=raw_obj,
            defaults={
                'cleaned_content': cleaned,
                'tokenized_content': '',
                'removed_special_chars': True,
                'removed_stopwords': False,
            },
        )
        raw_obj.transit_to(RawText.STATUS_CLEANED)
        messages.success(request, '清洗完成，已写入清洗结果并将状态设为「已清洗」')
        return redirect('rawtext_detail', raw_id=raw_id)


