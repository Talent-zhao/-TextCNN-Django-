# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

from django.contrib import messages
from django.core.paginator import Paginator
from django.db import IntegrityError, transaction
from django.db.models import Prefetch
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views import View
from django.utils.decorators import method_decorator

from app.dataViews import build_admin_head_tail_pagination
from app.models import AnnotationRecord, LabelDefinition, RawText, User
from app.userViews import check_admin_access, get_admin_panel_user


LABEL_PRESETS = {
    'binary': [
        (0, '正常文本', 0, '无明显风险'),
        (1, '风险文本', 2, '存在心理风险线索'),
    ],
    'four': [
        (0, '正常文本', 0, '无明显风险'),
        (1, '轻度负向情绪', 1, '轻度负向表达'),
        (2, '中风险心理异常倾向', 2, '需重点关注'),
        (3, '高风险危机表达', 3, '高危表达，需尽快复核'),
    ],
}

# 内置二分类 / 四分类涉及的全部 code（仅对这些做启用/停用，不动自定义 code）
ALL_PRESET_CODES = set()
for _preset_rows in LABEL_PRESETS.values():
    for row in _preset_rows:
        ALL_PRESET_CODES.add(row[0])


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


def _rename_if_name_taken(_exclude_pk, conflicting_obj):
    """name 全局唯一：先把占位行改名再写入目标名称。"""
    if conflicting_obj is None:
        return
    suffix = '_{}'.format(conflicting_obj.pk)
    base = (conflicting_obj.name or 'label')[: max(1, 64 - len(suffix))]
    conflicting_obj.name = (base + suffix)[-64:]
    conflicting_obj.save(update_fields=['name'])


def ensure_default_labels(mode='four'):
    """
    写入内置标签集；解决 name 唯一约束下的重名冲突；
    切换二分类时会停用 code 2、3，切换四分类时全部启用 0–3（自定义 code 不受影响）。
    """
    preset = LABEL_PRESETS.get(mode, LABEL_PRESETS['four'])
    preset_codes = {row[0] for row in preset}
    try:
        with transaction.atomic():
            for code, name, risk_level, desc in preset:
                obj = LabelDefinition.objects.filter(code=code).first()
                if obj is None:
                    taken = LabelDefinition.objects.filter(name=name).first()
                    if taken:
                        _rename_if_name_taken(None, taken)
                    LabelDefinition.objects.create(
                        code=code,
                        name=name,
                        risk_level=risk_level,
                        description=desc,
                        is_active=True,
                    )
                else:
                    taken = LabelDefinition.objects.filter(name=name).exclude(pk=obj.pk).first()
                    if taken:
                        _rename_if_name_taken(obj.pk, taken)
                    obj.name = name
                    obj.risk_level = risk_level
                    obj.description = desc
                    obj.is_active = True
                    obj.save(
                        update_fields=['name', 'risk_level', 'description', 'is_active']
                    )
            LabelDefinition.objects.filter(code__in=ALL_PRESET_CODES).exclude(
                code__in=preset_codes
            ).update(is_active=False)
    except IntegrityError as e:
        raise ValueError('标签初始化失败（名称或编码冲突），请检查 LabelDefinition：{}'.format(e)) from e


def ensure_default_labels_if_empty(mode='four'):
    """避免每次打开列表都写库，减轻 SQLite 锁竞争。"""
    if LabelDefinition.objects.exists():
        return
    ensure_default_labels(mode)


@method_decorator(check_admin_access, name='dispatch')
class AnnotationListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        ensure_default_labels_if_empty('four')
        keyword = request.GET.get('keyword', '').strip()
        page = int(request.GET.get('page', 1))
        status = request.GET.get('status', RawText.STATUS_PENDING_LABEL)

        ann_prefetch = Prefetch(
            'annotations',
            queryset=AnnotationRecord.objects.select_related('label').order_by('-created_at'),
            to_attr='prefetched_annotations',
        )
        queryset = (
            RawText.objects.all()
            .order_by('-created_at')
            .prefetch_related(ann_prefetch)
        )
        if keyword:
            queryset = queryset.filter(content__icontains=keyword)
        if status:
            queryset = queryset.filter(status=status)

        paginator = Paginator(queryset, 20)
        page_obj = paginator.get_page(page)
        labels = LabelDefinition.objects.filter(is_active=True).order_by('code')

        context = {
            'userinfo': userinfo,
            'page_obj': page_obj,
            'labels': labels,
            'keyword': keyword,
            'status': status,
            'status_choices': RawText.STATUS_CHOICES,
        }
        context.update(build_admin_head_tail_pagination(paginator, page_obj))
        return render(request, 'annotation/annotation_list.html', context)

    def post(self, request):
        action = request.POST.get('action')
        if action == 'single':
            return self._single_annotate(request)
        if action == 'batch':
            return self._batch_annotate(request)
        if action == 'batch_send_predict':
            return self._batch_send_to_pending_predict(request)
        if action == 'init_binary':
            try:
                ensure_default_labels('binary')
            except ValueError as e:
                return JsonResponse({'status': 0, 'msg': str(e)})
            return JsonResponse(
                {
                    'status': 1,
                    'msg': '二分类标签已就绪（当前仅启用 code 0、1）；下拉框将只显示 2 个标签',
                }
            )
        if action == 'init_four':
            try:
                ensure_default_labels('four')
            except ValueError as e:
                return JsonResponse({'status': 0, 'msg': str(e)})
            return JsonResponse(
                {
                    'status': 1,
                    'msg': '四分类标签已就绪（启用 code 0–3）',
                }
            )
        return JsonResponse({'status': 0, 'msg': '不支持的操作'})

    def _single_annotate(self, request):
        userinfo = _get_login_userinfo(request)
        raw_id = request.POST.get('raw_id')
        label_id = request.POST.get('label_id')
        remark = request.POST.get('remark', '')

        raw_obj = RawText.objects.filter(id=raw_id).first()
        label_obj = LabelDefinition.objects.filter(id=label_id).first()
        if not raw_obj or not label_obj:
            return JsonResponse({'status': 0, 'msg': '参数无效'})

        AnnotationRecord.objects.create(
            raw_text=raw_obj,
            label=label_obj,
            annotator=userinfo,
            dataset_split='train',
            remark=remark
        )

        if raw_obj.status in (RawText.STATUS_PENDING_LABEL, RawText.STATUS_CLEANED):
            if raw_obj.status == RawText.STATUS_CLEANED:
                raw_obj.transit_to(RawText.STATUS_PENDING_LABEL)
            raw_obj.transit_to(RawText.STATUS_LABELED)
        return JsonResponse({'status': 1, 'msg': '单条标注成功，状态为「已标注」。可在本页筛选「已标注」查看，或点「送待预测」进入预测环节。'})

    def _batch_annotate(self, request):
        userinfo = _get_login_userinfo(request)
        ids = request.POST.get('raw_ids', '')
        label_id = request.POST.get('label_id')
        remark = request.POST.get('remark', '')
        label_obj = LabelDefinition.objects.filter(id=label_id).first()
        if not label_obj:
            return JsonResponse({'status': 0, 'msg': '标签不存在'})

        raw_ids = [int(i) for i in ids.split(',') if i.strip().isdigit()]
        rows = RawText.objects.filter(id__in=raw_ids)
        count = 0
        for row in rows:
            AnnotationRecord.objects.create(
                raw_text=row,
                label=label_obj,
                annotator=userinfo,
                dataset_split='train',
                remark=remark
            )
            if row.status == RawText.STATUS_CLEANED:
                row.transit_to(RawText.STATUS_PENDING_LABEL)
            if row.status == RawText.STATUS_PENDING_LABEL:
                row.transit_to(RawText.STATUS_LABELED)
            count += 1
        return JsonResponse(
            {
                'status': 1,
                'msg': '批量标注成功，共 {} 条；状态为「已标注」，可在本页筛选查看或送待预测。'.format(count),
            }
        )

    def _batch_send_to_pending_predict(self, request):
        ids = request.POST.get('raw_ids', '')
        raw_ids = [int(i) for i in ids.split(',') if i.strip().isdigit()]
        if not raw_ids:
            return JsonResponse({'status': 0, 'msg': '请先勾选记录'})
        n = 0
        for row in RawText.objects.filter(id__in=raw_ids, status=RawText.STATUS_LABELED).order_by('id'):
            row.transit_to(RawText.STATUS_PENDING_PREDICT)
            n += 1
        return JsonResponse(
            {
                'status': 1,
                'msg': '已送待预测 {} 条（仅处理状态为「已标注」的勾选行；可至「待预测」页执行预测）'.format(n),
            }
        )
