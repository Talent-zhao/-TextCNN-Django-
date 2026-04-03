import json

from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render
from django.views import View
from django.utils.decorators import method_decorator

from app.models import ExportLog, User
from app.userViews import check_admin_access, get_admin_panel_user


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


@method_decorator(check_admin_access, name='dispatch')
class ExportLogListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        page_raw = (request.GET.get('page') or '1').strip()
        page = int(page_raw) if page_raw.isdigit() else 1
        export_type = (request.GET.get('export_type') or '').strip()
        export_format = (request.GET.get('export_format') or '').strip()

        rows = ExportLog.objects.select_related('exporter').all().order_by('-created_at')
        valid_types = {k for k, _ in ExportLog.EXPORT_TYPE_CHOICES}
        valid_formats = {k for k, _ in ExportLog.EXPORT_FORMAT_CHOICES}
        if export_type in valid_types:
            rows = rows.filter(export_type=export_type)
        if export_format in valid_formats:
            rows = rows.filter(export_format=export_format)

        page_obj = Paginator(rows, 20).get_page(page)
        return render(
            request,
            'audit/export_log_list.html',
            {
                'userinfo': userinfo,
                'page_obj': page_obj,
                'export_type': export_type,
                'export_format': export_format,
                'type_choices': ExportLog.EXPORT_TYPE_CHOICES,
                'format_choices': ExportLog.EXPORT_FORMAT_CHOICES,
            },
        )


@method_decorator(check_admin_access, name='dispatch')
class ExportLogDetailView(View):
    def get(self, request, log_id):
        userinfo = _get_login_userinfo(request)
        row = get_object_or_404(ExportLog.objects.select_related('exporter'), id=log_id)
        filter_data = {}
        filter_json_pretty = '{}'
        if row.filter_json:
            try:
                filter_data = json.loads(row.filter_json)
                filter_json_pretty = json.dumps(filter_data, ensure_ascii=False, indent=2)
            except Exception:
                filter_data = {'raw': row.filter_json}
                filter_json_pretty = json.dumps(filter_data, ensure_ascii=False, indent=2)
        return render(
            request,
            'audit/export_log_detail.html',
            {
                'userinfo': userinfo,
                'row': row,
                'filter_data': filter_data,
                'filter_json_pretty': filter_json_pretty,
            },
        )
