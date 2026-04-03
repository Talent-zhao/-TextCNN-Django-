import json

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View

from app.models import ModelInfo, User, ModelPlan, UserPlan
from app.userViews import check_admin_access, get_admin_panel_user


def _admin_ctx(request):
    return {'userinfo': get_admin_panel_user(request)}


def _parse_allowed_types(request):
    if request.POST.get('algo_mode') != 'pick':
        return ''
    valid = {c[0] for c in ModelInfo.assignable_model_type_choices()}
    picked = [k for k in request.POST.getlist('algo_type') if k in valid]
    return json.dumps(picked, ensure_ascii=False)


def _algo_form_context(edit_obj):
    if edit_obj is None:
        return {'algo_mode': 'all', 'picked_list': []}
    s = edit_obj.allowed_model_types_set()
    if s is None:
        return {'algo_mode': 'all', 'picked_list': []}
    return {'algo_mode': 'pick', 'picked_list': list(s)}


def _published_model_registrations_qs():
    return ModelInfo.objects.filter(status='ready', listed_for_users=True).order_by(
        'model_type', 'name', 'version', 'id'
    )


def _plan_algo_form_context(plan):
    from app.plan_allowance import parse_plan_id_allow, parse_plan_type_allow

    if plan is None:
        return {'plan_algo_mode': 'all', 'plan_picked_ids': [], 'plan_has_legacy_types_only': False}
    ids = parse_plan_id_allow(plan.allowed_model_ids_json)
    if ids is not None and len(ids) > 0:
        return {
            'plan_algo_mode': 'pick',
            'plan_picked_ids': sorted(ids),
            'plan_has_legacy_types_only': False,
        }
    types = parse_plan_type_allow(plan.allowed_model_types_json)
    if types is not None and len(types) > 0:
        expanded = list(
            ModelInfo.objects.filter(
                status='ready', listed_for_users=True, model_type__in=types
            )
            .order_by('model_type', 'name', 'id')
            .values_list('id', flat=True)
        )
        return {
            'plan_algo_mode': 'pick',
            'plan_picked_ids': expanded,
            'plan_has_legacy_types_only': True,
        }
    return {'plan_algo_mode': 'all', 'plan_picked_ids': [], 'plan_has_legacy_types_only': False}


@method_decorator(check_admin_access, name='dispatch')
class UserMgmtListView(View):
    def get(self, request):
        q = (request.GET.get('q') or '').strip()
        qs = User.objects.all().order_by('-id')
        if q:
            qs = qs.filter(Q(username__icontains=q) | Q(tel__icontains=q))
        paginator = Paginator(qs, 15)
        page_obj = paginator.get_page(request.GET.get('page') or 1)
        ctx = _admin_ctx(request)
        ctx.update({'page_obj': page_obj, 'q': q})
        return render(request, 'workbench/user_list.html', ctx)


@method_decorator(check_admin_access, name='dispatch')
class UserMgmtFormView(View):
    def get(self, request, pk=None):
        edit_obj = get_object_or_404(User, pk=pk) if pk else None
        ctx = _admin_ctx(request)
        ctx.update(
            {
                'edit_obj': edit_obj,
                'role_choices': User.ROLE_CHOICES,
            }
        )
        return render(request, 'workbench/user_form.html', ctx)

    def post(self, request, pk=None):
        edit_obj = get_object_or_404(User, pk=pk) if pk else None
        actor = get_admin_panel_user(request)

        username = (request.POST.get('username') or '').strip()
        tel = (request.POST.get('tel') or '').strip()
        pwd = (request.POST.get('pwd') or '').strip()
        email = (request.POST.get('email') or '').strip() or None
        role = request.POST.get('role') or 'user'
        if role not in dict(User.ROLE_CHOICES):
            role = 'user'
        is_active = request.POST.get('is_active') == 'on'
        admin_note = (request.POST.get('admin_note') or '').strip() or None
        sex = request.POST.get('sex') or None
        if sex and sex not in dict(User.SEX_CHOICE):
            sex = None
        address = (request.POST.get('address') or '').strip() or None
        # 已移除用户管理页“可用算法”表单；保持历史值不被误清空
        if 'algo_mode' in request.POST:
            allowed_json = _parse_allowed_types(request)
        elif edit_obj is not None:
            allowed_json = edit_obj.allowed_model_types or ''
        else:
            allowed_json = ''

        def _rerender_form():
            ctx = _admin_ctx(request)
            ctx.update(
                {
                    'edit_obj': edit_obj,
                    'role_choices': User.ROLE_CHOICES,
                    'form_post': request.POST,
                }
            )
            return render(request, 'workbench/user_form.html', ctx)

        if not username or not tel:
            messages.error(request, '昵称和手机号为必填项')
            return _rerender_form()

        if edit_obj is None:
            if not pwd:
                messages.error(request, '新建用户必须设置登录密码')
                return _rerender_form()
            if User.objects.filter(tel=tel).exists():
                messages.error(request, '该手机号已被注册')
                return _rerender_form()
        else:
            if User.objects.filter(tel=tel).exclude(pk=edit_obj.pk).exists():
                messages.error(request, '该手机号已被其他用户使用')
                return _rerender_form()

        if edit_obj and edit_obj.id == actor.id:
            if role != 'admin':
                messages.error(request, '不能取消自己的管理员角色')
                return redirect(reverse('user_mgmt_edit', args=[edit_obj.pk]))
            if not is_active:
                messages.error(request, '不能停用自己的账号')
                return redirect(reverse('user_mgmt_edit', args=[edit_obj.pk]))

        if edit_obj and edit_obj.role == 'admin' and edit_obj.is_active:
            if role != 'admin' or not is_active:
                others = User.objects.filter(role='admin', is_active=True).exclude(pk=edit_obj.pk).count()
                if others == 0:
                    messages.error(request, '系统中至少需要一名已启用的管理员')
                    return redirect(reverse('user_mgmt_edit', args=[edit_obj.pk]))

        if edit_obj is None:
            User.objects.create(
                username=username,
                tel=tel,
                pwd=pwd,
                email=email,
                role=role,
                is_active=is_active,
                admin_note=admin_note,
                sex=sex,
                address=address,
                allowed_model_types=allowed_json,
            )
            messages.success(request, '用户已创建')
        else:
            edit_obj.username = username
            edit_obj.tel = tel
            edit_obj.email = email
            edit_obj.role = role
            edit_obj.is_active = is_active
            edit_obj.admin_note = admin_note
            edit_obj.sex = sex
            edit_obj.address = address
            edit_obj.allowed_model_types = allowed_json
            if pwd:
                edit_obj.pwd = pwd
            edit_obj.save()
            messages.success(request, '用户信息已保存')

        return redirect(reverse('user_mgmt_list'))


@method_decorator(check_admin_access, name='dispatch')
class UserMgmtDeleteView(View):
    def post(self, request, pk):
        actor = get_admin_panel_user(request)
        target = get_object_or_404(User, pk=pk)
        if target.id == actor.id:
            messages.error(request, '不能删除当前登录账号')
            return redirect(reverse('user_mgmt_list'))
        if target.role == 'admin':
            messages.error(request, '不能直接删除管理员账号，请先将其角色改为非管理员')
            return redirect(reverse('user_mgmt_list'))
        name = target.username
        target.delete()
        messages.success(request, '已删除用户：{}'.format(name))
        return redirect(reverse('user_mgmt_list'))


@method_decorator(check_admin_access, name='dispatch')
class ModelPlanListView(View):
    def get(self, request):
        ctx = _admin_ctx(request)
        ctx['plans'] = ModelPlan.objects.all().order_by('-id')
        return render(request, 'workbench/plan_list.html', ctx)


@method_decorator(check_admin_access, name='dispatch')
class ModelPlanDeleteView(View):
    def post(self, request, pk):
        plan = get_object_or_404(ModelPlan, pk=pk)
        name = plan.name
        bind_cnt = UserPlan.objects.filter(plan=plan).count()
        plan.delete()
        if bind_cnt > 0:
            messages.success(request, '已删除方案「{}」，并移除 {} 条用户套餐绑定记录'.format(name, bind_cnt))
        else:
            messages.success(request, '已删除方案「{}」'.format(name))
        return redirect(reverse('model_plan_list'))


@method_decorator(check_admin_access, name='dispatch')
class ModelPlanFormView(View):
    def get(self, request, pk=None):
        plan = get_object_or_404(ModelPlan, pk=pk) if pk else None
        ctx = _admin_ctx(request)
        ctx.update(
            {
                'plan': plan,
                'model_registrations': list(_published_model_registrations_qs()),
                **_plan_algo_form_context(plan),
            }
        )
        return render(request, 'workbench/plan_form.html', ctx)

    def post(self, request, pk=None):
        plan = get_object_or_404(ModelPlan, pk=pk) if pk else None
        name = (request.POST.get('name') or '').strip()
        description = (request.POST.get('description') or '').strip()
        is_active = request.POST.get('is_active') == 'on'
        total_quota = int(request.POST.get('total_quota') or 0)
        plan_algo_mode = (request.POST.get('plan_algo_mode') or 'all').strip()
        valid_ids = set(_published_model_registrations_qs().values_list('id', flat=True))
        picked_ids = []
        seen = set()
        for x in request.POST.getlist('model_reg_id'):
            if str(x).isdigit():
                xi = int(x)
                if xi in valid_ids and xi not in seen:
                    seen.add(xi)
                    picked_ids.append(xi)
        allowed_types_raw = ''
        allowed_ids_raw = ''
        if plan_algo_mode == 'pick':
            allowed_ids_raw = json.dumps(picked_ids, ensure_ascii=False)
        if not name:
            messages.error(request, '方案名称不能为空')
            return redirect(reverse('model_plan_list'))
        if plan_algo_mode == 'pick' and not picked_ids:
            messages.error(request, '已选择「仅允许勾选的模型登记」，请至少勾选一条「就绪 + 开放前台」的登记')
            ctx = _admin_ctx(request)
            ctx.update(
                {
                    'plan': plan,
                    'model_registrations': list(_published_model_registrations_qs()),
                    'plan_algo_mode': 'pick',
                    'plan_picked_ids': picked_ids,
                    'plan_has_legacy_types_only': False,
                }
            )
            return render(request, 'workbench/plan_form.html', ctx)
        data = {
            'name': name,
            'description': description[:255],
            'is_active': is_active,
            'total_quota': total_quota,
            'allowed_model_types_json': allowed_types_raw,
            'allowed_model_ids_json': allowed_ids_raw,
        }
        if plan is None:
            ModelPlan.objects.create(**data)
            messages.success(request, '方案已创建')
        else:
            for k, v in data.items():
                setattr(plan, k, v)
            plan.save()
            messages.success(request, '方案已更新')
        return redirect(reverse('model_plan_list'))


@method_decorator(check_admin_access, name='dispatch')
class UserPlanListView(View):
    def get(self, request):
        ctx = _admin_ctx(request)
        ctx['rows'] = UserPlan.objects.select_related('user', 'plan').order_by('-created_at')
        return render(request, 'workbench/user_plan_list.html', ctx)


@method_decorator(check_admin_access, name='dispatch')
class UserPlanDeleteView(View):
    def post(self, request, pk):
        up = get_object_or_404(UserPlan.objects.select_related('user', 'plan'), pk=pk)
        user_label = up.user.username if up.user else '未知用户'
        plan_label = up.plan.name if up.plan else '未知方案'
        up.delete()
        messages.success(request, '已删除用户套餐：{} / {}'.format(user_label, plan_label))
        return redirect(reverse('user_plan_list'))


@method_decorator(check_admin_access, name='dispatch')
class UserPlanFormView(View):
    def get(self, request, pk=None):
        up = get_object_or_404(UserPlan, pk=pk) if pk else None
        ctx = _admin_ctx(request)
        ctx.update(
            {
                'user_plan': up,
                'users': User.objects.filter(is_active=True).order_by('username', 'id'),
                # 后台绑定需看到全部方案；仅「启用」方案会过滤时，新建方案若未勾选「启用」会导致下拉为空
                'plans': ModelPlan.objects.all().order_by('-is_active', 'name', 'id'),
            }
        )
        return render(request, 'workbench/user_plan_form.html', ctx)

    def post(self, request, pk=None):
        up = get_object_or_404(UserPlan, pk=pk) if pk else None
        user_id = (request.POST.get('user_id') or '').strip()
        plan_id = (request.POST.get('plan_id') or '').strip()
        remaining_quota_raw = (request.POST.get('remaining_quota') or '').strip()
        status = request.POST.get('status') or 'pending'

        if not user_id.isdigit() or not plan_id.isdigit():
            messages.error(request, '用户与方案为必选项')
            return redirect(reverse('user_plan_list'))

        user = User.objects.filter(id=int(user_id)).first()
        plan = ModelPlan.objects.filter(id=int(plan_id)).first()
        if not user or not plan:
            messages.error(request, '用户或方案不存在')
            return redirect(reverse('user_plan_list'))

        # 留空时自动使用方案总额度
        if remaining_quota_raw == '':
            remaining_quota = int(plan.total_quota or 0)
        else:
            remaining_quota = int(remaining_quota_raw or 0)

        if up is None:
            UserPlan.objects.create(
                user=user,
                plan=plan,
                remaining_quota=remaining_quota,
                status=status,
            )
            messages.success(request, '用户套餐已创建')
        else:
            up.plan = plan
            up.remaining_quota = remaining_quota
            up.status = status
            up.save()
            messages.success(request, '用户套餐已更新')
        return redirect(reverse('user_plan_list'))
