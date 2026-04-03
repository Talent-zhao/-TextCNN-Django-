from django.contrib import messages
from django.shortcuts import render, redirect, HttpResponse
from django.http import JsonResponse
from django.urls import reverse
from django.db.models import Sum, Count, Min, Max, Q, Avg
from django.db.models import Q,F
from django.views import View
from django.utils.decorators import method_decorator
from django.utils import timezone
import  os
import time
import datetime
import json
import pandas as pd
import numpy as np
from app.models import *

# 验证登录状态装饰器
def check_login(func):
    def wrapper(request, *args, **kwargs):
        user_raw = request.get_signed_cookie('user', salt='salt', default='')
        if not user_raw:
            return redirect(reverse('login'))
        try:
            data = json.loads(user_raw)
            uid = data.get('uid') if isinstance(data, dict) else None
            if uid is None:
                return redirect(reverse('login'))
            if not User.objects.filter(id=int(uid), is_active=True).exists():
                return redirect(reverse('login'))
        except (TypeError, ValueError, json.JSONDecodeError):
            return redirect(reverse('login'))
        return func(request, *args, **kwargs)
    return wrapper


def check_admin_access(func):
    """
    仅承认「管理员登录」下发的 admin 签名 Cookie。
    前台 user Cookie 即使角色为管理员也不可进入 admin_panel，须单独在 /admin_panel/login/ 登录，避免两套会话混用。
    """
    def wrapper(request, *args, **kwargs):
        admin_raw = request.get_signed_cookie('admin', salt='admin_salt', default='')
        if not admin_raw:
            return redirect(reverse('admin_login'))
        try:
            ad = json.loads(admin_raw)
            if not isinstance(ad, dict):
                return redirect(reverse('admin_login'))
        except (TypeError, ValueError, json.JSONDecodeError):
            return redirect(reverse('admin_login'))
        return func(request, *args, **kwargs)
    return wrapper

# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

def get_admin_panel_user(request):
    """
    后台模板中的 userinfo：仅根据 admin_panel 的「admin」Cookie 解析。
    与前台「user」Cookie 完全分离；Django 超级用户登录时 admin.uid 可能为 auth.User 主键，按 username 匹配业务 User。
    """
    def _load_admin_cookie():
        try:
            raw = request.get_signed_cookie('admin', salt='admin_salt', default='')
            if not raw:
                return None
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    ad = _load_admin_cookie()
    if ad:
        auid = ad.get('uid')
        if auid is not None:
            try:
                u = User.objects.filter(id=int(auid), is_active=True).first()
                if u:
                    return u
            except (TypeError, ValueError):
                pass
        uname = ad.get('username')
        if uname:
            u = User.objects.filter(username=uname, is_active=True).first()
            if u:
                return u

    from django.core.exceptions import PermissionDenied

    raise PermissionDenied('登录态无效或已过期，请重新登录后台。')


# ================================================= #
# ****************** 登录   ******************* #
# ================================================= #
class loginView(View):
    def get(self,request):
        return render(request,'user/login.html',locals())
    def post(self,request):
        data = request.POST
        tel, pwd = data.get('tel'), data.get('pwd')
        user = User.objects.filter(tel=tel, pwd=pwd).first()
        if user:
            if not user.is_active:
                msg = '该账号已被禁用，请联系管理员。'
                messages.error(request, msg)
                return render(request, 'user/login.html', locals())
            obj = redirect('index')
            obj.set_signed_cookie('user', json.dumps({
                'username': user.username,
                'uid': user.id,
            }), max_age=60 * 60 * 24, salt='salt')
            return obj
        msg = '账号信息输入错误，请重新检查！'
        messages.error(request, msg)
        return render(request, 'user/login.html', locals())

# 注册页面
class registerViews(View):
    def get(self, request):
        return render(request, 'user/register.html', locals())

    def post(self, request):
        data = request.POST
        tel, pwd,  username = data.get('tel'), data.get('pwd'),  data.get('username')
        # 如果用户不存在 则注册
        if not User.objects.filter(tel=tel):
            User.objects.create(tel=tel, username=username, pwd=pwd)
            msg = '账号注册成功！请登录吧！'
            messages.info(request, msg)
            return redirect('login')
        # 否则就是用户已经存在了
        else:
            msg = '该手机号已经存在了，请检查!'
            messages.error(request, msg)
            return render(request, 'user/register.html', locals())


# zhux注销登录 返回首页
def logout(request):
    obj = redirect(reverse('index'))
    obj.delete_cookie('user')
    return obj


# 个人信息修改
@method_decorator(check_login, name='dispatch')
class myProfileView(View):
    def _current_user(self, request):
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        return User.objects.get(id=user['uid'])

    def _active_user_plan(self, userinfo):
        now = timezone.now()
        qs = UserPlan.objects.filter(user=userinfo, status='active').select_related('plan').order_by('-updated_at', '-id')
        for up in qs:
            if up.start_time and up.start_time > now:
                continue
            if up.end_time and up.end_time < now:
                continue
            if up.plan and up.plan.is_active is False:
                continue
            return up
        return None

    def _allowed_algo_display_list(self, userinfo, user_plan):
        from app.plan_allowance import parse_plan_id_allow, parse_plan_type_allow

        qs = ModelInfo.objects.filter(status='ready', listed_for_users=True).order_by('model_type', 'name', 'id')
        user_limit = userinfo.allowed_model_types_set()
        if user_limit is not None:
            qs = qs.filter(model_type__in=user_limit)

        if user_plan and user_plan.plan:
            p = user_plan.plan
            id_allow = parse_plan_id_allow(p.allowed_model_ids_json)
            if id_allow is not None and len(id_allow) > 0:
                qs = qs.filter(id__in=id_allow)
            else:
                t_allow = parse_plan_type_allow(p.allowed_model_types_json)
                if t_allow is not None:
                    if len(t_allow) == 0:
                        qs = qs.none()
                    else:
                        qs = qs.filter(model_type__in=t_allow)

        type_dict = dict(ModelInfo.MODEL_TYPE_CHOICES)
        out = []
        for mi in qs:
            out.append(
                {
                    'key': str(mi.id),
                    'name': '{} / {} — {}'.format(
                        mi.name,
                        mi.version,
                        type_dict.get(mi.model_type, mi.model_type),
                    ),
                }
            )
        return out

    def get(self, request):
        userinfo = self._current_user(request)
        active_user_plan = self._active_user_plan(userinfo)
        available_plans = ModelPlan.objects.filter(is_active=True).order_by('name', 'id')
        allowed_algo_list = self._allowed_algo_display_list(userinfo, active_user_plan)
        return render(request, 'user/myProfile.html', locals())

    def post(self, request):
        userinfo = self._current_user(request)
        pk = int(userinfo.id)
        action = (request.POST.get('action') or 'update_profile').strip()

        if action == 'virtual_recharge':
            active_up = self._active_user_plan(userinfo)
            if active_up is None:
                messages.error(request, '当前没有可用套餐，无法购买额度；请先选择套餐。')
                return redirect('myProfile')
            try:
                buy_count = int(request.POST.get('buy_count') or 0)
            except (TypeError, ValueError):
                buy_count = 0
            if buy_count <= 0:
                messages.error(request, '购买次数必须大于 0')
                return redirect('myProfile')
            before = int(active_up.remaining_quota or 0)
            active_up.remaining_quota = before + buy_count
            active_up.save(update_fields=['remaining_quota', 'updated_at'])
            messages.success(request, '虚拟购买成功：额度 +{}，当前剩余 {}'.format(buy_count, active_up.remaining_quota))
            return redirect('myProfile')

        if action == 'virtual_change_plan':
            plan_id = (request.POST.get('plan_id') or '').strip()
            if not plan_id.isdigit():
                messages.error(request, '请选择要切换的套餐')
                return redirect('myProfile')
            target_plan = ModelPlan.objects.filter(id=int(plan_id), is_active=True).first()
            if not target_plan:
                messages.error(request, '目标套餐不存在或未启用')
                return redirect('myProfile')

            now = timezone.now()
            # 将现有 active 套餐置为 expired，避免并行冲突
            UserPlan.objects.filter(user=userinfo, status='active').update(status='expired', end_time=now)
            UserPlan.objects.create(
                user=userinfo,
                plan=target_plan,
                remaining_quota=int(target_plan.total_quota or 0),
                start_time=now,
                status='active',
            )
            messages.success(request, '虚拟切换套餐成功：{}（额度重置为 {}）'.format(target_plan.name, target_plan.total_quota))
            return redirect('myProfile')

        data = {
            'username': (request.POST.get('username') or '').strip(),
            'sex': (request.POST.get('sex') or '').strip(),
            'email': (request.POST.get('email') or '').strip(),
            'tel': (request.POST.get('tel') or '').strip(),
            'address': (request.POST.get('address') or '').strip(),
        }
        if not data['username'] or not data['tel']:
            messages.error(request, '用户名和手机号不能为空')
            return redirect('myProfile')

        if User.objects.filter(tel=data['tel']).exclude(id=pk).exists():
            messages.error(request, '手机号已被其他账号占用')
            return redirect('myProfile')

        User.objects.filter(id=pk).update(**data)

        avatar = request.FILES.get('avatar')
        if avatar:
            os.makedirs(os.path.join('media', 'avatar'), exist_ok=True)
            filename = os.path.join('media', 'avatar', avatar.name)
            savename = os.path.join('avatar', avatar.name)
            f = open(filename, 'wb')
            for line in avatar.chunks():
                f.write(line)
            f.close()
            User.objects.filter(id=pk).update(avatar=savename)

        messages.info(request, '信息修改成功！')
        obj = redirect('myProfile')
        obj.set_signed_cookie(
            'user',
            json.dumps({'username': data['username'], 'uid': pk}),
            max_age=60 * 60 * 24,
            salt='salt',
        )
        return obj


# 修改密码
# @method_decorator(check_login,name='get') #
class resetPWdView(View):
    def get(self, request):
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        userinfo = User.objects.get(id=user['uid'])
        return render(request, 'user/resetPWd.html', locals())

    def post(self, request):
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        data = request.POST
        data = dict(data)
        print(data)
        # 得到表单参数
        for k, v in data.items():
            data[k] = v[0]
        pk = int(data['pk'])
        del data['pk']
        # 旧密码不对
        if User.objects.get(id=user['uid']).pwd != data['pwd']:
            messages.info(request, '您输入的旧密码不正确！')
            return render(request, 'user/resetPWd.html', locals())
        else: # 密码对了
            User.objects.filter(id=user['uid']).update(pwd = data['newpwd'])
            messages.info(request, '密码修改成功,请登录')
            return redirect('login')


@check_login
def role_debug(request):
    user_cookie = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    uid = user_cookie.get('uid')
    userinfo = User.objects.filter(id=uid).first()
    role = userinfo.role if userinfo else 'user'
    is_admin = bool(userinfo and role == 'admin')
    return JsonResponse({
        'uid': uid,
        'username': userinfo.username if userinfo else '',
        'tel': userinfo.tel if userinfo else '',
        'role': role,
        'role_is_admin': is_admin,
        'can_access_admin_panel_with_user_cookie_only': False,
        'hint': '后台与前台登录已分离：即使角色为管理员，也须在 /admin_panel/login/ 单独登录后持有 admin Cookie 方可进入 admin_panel',
    })
