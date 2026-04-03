from django.contrib import messages
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from django.utils.decorators import method_decorator
import json
from functools import wraps
from app.models import User


def check_admin_login(func):
    """
    Django 4.x 下用于 CBV 的登录校验装饰器。
    注意：这里的 wrapper 不能带 self 参数，否则配合 method_decorator 会出现参数错位。
    """
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        admin = request.get_signed_cookie('admin', salt='admin_salt', default='')
        if not admin:
            return redirect(reverse('admin_login'))
        return func(request, *args, **kwargs)
    return wrapper


class AdminLoginView(View):
    def get(self, request):
        return render(request, 'admin/login.html', locals())

    def post(self, request):
        data = request.POST
        username = data.get('username')
        password = data.get('password')

        from django.contrib.auth import authenticate
        user = authenticate(username=username, password=password)

        if user and user.is_superuser:
            obj = redirect('user_mgmt_list')
            obj.set_signed_cookie('admin', json.dumps({
                'username': user.username,
                'uid': user.id,
            }), max_age=60 * 60 * 24, salt='admin_salt')
            return obj
        user_obj = User.objects.filter(username=username, pwd=password).first()
        if user_obj:
            if not user_obj.is_active:
                msg = '该账号已被禁用'
                messages.error(request, msg)
                return render(request, 'admin/login.html', locals())
            if user_obj.role == 'admin' or user_obj.username == 'admin':
                obj = redirect('user_mgmt_list')
                obj.set_signed_cookie('admin', json.dumps({
                    'username': user_obj.username,
                    'uid': user_obj.id,
                }), max_age=60 * 60 * 24, salt='admin_salt')
                return obj

        msg = '管理员账号或密码错误！'
        messages.error(request, msg)
        return render(request, 'admin/login.html', locals())


def admin_logout(request):
    obj = redirect(reverse('admin_login'))
    obj.delete_cookie('admin')
    return obj
