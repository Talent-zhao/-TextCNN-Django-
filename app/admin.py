from django.contrib import admin
from app.models import *
# admin.py
admin.site.site_header = '管理系统' #  登录显示
admin.site.site_title = '管理系统' # title
admin.site.index_title = '管理系统' #

# Register your models here.
@admin.register(User)
#admin.site.register(User,  UserAdmin)
class UserAdmin(admin.ModelAdmin):
    # list_display用于设置列表页面要显示的不同字段
    list_display = ['id','username','tel','pwd']
    # search_fields用于设置搜索栏中要搜索的不同字段
    search_fields = ['id','username','tel','pwd']
    # 设置过滤器，在后台数据的右侧生成导航栏，如有外键应使用双下划线连接两个模型的字段
    list_filter = ['id','username','tel','pwd']


from django.contrib.auth.models import Group, User
admin.site.unregister(Group)
admin.site.unregister(User)