"""
配置文件
"""
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sqlite数据库
# timeout：获取写锁前的等待秒数，减轻「database is locked」（多线程 runserver、后台导入、Navicat 等同时访问时）
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db', 'db.sqlite3'),
        'OPTIONS': {
            'timeout': 30,
        },
    }
}
# mysql数据库
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql', # 数据库引擎
#         'NAME': 'django_project', # 数据库名 必须填写
#         'USER': 'root', # 账号
#         'PASSWORD': '123456', # 密码
#         'HOST': '127.0.0.1', # HOST
#         'POST': 3306, # 端口
#     }
# }
# 如果是mysql 没有创建系统数据库 自动执行下面创建数据库
if DATABASES['default']['ENGINE'].endswith('mysql'):
    db_name = DATABASES['default']['NAME']
    print(f"mysql数据库名字:{db_name}")
    from utils.connect_mysql import ConnectMysql
    con = ConnectMysql(passwd='123456')
    if db_name not in con.get_all_db():
        # con.delete_db(db_name)
        con.create_db(db_name, 'utf8mb4', 'utf8mb4_general_ci')


