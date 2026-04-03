from django.apps import AppConfig


def _sqlite_pragmas(sender, connection, **kwargs):
    if connection.vendor != 'sqlite':
        return
    with connection.cursor() as cursor:
        # 读写并发更好，显著减少开发环境下「database is locked」
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA synchronous=NORMAL;')
        cursor.execute('PRAGMA busy_timeout=30000;')


class DemoConfig(AppConfig):
    name = 'app'
    verbose_name = 'app'

    def ready(self):
        from django.db.backends.signals import connection_created

        connection_created.connect(_sqlite_pragmas)
