import pymysql


class ConnectMysql():
    def __init__(self, host='127.0.0.1', port=3306, user='root', passwd='123456', db=''):
        """
        host='localhost', port=3306, user='root', passwd='', db=''
        :param host:
        :param port:
        :param user:
        :param passwd:
        :param db:
        """
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        # cursorclass = pymysql.cursors.DictCursor,

    def get_conn(self):
        """
        获取连接对象conn，建立数据库的连接
        db:表示数据库名称
        host 访问服务端的ip，user 访问服务端的用户，passwd访问服务端的用户密码，
        db 访问服务端的数据库，charset 访问时采用的编码方式）
        :return:
        """
        # cursorclass = pymysql.cursors.DictCursor,
        conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                               passwd=self.passwd, db=self.db)
        return conn

    def insert(self, sql, args):
        """
        插入一一条数据
        sql = 'INSERT INTO student VALUES(%s,%s,%s);'
        insert(sql, ('2', 'wang', '10'))
        """
        conn = self.get_conn()
        cur = conn.cursor()
        result = cur.execute(sql, args)
        # print("插入{}条数据".format(result))
        conn.commit()
        cur.close()
        conn.close()

    def insert_many(self, sql, args):
        """插入多条数据
        sql = 'insert into student VALUES (%s,%s,%s)'
        args = [('003', 'li', '11'), ('004', 'sun', '12'), ('005', 'zhao', '13')]
        insert_many(sql=sql, args=args)
        """
        conn = self.get_conn()
        cur = conn.cursor()
        result = cur.executemany(query=sql, args=args)
        # print("插入{}条数据".format(result))
        conn.commit()
        cur.close()
        conn.close()

    def update(self, sql, args):
        """
        更新数据库
        sql = 'UPDATE student SET Sname=%s WHERE Sno = %s;'
        args = ('wangprince', '2')
        update(sql, args)
        :param sql:
        :param args:
        :return:
        """
        conn = self.get_conn()
        cur = conn.cursor()
        result = cur.execute(sql, args)
        # print("更新了{}条数据".format(result))
        conn.commit()
        cur.close()
        conn.close()

    def delete(self, sql, args):
        """
        删除数据库
        :param sql:
        :param args:
        sql = 'DELETE FROM student WHERE Sno = %s;'
        args = ('2',)  # 单个元素的tuple写法
        delete(sql, args)
        :return:
        """
        conn = self.get_conn()
        cur = conn.cursor()
        result = cur.execute(sql, args)
        # print("删除{}条数据".format(result))
        conn.commit()
        cur.close()
        conn.close()

    def execute(self,sql):
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        results = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()
    def query(self, sql, args):
        """
        查询
        sql = 'SELECT  * FROM student;'
        query(sql, None)
        :param sql:
        :param args:
        :return:
        """
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(sql, args)
        results = cur.fetchall()
        # for row in results:
        #     id = row[0]
        #     name = row[1]
        #     age = row[2]
        conn.commit()
        cur.close()
        conn.close()
        return results

    def get_all_db(self):
        """
        获取所有数据库名
        :return: list
        """
        # 排除自带的数据库
        exclude_list = ["sys", "information_schema", "mysql", "performance_schema"]
        sql = "show databases"  # 显示所有数据库
        res = self.query(sql, args=())

        # print(res)
        if not res:  # 判断结果非空
            return False
        db_list = []  # 数据库列表
        for i in res:
            db_name = i[0]
            # 判断不在排除列表时
            if db_name not in exclude_list:
                db_list.append(db_name)

        # if not db_list:
        #     return False

        return db_list

    def get_all_table(self):
        """
        获取所有数据库名
        :return: list
        """

        sql = "show tables"  # 显示所有数据库
        res = self.query(sql, args=())
        table_list = [tuple[0] for tuple in res]

        return table_list

    def get_table_names(self, tabls_name):
        """
        获取表名
        :param tabls_name:
        :return:
        """
        sql = "select * from `%s`" % tabls_name  # 显示所有数据库
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        col_name_list = [tuple[0] for tuple in cursor.description]

        return col_name_list

    def create_db(self, name, zifu, paixu):
        """
        con = ConnectMysql()
        con.create_db('test', 'utf8mb4', 'utf8mb4_general_ci')
        根据 创建数据库
        :param name: 名字
        :param zifu: 字符
        :param paixu: 排序规则
        :return:
        """
        # 例子 create database test2 DEFAULT CHARACTER SET gbk COLLATE gbk_chinese_ci;
        sql = "create database %s DEFAULT CHARACTER SET %s COLLATE %s;" % (name, zifu, paixu)
        res = self.query(sql, args=())

    def delete_db(self, dbname):
        """
        删除数据库
        :param dbname:
        :return:
        """
        # 例子 DROP DATABASE test_db_del;
        sql = "DROP DATABASE IF EXISTS %s;" % (dbname)
        res = self.query(sql, args=())

    def query_db_charset(self, dbname):
        """
        查看数据库的编码 和排序规则
        :param dbname:
        :return:
        """
        self.db = dbname
        # show variables like 'character_set_database';
        sql = """
        SELECT SCHEMA_NAME,DEFAULT_CHARACTER_SET_NAME,DEFAULT_COLLATION_NAME FROM INFORMATION_SCHEMA.SCHEMATA;"""  # 查看字符编码
        res = self.query(sql, args=())

        return_data = tuple()
        for  item in res:
            if item[0] == dbname:
                return_data = (item[1],item[2])
        return return_data

    def update_db_zifu_paixu(self,dbname,zifu,paixu):
        # alter database aaaaaaaaaaaaaa character set utf8 collate utf8_general_ci;
        sql = "alter database %s character set %s collate %s;" % (dbname,zifu,paixu)
        res = self.query(sql, args=())


    def close_connect(self):
        """
        关闭链接
        :return:
        """
        conn = self.get_conn()
        conn.close()


""""
1 查重 字段类型
SELECT COLUMN_NAME,COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='app_aqi';
"""
if __name__ == '__main__':
    con = ConnectMysql()
    # con.create_db('test', 'utf8mb4', 'utf8mb4_general_ci')
    r = con.get_all_db()
    print(r)
    pass