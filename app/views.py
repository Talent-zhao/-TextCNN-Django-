# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

import jieba
import joblib
from django.conf import settings
from django.shortcuts import render, redirect, HttpResponse
from django.http import JsonResponse
from django.urls import reverse
from django.db.models import Count
from django.views import View
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.core.paginator import Paginator
import os
import json
import hashlib
import datetime as dt
import pandas as pd
from tqdm import tqdm
from app.models import *
from app.scrawldata import scrawl_tieba
from app.userViews import check_login

from algorithm.model_paths import (
    NUM2NAME_REL,
    legacy_sklearn_model_rel,
    sklearn_model_rel,
    sklearn_tfidf_primary_rel,
)


def _front_user_from_cookie(request):
    """解析前台 user 签名 Cookie，缺少 uid 或用户不存在时返回 None。"""
    try:
        raw = request.get_signed_cookie('user', salt='salt', default='{}')
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return None
        uid = data.get('uid')
        if uid is None:
            return None
        return User.objects.filter(id=int(uid), is_active=True).first()
    except Exception:
        return None


def _write_prediction_usage_log(userinfo, model_info, content, success, message):
    try:
        PredictionUsageLog.objects.create(
            user=userinfo if userinfo else None,
            model_info=model_info if model_info else None,
            input_length=len(content or ''),
            success=bool(success),
            message=(message or '')[:255],
        )
    except Exception:
        # 日志失败不影响主流程
        pass


def _user_active_plan(userinfo):
    """当前前台用户生效中的 UserPlan（与 personal center 逻辑一致）。"""
    now = timezone.now()
    qs = (
        UserPlan.objects.filter(user=userinfo, status='active')
        .select_related('plan')
        .order_by('-updated_at', '-id')
    )
    for up in qs:
        if up.start_time and up.start_time > now:
            continue
        if up.end_time and up.end_time < now:
            continue
        pl = up.plan
        if pl is not None and not pl.is_active:
            continue
        return up
    return None


def _predict_models_for_user_plan(userinfo, user_plan):
    """
    在线预测可选模型：已发布 + 用户可用算法 + 当前套餐允许的模型登记（ID）或兼容旧数据按类型。
    """
    from app.plan_allowance import model_info_passes_plan

    if user_plan is None or user_plan.plan is None:
        return []
    published_qs = ModelInfo.objects.filter(
        status='ready', listed_for_users=True
    ).order_by('-is_active', '-created_at')
    candidates = userinfo.filter_models_for_user(published_qs)
    plan = user_plan.plan
    return [m for m in candidates if model_info_passes_plan(m, plan)]


def init_user():
    # 初始化普通用户
    from app import models
    if not models.User.objects.filter(tel="user", pwd="123456"):
        models.User.objects.create(tel="user", pwd="123456", username="user")
    # 初始化管理员
    from django.contrib.auth.models import User
    if not User.objects.filter(username='admin'):
        User.objects.create_superuser('admin', 'admin@qq.com', 'admin')
def clean_str(string):
    """
    对数据集中的字符串做清洗.
    """
    import re
    # 去除标点符号
    string = re.sub("[\s+\.\!\/\+\"\_\[\]\-@,:~$%^*(+'；、：“”．+|——！，。？?、~@#￥%……&*（）]+", "", string)
    # 去除网页
    string = re.sub(r"<[^>].*?>|&.*?;", "", string)
    # 去除数字和英文
    string = re.sub("[a-zA-Z0-9]", "", string)
    return string.strip().lower()
def get_stopword():
    """
    获取停用词
    """
    stopwords = [item.strip() for item in open('stopwords.txt', 'r',
                                           encoding='utf-8').readlines()]
    return stopwords
def sent2word(line):
    """
    jieba分词
    """
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    segList = jieba.cut(line.strip(),cut_all=False)
    segSentence = ''
    for word in segList:
        # 去除停用词
        if word != '\t' and word not in stopwords:
            segSentence += word + " "
    return segSentence.strip()


def read_json(filename):
    """
    从一个json文件读取数据
    """
    return json.load(open(filename,"r",encoding="utf-8"))
def fenlei():
    """
    兼容旧数据链路：对 Comment.emotion 为空的数据做旧版分类并回写 emotion 字段。
    新导入贴吧数据的主预测链路应走 RawText + PredictService + PredictionResult。
    :return:
    """
    # Comment.objects.all().update(emotion='')
    id_list = []
    texts = []
    for i in Comment.objects.filter(emotion=''):
        id_list.append(i.id)
        texts.append(i.content)
    # 没有未分类的 返回
    if len(id_list)==0:
        return

    content_list = [sent2word(clean_str(line)) for line in texts]
    # 模型加载（优先 model/svm/，兼容旧版扁平路径）
    base = settings.BASE_DIR

    def _proj(*parts):
        return os.path.join(base, *parts)

    mpath = _proj(*sklearn_model_rel('svm').split('/'))
    if not os.path.isfile(mpath):
        mpath = _proj(*legacy_sklearn_model_rel('svm').split('/'))
    vpath = _proj(*sklearn_tfidf_primary_rel('svm').split('/'))
    if not os.path.isfile(vpath):
        vpath = _proj('model', 'tfidfVectorizer.pkl')
    npath = _proj(*NUM2NAME_REL.split('/'))
    model = joblib.load(mpath)
    tv = joblib.load(vpath)
    num2name = read_json(npath)
    print(num2name)
    predata = tv.transform(content_list)
    result_list = model.predict(predata)
    for pk, emotion in tqdm(zip(id_list, result_list)):

        emotion = str(emotion)
        # print(emotion)
        emotion = num2name[emotion]
        Comment.objects.filter(id=pk).update(emotion=emotion)
    print("分类完成！更新数据库")

    # 一开始的占位符
    # print("开始预测中")
    # labels = ['积极','消极','中性']
    # labels = [labels[ i%3 ]   for i in range(len(id_list))]
    # for id, label in tqdm(zip(id_list, labels)):
    #
    #     Comment.objects.filter(id=id).update(emotion=label)
    # print("情感分类结束")
@check_login
# 注意：旧版本在模块导入时会自动执行 fenlei()，会导致服务启动阶段阻塞。
# 现改为显式触发，避免耦合。
def trigger_fenlei(request):
    # 兼容入口：仅用于历史 Comment 数据补分类，非新链路主入口。
    fenlei()
    return JsonResponse({'status': 1, 'msg': '旧链路兼容分类任务执行完成（Comment.emotion）'})
# 首页
@method_decorator(check_login,name='get') #
class IndexView(View):
    def get(self,request):
        # 用户信息
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        # 获取用户的obj
        userinfo = User.objects.get(id=user["uid"])
        # 用户id  用户名
        uid,username = userinfo.id,userinfo.username
        # 所有数据（按最新入库优先，保证刚爬取的数据立即出现在评论列表前页）
        all_qs = Comment.objects.all().order_by('-id')

        # 分页
        # 总数据
        total = all_qs.count()
        pagenum = request.GET.get("pagenum", 1)
        pagenum = int(pagenum)

        pageSize = 20
        # 开始
        begin = (pagenum - 1) * pageSize
        # 结束
        end = (pagenum - 0) * pageSize
        data_list = list(all_qs[begin:end])

        # 稳定关联：Comment -> CommentRawTextMap -> RawText -> PredictionResult(最新一条)
        comment_ids = [c.id for c in data_list]
        map_rows = (
            CommentRawTextMap.objects
            .filter(comment_id__in=comment_ids)
            .select_related('raw_text')
        )
        comment_to_raw = {m.comment_id: m.raw_text for m in map_rows}

        raw_ids = [rt.id for rt in comment_to_raw.values()]
        latest_pred_map = {}
        if raw_ids:
            pred_rows = (
                PredictionResult.objects
                .filter(raw_text_id__in=raw_ids)
                .select_related('predicted_label', 'model_info')
                .order_by('raw_text_id', '-created_at')
            )
            for pr in pred_rows:
                if pr.raw_text_id not in latest_pred_map:
                    latest_pred_map[pr.raw_text_id] = pr

        for c in data_list:
            rt = comment_to_raw.get(c.id)
            pr = latest_pred_map.get(rt.id) if rt else None
            if pr:
                c.predicted_label_name = pr.predicted_label.name if pr.predicted_label else '未标注'
                c.prediction_risk_level = pr.risk_level
                c.prediction_final_score = pr.final_risk_score
                c.prediction_hit_keywords = pr.hit_keywords or ''
                c.prediction_model_name = pr.model_info.name if pr.model_info else ''
                c.prediction_model_version = pr.model_info.version if pr.model_info else ''
            else:
                # 兼容旧历史数据：优先旧 emotion，再回退“未分析”
                c.predicted_label_name = c.emotion or '未分析'
                c.prediction_risk_level = ''
                c.prediction_final_score = None
                c.prediction_hit_keywords = ''
                c.prediction_model_name = ''
                c.prediction_model_version = ''


        return render(request,'index.html',locals())
    def post(self,request):
        return JsonResponse({'status':1,'msg':'操作成功'} )


def init(tid):
    """
    初始化贴吧导入：
    1) 保留写入 Comment（原始展示）
    2) 同步写入 RawText（统一预测输入）
    3) 批量触发 PredictService 生成 PredictionResult（统一预测结果）
    """
    filename = os.path.join("data", f"{tid}.csv")
    raw_data = pd.read_csv(filename)

    # 去重
    print(f"去重前的长度{len(raw_data)}")
    raw_data = raw_data.drop_duplicates()
    print(f"去重后的长度{len(raw_data)}")

    print(raw_data.columns)

    # 贴吧来源配置（统一预测链路使用）
    text_source, _ = TextSource.objects.get_or_create(
        name='贴吧抓取',
        source_type='tieba',
        source_key='tieba_thread',
        defaults={'is_enabled': True},
    )
    # 本次需要进入统一预测链路的 RawText ID
    pending_predict_rawtext_ids = []

    created_comments = 0
    created_rawtexts = 0

    for i in tqdm(range(len(raw_data))):
        row = raw_data.iloc[i]

        # ---------- time ----------
        time1 = ''
        time_date = None
        publish_time = None
        if pd.notna(row["time"]):
            raw_time = str(row["time"]).strip()
            time1 = raw_time.split()[0]
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    publish_time = dt.datetime.strptime(raw_time, fmt)
                    break
                except Exception:
                    continue
            if publish_time is None:
                try:
                    publish_time = dt.datetime.strptime(time1, "%Y-%m-%d")
                except Exception:
                    publish_time = None
        if publish_time is not None:
            time_date = publish_time.date()
        else:
            try:
                time_date = dt.datetime.strptime(time1, "%Y-%m-%d").date() if time1 else dt.date.today()
            except Exception:
                time_date = dt.date.today()

        # ---------- floor（int） ----------
        floor = 0
        floor_raw = row["floor"]
        if pd.notna(floor_raw):
            if isinstance(floor_raw, str):
                s = floor_raw.strip()
                if "楼" in s:
                    num = s.replace("楼", "").strip()
                    floor = int(num) if num.isdigit() else 0
                elif s.isdigit():
                    floor = int(s)
            elif isinstance(floor_raw, (int, float)):
                try:
                    floor = int(floor_raw)
                except (ValueError, OverflowError):
                    floor = 0

        # ---------- level（int；空单元格常为 NaN） ----------
        level = 0
        level_raw = row["level"]
        if pd.notna(level_raw):
            if isinstance(level_raw, str):
                s = level_raw.strip()
                level = int(s) if s.isdigit() else 0
            elif isinstance(level_raw, (int, float)):
                try:
                    level = int(level_raw)
                except (ValueError, OverflowError):
                    level = 0

        # ---------- name ----------
        name = row["name"] if isinstance(row["name"], str) else ''

        # ---------- user_url ----------
        user_url = row["user_url"] if isinstance(row["user_url"], str) else ''

        # ---------- img ----------
        img = row["img"] if isinstance(row["img"], str) else ''
        if img and not img.startswith('https'):
            img = 'https:' + img

        # ---------- content ----------
        content = ''
        if pd.notna(row["content"]):
            content = ' '.join(str(row["content"]).split())

        # ---------- reply（int） ----------
        reply = 0
        reply_raw = row["reply"]
        if pd.notna(reply_raw):
            if isinstance(reply_raw, str):
                reply_raw = reply_raw.replace("回复", "").strip()
                if reply_raw.startswith("(") and reply_raw.endswith(")"):
                    reply_raw = reply_raw[1:-1]
                reply = int(reply_raw) if reply_raw.isdigit() else 0
            elif isinstance(reply_raw, (int, float)):
                try:
                    reply = int(reply_raw)
                except (ValueError, OverflowError):
                    reply = 0

        # ---------- ip ----------
        loc = ''
        ip_raw = row["ip"]
        if isinstance(ip_raw, str):
            loc = ip_raw.replace('IP属地:', '').strip()

        # ---------- sex ----------
        sex = ''
        if "sex" in raw_data.columns:
            sex_raw = row.get("sex")
            if isinstance(sex_raw, str):
                sex = sex_raw.strip()
            elif pd.notna(sex_raw):
                sex = str(sex_raw).strip()

        line = {
            'time': time_date,
            'floor': floor,
            'level': level,
            'name': name,
            'user_url': user_url,
            'img': img,
            'content': content,
            'reply': reply,
            'sex': sex,
            'loc': loc,
        }

        # RawText 去重标识：同 tid + 楼层 + 用户主页 + 文本内容
        dedup_hash = hashlib.sha256(
            "{}|{}|{}|{}".format(tid, floor, user_url, content).encode('utf-8')
        ).hexdigest()
        external_id = "tieba:{}:{}:{}".format(tid, floor, dedup_hash[:16])

        # 去重写入 / 复用 Comment
        comment_obj = Comment.objects.filter(
            time=time_date,
            floor=floor,
            user_url=user_url,
            name=name,
            content=content
        ).first()
        if comment_obj is None:
            comment_obj = Comment.objects.create(**line)
            created_comments += 1

        # 无论 Comment 是否已存在，都保证 RawText 同步存在（兼容重复导入）
        raw_obj = RawText.objects.filter(
            source=text_source,
            external_id=external_id,
        ).first()
        if raw_obj is None:
            raw_obj = RawText.objects.create(
                source=text_source,
                external_id=external_id,
                author_name=name or '',
                content=content or '',
                publish_time=publish_time,
                status=RawText.STATUS_PENDING_PREDICT,
                dedup_hash=dedup_hash,
            )
            created_rawtexts += 1

        # 建立稳定映射（幂等）
        map_obj, created = CommentRawTextMap.objects.get_or_create(
            comment=comment_obj,
            defaults={'raw_text': raw_obj},
        )
        if not created and map_obj.raw_text_id != raw_obj.id:
            map_obj.raw_text = raw_obj
            map_obj.save(update_fields=['raw_text'])

        # 重复预测控制：
        # - 仅当状态为 pending_predict 时加入批量预测
        # - 避免同一次导入重复加入同一 rawtext_id
        # 说明：run_prediction_for_rawtext 内还会再次做状态校验，形成双重保护。
        if raw_obj.status == RawText.STATUS_PENDING_PREDICT and raw_obj.id not in pending_predict_rawtext_ids:
            pending_predict_rawtext_ids.append(raw_obj.id)
# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

    # 导入完成后自动进入统一预测链路
    # 同步跑 TextCNN/Torch 在部分 Windows 环境会触发原生崩溃并直接结束 runserver；
    # 入库场景默认临时禁用 Torch，改用 sklearn/规则（与 PredictService 中 DISABLE_TORCH_MODELS 说明一致）。
    if pending_predict_rawtext_ids:
        from app.services.predict_service import PredictService

        allow_torch_on_import = os.environ.get("SCRAWL_ALLOW_TORCH_ON_IMPORT") == "1"
        prev_disable_torch = os.environ.get("DISABLE_TORCH_MODELS")
        try:
            if not allow_torch_on_import:
                os.environ["DISABLE_TORCH_MODELS"] = "1"
                print(
                    "贴吧入库：批量预测已临时跳过 TextCNN/Torch（防进程崩溃）；"
                    "需要导入时跑深度学习请设置 SCRAWL_ALLOW_TORCH_ON_IMPORT=1"
                )
            PredictService.run_batch_prediction(pending_predict_rawtext_ids, model_info_id=None)
        finally:
            if not allow_torch_on_import:
                if prev_disable_torch is None:
                    os.environ.pop("DISABLE_TORCH_MODELS", None)
                else:
                    os.environ["DISABLE_TORCH_MODELS"] = prev_disable_torch
    return {
        'rows': int(len(raw_data)),
        'created_comments': int(created_comments),
        'created_rawtexts': int(created_rawtexts),
        'predict_count': int(len(pending_predict_rawtext_ids)),
    }
class scrawl(View):
    # get请求
    def get(self,request):
        # 用户信息
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        # 获取用户的obj
        userinfo = User.objects.get(id=user["uid"])
        # 用户id  用户名
        uid, username = userinfo.id, userinfo.username
        return render(request, 'scrawl.html', locals())

    # psot请求
    def post(self,request):
        # post 数据
        tid = request.POST.get('tid')

        # 爬数据
        try:
            scrawl_tieba(tid)
        except Exception as e:
            return JsonResponse(
                {
                    'code': 500,
                    'msg': '爬取失败：{}'.format(e),
                },
                status=500,
            )

        # 录入数据
        stats = init(tid)
        rows = int(stats.get('rows') or 0)
        if rows <= 0:
            return JsonResponse(
                {
                    'code': 400,
                    'msg': '抓取结果为空：未解析到可入库评论。请更换帖子ID或更新 Cookie 后重试。',
                },
                status=400,
            )

        return JsonResponse(
            {
                'data': 'data',
                'code': 200,
                'msg': '操作成功：抓取{}条，新增评论{}条，新增原始文本{}条，进入预测{}条'.format(
                    rows,
                    stats.get('created_comments', 0),
                    stats.get('created_rawtexts', 0),
                    stats.get('predict_count', 0),
                ),
                'count': rows,
                'redirect_url': reverse('index'),
            }
        )
def plot1(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    raw_data = Comment.objects.all()
    texts = [sent2word(i.content) for i in raw_data]
    texts_list = " ".join(texts).split()
    text2num = {i: texts_list.count(i)
                for i in set(texts_list)}
    data = [
        {
            "name": k,
            "value": v
        }
        for k, v in text2num.items()
    ]
    data = sorted(data, key=lambda x: x['value'], reverse=True)
    return render(request,'plot1.html',locals())

def plot2(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    raw_data = Comment.objects.all()

    tmp_data = [i for i in raw_data.order_by("time")]
    tmp_data = [f"{i.time.year}-{i.time.month}-{i.time.day}" for i in tmp_data]
    # 时间去重
    x_data = []
    for i in tmp_data:
        if i not in x_data:
            x_data.append(i)
    # 日期搞成整数
    x_data_date = [tuple(map(int, i.split("-"))) for i in x_data]

    y_data = [raw_data.filter(time__year=i[0], time__month=i[1], time__day=i[2]).count() for i in x_data_date]

    return render(request,'plot2.html',locals())

def plot3(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    from django.db.models import Count

    # 新链路：基于 PredictionResult 聚合（仅统计贴吧来源）
    group_data = (
        PredictionResult.objects
        .filter(raw_text__source__source_type='tieba')
        .values('predicted_label__name', 'risk_level')
        .annotate(count=Count('id'))
    )
    # 排序
    sort_data = sorted(group_data, key=lambda x: x['count'], reverse=True)
    # 折线图数据
    x_data = [
        i['predicted_label__name'] or '未标注标签(风险{})'.format(i['risk_level'])
        for i in sort_data
    ]
    y_data = [i['count'] for i in sort_data]
    # 饼图数据
    data = [
        {
            'name': item['predicted_label__name'] or '未标注标签(风险{})'.format(item['risk_level']),
            'value': item['count'],
        }
        for item in sort_data
    ]

    return render(request,'plot3.html',locals())

def plot4(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    # 获取省份数据
    data = Comment.objects.values('loc').annotate(count=Count('loc'))
    data = [
        {
            'provinceName': item['loc'],
            'shopCount': item['count'],
        }
        for item in data
    ]

    return render(request,'plot4.html',locals())

def plot5(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    from django.db.models import Sum, Count, Min, Max,Avg
    raw_data = Comment.objects.all()
    # 分组统计
    group_data = raw_data.values('name').annotate(count=Count('name'))
    # 排序
    sort_data = sorted(group_data,key=lambda x: x['count'],reverse=True)
    # 折线图数据
    x_data = [i['name'] for i in sort_data]
    y_data = [i['count'] for i in sort_data]

    return render(request,'plot5.html',locals())

def plot6(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    from django.db.models import Sum, Count, Min, Max, Avg
    raw_data = Comment.objects.all()
    # 分组统计
    group_data = raw_data.values('level').annotate(count=Count('level'))
    # 排序
    sort_data = sorted(group_data, key=lambda x: x['level'], reverse=False)
    # 折线图数据
    x_data = [i['level'] for i in sort_data]
    y_data = [i['count'] for i in sort_data]

    return render(request,'plot6.html',locals())
def plot7(request):
    # 用户信息
    user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
    # 获取用户的obj
    userinfo = User.objects.get(id=user["uid"])
    # 用户id  用户名
    uid,username = userinfo.id,userinfo.username
    raw_data = Comment.objects.all()
    data = [
        [
            len(i.content), i .reply
        ]
        for i in raw_data
    ]
    return render(request,'plot7.html',locals())
# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty


@method_decorator(check_login, name='get')
class HighRiskListView(View):
    def get(self, request):
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        userinfo = User.objects.get(id=user["uid"])
        uid, username = userinfo.id, userinfo.username

        min_risk_level = request.GET.get('min_risk_level', '2').strip()
        if not min_risk_level.isdigit():
            min_risk_level = '2'
        min_risk_level_int = int(min_risk_level)

        page = request.GET.get('page', '1').strip()
        if not page.isdigit():
            page = '1'
        page_int = int(page)

        rows = (
            CommentRawTextMap.objects
            .select_related('comment', 'raw_text')
            .filter(raw_text__predictions__risk_level__gte=min_risk_level_int)
            .order_by('-raw_text__predictions__created_at')
        )

        result_rows = []
        seen = set()
        for m in rows:
            if m.comment_id in seen:
                continue
            seen.add(m.comment_id)
            pr = (
                PredictionResult.objects
                .select_related('predicted_label', 'model_info')
                .filter(raw_text_id=m.raw_text_id, risk_level__gte=min_risk_level_int)
                .order_by('-created_at')
                .first()
            )
            if pr is None:
                continue
            result_rows.append({
                'comment': m.comment,
                'raw_text': m.raw_text,
                'prediction': pr,
            })

        paginator = Paginator(result_rows, 20)
        page_obj = paginator.get_page(page_int)
        total = paginator.count

        return render(
            request,
            'high_risk_list.html',
            {
                'userinfo': userinfo,
                'uid': uid,
                'username': username,
                'page_obj': page_obj,
                'total': total,
                'min_risk_level': min_risk_level_int,
            },
        )


@method_decorator(check_login, name='get')
class RiskDetailView(View):
    def get(self, request, comment_id):
        user = json.loads(request.get_signed_cookie('user', salt='salt', default='{}'))
        userinfo = User.objects.get(id=user["uid"])
        uid, username = userinfo.id, userinfo.username

        comment = Comment.objects.filter(id=comment_id).first()
        if comment is None:
            return HttpResponse('评论不存在', status=404)

        map_obj = (
            CommentRawTextMap.objects
            .select_related('raw_text')
            .filter(comment_id=comment_id)
            .first()
        )
        raw_obj = map_obj.raw_text if map_obj else None

        prediction_history = []
        latest_prediction = None
        detail_pretty = ''

        if raw_obj is not None:
            prediction_history = list(
                PredictionResult.objects
                .select_related('predicted_label', 'model_info')
                .filter(raw_text_id=raw_obj.id)
                .order_by('-created_at')
            )
            if prediction_history:
                latest_prediction = prediction_history[0]
                if latest_prediction.detail_json:
                    try:
                        d = json.loads(latest_prediction.detail_json)
                        detail_pretty = json.dumps(d, ensure_ascii=False, indent=2)
                    except Exception:
                        detail_pretty = latest_prediction.detail_json

        return render(
            request,
            'risk_detail.html',
            {
                'userinfo': userinfo,
                'uid': uid,
                'username': username,
                'comment': comment,
                'raw_obj': raw_obj,
                'latest_prediction': latest_prediction,
                'prediction_history': prediction_history,
                'detail_pretty': detail_pretty,
                'legacy_emotion': comment.emotion or '',
            },
        )
# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

@method_decorator(check_login, name='dispatch')
class predict(View):
    def get(self, request):
        userinfo = _front_user_from_cookie(request)
        if not userinfo:
            return redirect(reverse('login'))
        uid, username = userinfo.id, userinfo.username

        from app.services.predict_service import PredictService

        user_plan = _user_active_plan(userinfo)
        predict_plan_missing = user_plan is None
        predict_models = _predict_models_for_user_plan(userinfo, user_plan)
        allowed_ids = {m.id for m in predict_models}

        default_predict_model_id = None
        try:
            am = PredictService.load_active_model()
            if (
                am
                and am.listed_for_users
                and userinfo.can_use_model_info(am)
                and am.id in allowed_ids
            ):
                default_predict_model_id = am.id
        except Exception:
            pass

        if default_predict_model_id is None and predict_models:
            default_predict_model_id = predict_models[0].id

        return render(
            request,
            'predict.html',
            {
                'userinfo': userinfo,
                'uid': uid,
                'username': username,
                'user_plan': user_plan,
                'predict_plan_missing': predict_plan_missing,
                'predict_models': predict_models,
                'default_predict_model_id': default_predict_model_id,
            },
        )

    def post(self, request):
        userinfo = _front_user_from_cookie(request)
        if not userinfo:
            _write_prediction_usage_log(None, None, '', False, '未登录调用')
            return JsonResponse({'status': 0, 'msg': '请先登录后再使用在线预测'}, status=401)

        from app.services.predict_service import PredictService

        content = (request.POST.get('content') or '').strip()
        if not content:
            _write_prediction_usage_log(userinfo, None, content, False, '输入为空')
            return JsonResponse({'status': 0, 'msg': '请输入文本内容'})

        user_plan = _user_active_plan(userinfo)
        if user_plan is None:
            _write_prediction_usage_log(userinfo, None, content, False, '未绑定有效套餐')
            return JsonResponse(
                {'status': 0, 'msg': '您尚未绑定有效套餐，无法使用在线预测；请先在个人中心办理套餐'},
                status=403,
            )

        predict_models = _predict_models_for_user_plan(userinfo, user_plan)
        allowed_ids = {m.id for m in predict_models}
        if not allowed_ids:
            _write_prediction_usage_log(userinfo, None, content, False, '套餐下无可用模型')
            return JsonResponse(
                {
                    'status': 0,
                    'msg': '当前套餐未包含任何可用的已发布模型，请联系管理员调整授权方案或模型发布状态',
                },
                status=403,
            )

        model_info = None
        mid = (request.POST.get('model_id') or '').strip()
        if mid.isdigit():
            model_info = ModelInfo.objects.filter(
                id=int(mid), status='ready', listed_for_users=True
            ).first()
            if not model_info:
                _write_prediction_usage_log(userinfo, None, content, False, '模型不可用/未发布')
                return JsonResponse({'status': 0, 'msg': '所选模型不可用、未就绪或未开放前台使用'})
            if not userinfo.can_use_model_info(model_info):
                _write_prediction_usage_log(userinfo, model_info, content, False, '模型无权限')
                return JsonResponse({'status': 0, 'msg': '您无权使用该算法类型的模型'}, status=403)
            if model_info.id not in allowed_ids:
                _write_prediction_usage_log(userinfo, model_info, content, False, '模型不在当前套餐')
                return JsonResponse(
                    {'status': 0, 'msg': '所选模型不在您当前套餐允许的算法范围内'},
                    status=403,
                )
        else:
            try:
                am = PredictService.load_active_model()
            except Exception:
                am = None
            if am and am.id in allowed_ids and userinfo.can_use_model_info(am):
                model_info = am
            elif predict_models:
                model_info = predict_models[0]
            if model_info is None:
                _write_prediction_usage_log(userinfo, None, content, False, '无可用模型')
                return JsonResponse(
                    {'status': 0, 'msg': '当前没有可在套餐内使用的模型，请在上方选择具体模型或联系管理员'},
                    status=403,
                )

        try:
            pr = PredictService.predict_text_line(content, model_info=model_info)
        except Exception as e:
            _write_prediction_usage_log(userinfo, model_info, content, False, '预测失败: {}'.format(e))
            return JsonResponse({'status': 0, 'msg': '预测失败: {}'.format(e)})

        emotion = pr.get('label') or '无法判定'
        _write_prediction_usage_log(
            userinfo,
            model_info,
            content,
            True,
            '预测成功: {} / {}'.format(pr.get('model_name') or '', pr.get('model_version') or ''),
        )
        print(
            'predict demo:',
            emotion,
            pr.get('final_score'),
            pr.get('risk_level'),
            pr.get('model_name'),
            pr.get('model_version'),
        )

        return JsonResponse({
            'status': 1,
            'msg': '使用「{} / {}」→ 标签: {}'.format(
                pr.get('model_name') or '',
                pr.get('model_version') or '',
                emotion,
            ),
            'label': emotion,
            'risk_level': pr.get('risk_level'),
            'final_score': pr.get('final_score'),
            'model_id': pr.get('model_id'),
            'model_name': pr.get('model_name'),
            'model_version': pr.get('model_version'),
            'model_type': pr.get('model_type'),
            'model_type_display': pr.get('model_type_display'),
        })