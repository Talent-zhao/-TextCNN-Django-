import json

from django.db import models

# Create your models here.
class User(models.Model):
    SEX_CHOICE = (
        ('男','男'),
        ('女','女')
    )
    ROLE_CHOICES = (
        ('admin', '管理员'),
        ('user', '普通用户'),
    )
    id = models.AutoField(primary_key=True,verbose_name='id')
    time = models.DateField(auto_now_add=True,verbose_name='创建时间')
    email = models.EmailField(null=True,verbose_name='email',blank=True)
    #  必须有
    username = models.CharField(max_length=64,verbose_name='昵称')
    tel = models.CharField(max_length=64,verbose_name='手机号')
    pwd = models.CharField(max_length=64,verbose_name='密码',default="123456")
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default='user', verbose_name='角色')

    sex = models.CharField(max_length=4,verbose_name='性别',choices=SEX_CHOICE,null=True,blank=True)
    address = models.CharField(max_length=128,verbose_name='地址',null=True,blank=True)
    avatar  = models.ImageField(upload_to='avatar/',verbose_name='用户头像',null=True,blank=True)
    is_active = models.BooleanField(default=True, verbose_name='账号启用')
    allowed_model_types = models.TextField(
        blank=True,
        default='',
        verbose_name='可用算法类型',
        help_text='JSON 数组，如 ["svm","textcnn"]；留空表示不限制（前台仅能看到「就绪且前台可见」的模型登记）',
    )
    admin_note = models.CharField(max_length=255, blank=True, null=True, verbose_name='管理员备注')

    class Meta:
        verbose_name = '用户表'  # 定义在管理后台显示的名称
        verbose_name_plural = verbose_name  # 定义复数时的名称（去除复数的s）

    def __str__(self):
        return self.username

    def allowed_model_types_set(self):
        """None 表示不限制；否则为允许的 model_type 字符串集合。"""
        raw = (self.allowed_model_types or '').strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return None
            return {str(x) for x in data}
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def filter_models_for_user(self, queryset):
        s = self.allowed_model_types_set()
        if s is None:
            return queryset
        return queryset.filter(model_type__in=s)

    def can_use_model_info(self, model_info):
        if model_info is None:
            return True
        s = self.allowed_model_types_set()
        if s is None:
            return True
        return model_info.model_type in s

    def get_allowed_algo_summary(self):
        s = self.allowed_model_types_set()
        if s is None:
            return '全部'
        if not s:
            return '无（前台不可选模型）'
        from django.apps import apps

        MI = apps.get_model('app', 'ModelInfo')
        cmap = dict(MI.MODEL_TYPE_CHOICES)
        return '、'.join(cmap.get(k, k) for k in sorted(s))


class Comment(models.Model):
    id = models.AutoField(primary_key=True)
    time = models.DateField(verbose_name='发布时间')
    floor = models.IntegerField(verbose_name='楼层')
    name = models.CharField(max_length=256, verbose_name='评论者昵称')
    user_url = models.CharField(max_length=256, verbose_name='评论者主页')
    img = models.CharField(max_length=256, verbose_name='评论者头像')
    level = models.IntegerField(verbose_name='等级')
    content = models.CharField(max_length=256, verbose_name='评论内容')
    reply = models.IntegerField(verbose_name='回复数')

    sex = models.CharField(max_length=256, verbose_name='评论者性别')
    loc = models.CharField(max_length=256, verbose_name='评论者所在地')

    emotion_chooice = (
        ('没有抑郁倾向', '没有抑郁倾向'),
        ('一般抑郁倾向', '一般抑郁倾向'),
        ('轻度抑郁倾向', '轻度抑郁倾向'),
        ('严重抑郁倾向', '严重抑郁倾向'),
    )
    emotion = models.CharField(max_length=256, verbose_name='情感',choices=emotion_chooice,default="")

    class Meta:
        verbose_name_plural = '评论管理'  # 此时，admin中表的名字就是‘用户表‘


class CommentRawTextMap(models.Model):
    """评论与原始文本的稳定映射。"""
    comment = models.OneToOneField(Comment, on_delete=models.CASCADE, related_name='rawtext_map', verbose_name='评论')
    raw_text = models.ForeignKey('RawText', on_delete=models.CASCADE, related_name='comment_maps', verbose_name='原始文本')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '评论原始文本映射'
        verbose_name_plural = verbose_name


class TextSource(models.Model):
    """文本来源配置，支持贴吧/导入文件/手工录入等来源区分。"""
    SOURCE_TYPE_CHOICES = (
        ('tieba', '贴吧'),
        ('csv', 'CSV导入'),
        ('excel', 'Excel导入'),
        ('json', 'JSON导入'),
        ('manual', '手工录入'),
        ('api', '外部接口'),
    )

    name = models.CharField(max_length=64, verbose_name='来源名称')
    source_type = models.CharField(max_length=16, choices=SOURCE_TYPE_CHOICES, default='tieba', verbose_name='来源类型')
    source_key = models.CharField(max_length=128, blank=True, null=True, verbose_name='来源标识')
    is_enabled = models.BooleanField(default=True, verbose_name='是否启用')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '文本来源'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.name


class RawText(models.Model):
    """原始文本记录（数据接入层）。"""
    STATUS_PENDING_CLEAN = 'pending_clean'
    STATUS_CLEANED = 'cleaned'
    STATUS_PENDING_LABEL = 'pending_label'
    STATUS_LABELED = 'labeled'
    STATUS_PENDING_PREDICT = 'pending_predict'
    STATUS_PREDICTED = 'predicted'
    STATUS_REVIEWED = 'reviewed'

    STATUS_CHOICES = (
        (STATUS_PENDING_CLEAN, '未清洗'),
        (STATUS_CLEANED, '已清洗'),
        (STATUS_PENDING_LABEL, '待标注'),
        (STATUS_LABELED, '已标注'),
        (STATUS_PENDING_PREDICT, '待预测'),
        (STATUS_PREDICTED, '已预测'),
        (STATUS_REVIEWED, '已复核'),
    )

    source = models.ForeignKey(TextSource, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='来源')
    external_id = models.CharField(max_length=128, blank=True, null=True, verbose_name='外部ID')
    author_name = models.CharField(max_length=128, blank=True, null=True, verbose_name='作者')
    content = models.TextField(verbose_name='原始文本')
    publish_time = models.DateTimeField(blank=True, null=True, verbose_name='发布时间')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING_CLEAN, verbose_name='处理状态')
    dedup_hash = models.CharField(max_length=64, blank=True, null=True, verbose_name='去重哈希')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '原始文本'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.external_id or f'RawText#{self.id}'

    @classmethod
    def get_status_flow(cls):
        """
        状态流转规则（最小闭环）:
        未清洗 -> 已清洗 -> 待标注 -> 已标注 -> 待预测 -> 已预测 -> 已复核
        """
        return {
            cls.STATUS_PENDING_CLEAN: [cls.STATUS_CLEANED],
            cls.STATUS_CLEANED: [cls.STATUS_PENDING_LABEL],
            cls.STATUS_PENDING_LABEL: [cls.STATUS_LABELED],
            cls.STATUS_LABELED: [cls.STATUS_PENDING_PREDICT],
            cls.STATUS_PENDING_PREDICT: [cls.STATUS_PREDICTED],
            cls.STATUS_PREDICTED: [cls.STATUS_REVIEWED],
            cls.STATUS_REVIEWED: [],
        }

    def can_transit_to(self, target_status):
        flow = self.get_status_flow()
        return target_status in flow.get(self.status, [])

    def transit_to(self, target_status):
        if not self.can_transit_to(target_status):
            raise ValueError(f'非法状态流转: {self.status} -> {target_status}')
        self.status = target_status
        self.save(update_fields=['status', 'updated_at'])


class CleanText(models.Model):
    """文本预处理结果表。"""
    raw_text = models.OneToOneField(RawText, on_delete=models.CASCADE, related_name='clean_result', verbose_name='原始文本')
    cleaned_content = models.TextField(verbose_name='清洗后文本')
    tokenized_content = models.TextField(blank=True, null=True, verbose_name='分词结果')
    removed_special_chars = models.BooleanField(default=True, verbose_name='是否清洗特殊字符')
    removed_stopwords = models.BooleanField(default=False, verbose_name='是否去停用词')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '清洗文本'
        verbose_name_plural = verbose_name


class LabelDefinition(models.Model):
    """标签定义，兼容二分类与四分类。"""
    code = models.IntegerField(unique=True, verbose_name='标签编码')
    name = models.CharField(max_length=64, unique=True, verbose_name='标签名称')
    description = models.CharField(max_length=255, blank=True, null=True, verbose_name='标签描述')
    risk_level = models.IntegerField(default=0, verbose_name='风险等级')
    is_active = models.BooleanField(default=True, verbose_name='是否启用')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '标签定义'
        verbose_name_plural = verbose_name

    def __str__(self):
        return f'{self.code}-{self.name}'


class AnnotationRecord(models.Model):
    """样本标注记录。"""
    raw_text = models.ForeignKey(RawText, on_delete=models.CASCADE, related_name='annotations', verbose_name='原始文本')
    label = models.ForeignKey(LabelDefinition, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='标签')
    annotator = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='标注人')
    dataset_split = models.CharField(max_length=16, default='train', verbose_name='数据集划分')
    remark = models.CharField(max_length=255, blank=True, null=True, verbose_name='备注')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='标注时间')

    class Meta:
        verbose_name = '标注记录'
        verbose_name_plural = verbose_name


class ModelInfo(models.Model):
    """模型信息与版本管理。"""
    MODEL_TYPE_CHOICES = (
        ('svm', 'TF-IDF + SVM'),
        ('knn', 'TF-IDF + KNN（最近邻）'),
        ('rf', 'TF-IDF + RandomForest（随机森林）'),
        ('dt', 'TF-IDF + DecisionTree（决策树）'),
        ('lr', 'TF-IDF + LogisticRegression'),
        ('textcnn', 'TextCNN'),
        ('textrcnn', 'TextRCNN'),
        ('rule', '规则词典'),
        ('fusion', '融合策略模型'),
        ('hf_depr_bin', 'HF DistilBERT 抑郁二分类'),
        ('hf_mh_4cls', 'HF BERT 心理健康四分类'),
    )
    STATUS_CHOICES = (
        ('training', '训练中'),
        ('ready', '可用'),
        ('disabled', '已停用'),
        ('failed', '训练失败'),
    )

    name = models.CharField(max_length=64, verbose_name='模型名称')
    model_type = models.CharField(max_length=16, choices=MODEL_TYPE_CHOICES, verbose_name='模型类型')
    version = models.CharField(max_length=32, verbose_name='版本号')
    file_path = models.CharField(max_length=255, blank=True, null=True, verbose_name='模型文件路径')
    vectorizer_path = models.CharField(max_length=255, blank=True, null=True, verbose_name='向量器路径')
    description = models.CharField(max_length=255, blank=True, null=True, verbose_name='模型说明')
    config_json = models.TextField(blank=True, null=True, verbose_name='模型配置JSON')
    metrics_json = models.TextField(blank=True, null=True, verbose_name='评估指标JSON')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='ready', verbose_name='状态')
    is_active = models.BooleanField(default=False, verbose_name='是否启用')
    listed_for_users = models.BooleanField(
        default=False,
        verbose_name='前台与用户权限可见',
        help_text='勾选后：该登记会出现在前台「在线预测」模型列表，且用户管理中可按该类型分配权限；未勾选则仅后台可见。',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '模型信息'
        verbose_name_plural = verbose_name
        unique_together = ('name', 'version')

    def __str__(self):
        return f'{self.name}-{self.version}'

    @classmethod
    def assignable_model_type_choices(cls):
        """套餐 / 用户可用算法勾选列表。"""
        types = set(
            cls.objects.filter(status='ready', listed_for_users=True).values_list(
                'model_type', flat=True
            )
        )
        if cls.objects.filter(status='ready', model_type='fusion').exists():
            types.add('fusion')
        return tuple(c for c in cls.MODEL_TYPE_CHOICES if c[0] in types)


class TrainingTask(models.Model):
    """训练任务表。"""
    STATUS_CHOICES = (
        ('pending', '待执行'),
        ('running', '执行中'),
        ('success', '成功'),
        ('failed', '失败'),
    )

    task_name = models.CharField(max_length=128, verbose_name='任务名称')
    model = models.ForeignKey(ModelInfo, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='目标模型')
    creator = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='创建人')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='pending', verbose_name='任务状态')
    params_json = models.TextField(blank=True, null=True, verbose_name='训练参数JSON')
    result_json = models.TextField(blank=True, null=True, verbose_name='训练结果JSON')
    started_at = models.DateTimeField(blank=True, null=True, verbose_name='开始时间')
    ended_at = models.DateTimeField(blank=True, null=True, verbose_name='结束时间')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '训练任务'
        verbose_name_plural = verbose_name


class PredictionResult(models.Model):
    """在线/批量分析结果。"""
    raw_text = models.ForeignKey(RawText, on_delete=models.CASCADE, related_name='predictions', verbose_name='原始文本')
    model_info = models.ForeignKey(ModelInfo, on_delete=models.SET_NULL, blank=True, null=True, related_name='prediction_records', verbose_name='模型信息')
    model = models.ForeignKey(ModelInfo, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='模型')
    predicted_label = models.ForeignKey(LabelDefinition, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='预测标签')
    probability_score = models.FloatField(default=0.0, verbose_name='模型概率分')
    rule_score = models.FloatField(default=0.0, verbose_name='规则分')
    final_risk_score = models.FloatField(default=0.0, verbose_name='最终风险分')
    is_alert_triggered = models.BooleanField(default=False, verbose_name='是否触发预警')
    probability = models.FloatField(default=0.0, verbose_name='预测概率')
    risk_score = models.FloatField(default=0.0, verbose_name='风险得分')
    risk_level = models.IntegerField(default=0, verbose_name='风险等级')
    hit_keywords = models.CharField(max_length=512, blank=True, null=True, verbose_name='命中关键词')
    detail_json = models.TextField(blank=True, null=True, verbose_name='详情JSON')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='分析时间')

    class Meta:
        verbose_name = '预测结果'
        verbose_name_plural = verbose_name


class SystemLog(models.Model):
    """系统日志。"""
    LEVEL_CHOICES = (
        ('info', '信息'),
        ('warning', '警告'),
        ('error', '错误'),
    )

    operator = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='操作人')
    module = models.CharField(max_length=64, verbose_name='模块')
    action = models.CharField(max_length=128, verbose_name='动作')
    level = models.CharField(max_length=16, choices=LEVEL_CHOICES, default='info', verbose_name='日志级别')
    content = models.TextField(blank=True, null=True, verbose_name='内容')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='记录时间')

    class Meta:
        verbose_name = '系统日志'
        verbose_name_plural = verbose_name


class SystemConfig(models.Model):
    """系统参数配置（融合权重、阈值、模型路径等）。"""
    key = models.CharField(max_length=64, unique=True, verbose_name='配置键')
    value = models.CharField(max_length=255, verbose_name='配置值')
    value_type = models.CharField(max_length=16, default='str', verbose_name='值类型')
    description = models.CharField(max_length=255, blank=True, null=True, verbose_name='配置说明')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '系统配置'
        verbose_name_plural = verbose_name

    def __str__(self):
        return f'{self.key}={self.value}'


class FusionConfigPreset(models.Model):
    """预测融合 + 启用模型的命名快照，可一键恢复。"""
    is_auto = models.BooleanField(default=False, verbose_name='系统自动快照')
    name = models.CharField(max_length=128, verbose_name='方案名称')
    remark = models.CharField(max_length=255, blank=True, default='', verbose_name='备注')
    snapshot_json = models.TextField(verbose_name='快照数据JSON')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '融合配置快照'
        verbose_name_plural = verbose_name
        ordering = ['-updated_at']

    def __str__(self):
        return self.name


class ModelEvaluation(models.Model):
    """模型效果记录（可手动录入）。"""
    model_info = models.ForeignKey(ModelInfo, on_delete=models.CASCADE, related_name='evaluations', verbose_name='模型')
    model_version = models.CharField(max_length=32, verbose_name='模型版本')
    accuracy = models.FloatField(default=0.0, verbose_name='准确率')
    precision = models.FloatField(default=0.0, verbose_name='精确率')
    recall = models.FloatField(default=0.0, verbose_name='召回率')
    f1_score = models.FloatField(default=0.0, verbose_name='F1')
    auc = models.FloatField(blank=True, null=True, verbose_name='AUC')
    training_time_sec = models.FloatField(blank=True, null=True, verbose_name='训练耗时(秒)')
    is_overfitting = models.BooleanField(blank=True, null=True, verbose_name='是否过拟合')
    evaluated_at = models.DateTimeField(auto_now_add=True, verbose_name='评估时间')
    is_active_snapshot = models.BooleanField(default=False, verbose_name='当时是否启用')

    class Meta:
        verbose_name = '模型评估记录'
        verbose_name_plural = verbose_name


class AlgorithmExperimentRecord(models.Model):
    """算法对比实验记录（统一沉淀离线实验结果）。"""
    ALGO_TYPE_CHOICES = (
        ('svm', 'SVM'),
        ('knn', 'KNN'),
        ('rf', '随机森林'),
        ('dt', '决策树'),
        ('lr', '逻辑回归'),
        ('textcnn', 'TextCNN'),
        ('textrcnn', 'TextRCNN'),
    )
    # 与模型训练中心 training_hub 选项文案保持一致
    TRAIN_SAMPLE_CHOICES = (
        ('full', '全量（数据多则更慢，通常效果更好）'),
        ('8000', '随机子采样 8,000 条（快速试跑）'),
        ('15000', '随机子采样 15,000 条'),
        ('30000', '随机子采样 30,000 条'),
    )
    SKLEARN_MAX_FEATURES_CHOICES = (
        ('4000', '4,000（词表较小，省内存）'),
        ('8000', '8,000（默认，平衡）'),
        ('12000', '12,000（词表更大，更慢）'),
    )

    name = models.CharField(max_length=128, verbose_name='实验名称')
    dataset_version = models.CharField(max_length=64, verbose_name='数据集/数据版本')
    algorithm_type = models.CharField(max_length=16, choices=ALGO_TYPE_CHOICES, verbose_name='算法类型')
    train_sample_scale = models.CharField(
        max_length=16,
        choices=TRAIN_SAMPLE_CHOICES,
        blank=True,
        default='full',
        verbose_name='样本规模（传统 ML 与深度学习共用）',
        help_text='与训练中心「样本规模」一致',
    )
    sklearn_max_features = models.CharField(
        max_length=16,
        choices=SKLEARN_MAX_FEATURES_CHOICES,
        blank=True,
        default='8000',
        verbose_name='TF-IDF 特征上限（仅传统 ML）',
        help_text='与训练中心「TF-IDF 特征上限」一致；深度学习实验可保留默认作记录',
    )
    hyperparams_json = models.TextField(blank=True, default='{}', verbose_name='超参数JSON')
    training_time_sec = models.FloatField(blank=True, null=True, verbose_name='训练时间(秒)')
    accuracy = models.FloatField(default=0.0, verbose_name='Accuracy')
    precision = models.FloatField(default=0.0, verbose_name='Precision')
    recall = models.FloatField(default=0.0, verbose_name='Recall')
    f1_score = models.FloatField(default=0.0, verbose_name='F1')
    auc = models.FloatField(blank=True, null=True, verbose_name='AUC')
    remark = models.CharField(max_length=255, blank=True, default='', verbose_name='备注')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '算法对比实验记录'
        verbose_name_plural = verbose_name
        ordering = ['-created_at', '-id']

    def __str__(self):
        return '{} / {} / {}'.format(self.name, self.dataset_version, self.algorithm_type)


class ModelSelfCheckRecord(models.Model):
    """模型自检历史记录。"""
    mode = models.CharField(max_length=16, verbose_name='检测模式')
    summary = models.CharField(max_length=32, verbose_name='结果摘要')
    has_error = models.BooleanField(default=False, verbose_name='是否存在错误')
    detail_json = models.TextField(verbose_name='详细结果JSON')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '模型自检记录'
        verbose_name_plural = verbose_name


class ExportLog(models.Model):
    """导出任务日志。"""
    EXPORT_TYPE_CHOICES = (
        ('prediction_result', '预测结果'),
        ('self_check_history', '自检历史'),
    )
    EXPORT_FORMAT_CHOICES = (
        ('csv', 'CSV'),
        ('json', 'JSON'),
    )

    export_type = models.CharField(max_length=32, choices=EXPORT_TYPE_CHOICES, verbose_name='导出类型')
    export_format = models.CharField(max_length=16, choices=EXPORT_FORMAT_CHOICES, verbose_name='导出格式')
    exporter = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='导出人')
    filter_json = models.TextField(blank=True, null=True, verbose_name='筛选条件JSON')
    export_count = models.IntegerField(default=0, verbose_name='导出条数')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='导出时间')

    class Meta:
        verbose_name = '导出日志'
        verbose_name_plural = verbose_name


class PredictionUsageLog(models.Model):
    """前台预测调用日志（为后续配额/计费扩展预留）。"""
    user = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='调用用户')
    model_info = models.ForeignKey(
        ModelInfo,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='usage_logs',
        verbose_name='调用模型',
    )
    input_length = models.IntegerField(default=0, verbose_name='输入文本长度')
    success = models.BooleanField(default=False, verbose_name='是否成功')
    message = models.CharField(max_length=255, blank=True, default='', verbose_name='结果消息')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='调用时间')

    class Meta:
        verbose_name = '预测调用日志'
        verbose_name_plural = verbose_name


class ModelPlan(models.Model):
    """模型调用套餐/授权方案骨架。"""
    name = models.CharField(max_length=64, verbose_name='方案名称')
    description = models.CharField(max_length=255, blank=True, default='', verbose_name='描述')
    is_active = models.BooleanField(default=True, verbose_name='是否启用')
    total_quota = models.IntegerField(default=0, verbose_name='总调用额度')
    valid_from = models.DateTimeField(blank=True, null=True, verbose_name='生效时间')
    valid_to = models.DateTimeField(blank=True, null=True, verbose_name='失效时间')
    allowed_model_types_json = models.TextField(
        blank=True,
        default='',
        verbose_name='允许的模型类型（兼容旧数据）',
        help_text='JSON 数组，如 ["svm","textcnn"]；新方案在后台勾选「模型登记」后通常留空，仅按 allowed_model_ids_json 限制。',
    )
    allowed_model_ids_json = models.TextField(
        blank=True,
        default='',
        verbose_name='允许的模型登记ID',
        help_text='JSON 数组，对应「模型管理」中登记主键 id，如 [29,19]；非空时优先按登记限制，不再看类型字段。',
    )

    class Meta:
        verbose_name = '模型授权方案'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.name


class UserPlan(models.Model):
    """用户绑定的套餐/授权方案。"""
    STATUS_CHOICES = (
        ('pending', '待生效'),
        ('active', '生效中'),
        ('expired', '已过期'),
        ('suspended', '已停用'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='plans', verbose_name='用户')
    plan = models.ForeignKey(ModelPlan, on_delete=models.CASCADE, related_name='user_plans', verbose_name='方案')
    remaining_quota = models.IntegerField(default=0, verbose_name='剩余额度')
    start_time = models.DateTimeField(blank=True, null=True, verbose_name='开始时间')
    end_time = models.DateTimeField(blank=True, null=True, verbose_name='结束时间')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='pending', verbose_name='状态')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    class Meta:
        verbose_name = '用户套餐'
        verbose_name_plural = verbose_name


class QuotaUsageLog(models.Model):
    """额度扣减流水，关联调用日志。"""
    user = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='用户')
    plan = models.ForeignKey(UserPlan, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='用户套餐')
    model_info = models.ForeignKey(ModelInfo, on_delete=models.SET_NULL, blank=True, null=True, verbose_name='模型')
    usage_log = models.ForeignKey(
        PredictionUsageLog,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='quota_logs',
        verbose_name='调用日志',
    )
    delta = models.IntegerField(default=0, verbose_name='扣减数量')
    before_quota = models.IntegerField(default=0, verbose_name='扣减前额度')
    after_quota = models.IntegerField(default=0, verbose_name='扣减后额度')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    class Meta:
        verbose_name = '额度扣减流水'
        verbose_name_plural = verbose_name