"""DjangoWeb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app import views, userViews, adminViews, dataViews, annotationViews, riskViews, modelViews, trainViews, auditViews, user_mgmt_views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic.base import RedirectView

urlpatterns = [
    # path('admin/', admin.site.urls),  # Django默认admin后台

    # 管理员前端界面路由
    path('admin_panel/login/', adminViews.AdminLoginView.as_view(), name='admin_login'),
    path('admin_panel/logout/', adminViews.admin_logout, name='admin_logout'),
    path('admin_panel/dashboard/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False)),
    path('admin_panel/users/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False)),
    path('admin_panel/users/list/', user_mgmt_views.UserMgmtListView.as_view(), name='user_mgmt_list'),
    path('admin_panel/users/add/', user_mgmt_views.UserMgmtFormView.as_view(), name='user_mgmt_add'),
    path('admin_panel/users/<int:pk>/edit/', user_mgmt_views.UserMgmtFormView.as_view(), name='user_mgmt_edit'),
    path('admin_panel/users/<int:pk>/delete/', user_mgmt_views.UserMgmtDeleteView.as_view(), name='user_mgmt_delete'),
    path('admin_panel/plans/', user_mgmt_views.ModelPlanListView.as_view(), name='model_plan_list'),
    path('admin_panel/plans/add/', user_mgmt_views.ModelPlanFormView.as_view(), name='model_plan_add'),
    path('admin_panel/plans/<int:pk>/edit/', user_mgmt_views.ModelPlanFormView.as_view(), name='model_plan_edit'),
    path('admin_panel/plans/<int:pk>/delete/', user_mgmt_views.ModelPlanDeleteView.as_view(), name='model_plan_delete'),
    path('admin_panel/user_plans/', user_mgmt_views.UserPlanListView.as_view(), name='user_plan_list'),
    path('admin_panel/user_plans/add/', user_mgmt_views.UserPlanFormView.as_view(), name='user_plan_add'),
    path('admin_panel/user_plans/<int:pk>/edit/', user_mgmt_views.UserPlanFormView.as_view(), name='user_plan_edit'),
    path('admin_panel/user_plans/<int:pk>/delete/', user_mgmt_views.UserPlanDeleteView.as_view(), name='user_plan_delete'),
    path('admin_panel/comments/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False)),
    path('admin_panel/statistics/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False)),

    path('login/', userViews.loginView.as_view(), name= 'login'), # 登录
    path('register/', userViews.registerViews.as_view(), name= 'register'), # 注册
    path('logout/', userViews.logout, name= 'logout'), # 注销
    path('debug/role/', userViews.role_debug, name='role_debug'),
    path('myProfile/', userViews.myProfileView.as_view(), name= 'myProfile'), # 我的资料
    path('resetPWd/', userViews.resetPWdView.as_view(), name= 'resetPWd'), # 修改密码
    ################# 核心
    path('scrawl/',views.scrawl.as_view(),name='scrawl'), # 爬取评论
    #################  初步可视化
    path('plot1',views.plot1,name='plot1'), #词云图
    path('plot2',views.plot2,name='plot2'), #微博贴吧每日评论折线图

    #################  抑郁可视化
    path('plot3',views.plot3,name='plot3'), # 情感分类环形图
    path('plot4',views.plot4,name='plot4'), # 地域分布地图
    path('plot5',views.plot5,name='plot5'), # 用户评论数柱状图
    path('plot6',views.plot6,name='plot6'), # 用户等级数柱状图
    path('plot7',views.plot7,name='plot7'), # 评论长度和回复数散点图
    path('high_risk/', views.HighRiskListView.as_view(), name='high_risk_list'),
    path('risk_detail/<int:comment_id>/', views.RiskDetailView.as_view(), name='risk_detail'),


    path('', views.IndexView.as_view(), name= 'index'), #

    path('init/', views.init, name= 'init'), # 数据录入到数据库中
    path('predict/', views.predict.as_view(), name='predict'),  # 在线预测
    path('trigger_fenlei/', views.trigger_fenlei, name='trigger_fenlei'),
    # 兼容保留：工作台入口已下线，统一跳转到用户管理
    path('admin_panel/workbench/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False), name='workbench'),

    # 数据管理模块（最小可运行）
    path('admin_panel/data/import/', dataViews.rawtext_import_redirect, name='rawtext_import'),
    path('admin_panel/data/rawtexts/', dataViews.RawTextListView.as_view(), name='rawtext_list'),
    path('admin_panel/data/rawtexts/<int:raw_id>/', dataViews.RawTextDetailView.as_view(), name='rawtext_detail'),

    # 标注模块（最小可运行）
    path('admin_panel/annotation/list/', annotationViews.AnnotationListView.as_view(), name='annotation_list'),

    # 预测（后半闭环）
    path('admin_panel/risk/pending_predict/', riskViews.PendingPredictListView.as_view(), name='pending_predict_list'),
    path('admin_panel/risk/predictions/', riskViews.PredictionResultListView.as_view(), name='prediction_result_list'),
    path('admin_panel/risk/predictions/<int:prediction_id>/', riskViews.PredictionResultDetailView.as_view(), name='prediction_result_detail'),
    path('admin_panel/risk/usage_logs/', riskViews.PredictionUsageLogListView.as_view(), name='prediction_usage_log_list'),

    # 模型管理与融合配置
    path('admin_panel/model/list/', modelViews.ModelListView.as_view(), name='model_list'),
    path('admin_panel/model/add/', modelViews.ModelEditView.as_view(), name='model_add'),
    path('admin_panel/model/edit/<int:model_id>/', modelViews.ModelEditView.as_view(), name='model_edit'),
    path('admin_panel/model/config/', modelViews.RuntimeConfigView.as_view(), name='runtime_config'),
    path('admin_panel/model/active/', modelViews.ActiveModelView.as_view(), name='active_model'),
    path('admin_panel/model/evaluation/', modelViews.ModelEvaluationView.as_view(), name='model_evaluation'),
    path('admin_panel/model/algorithm_compare/', modelViews.AlgorithmCompareView.as_view(), name='algorithm_compare'),
    path('admin_panel/model/experiments/', modelViews.AlgorithmExperimentRecordListView.as_view(), name='algorithm_experiment_record_list'),
    path('admin_panel/model/experiments/add/', modelViews.AlgorithmExperimentRecordFormView.as_view(), name='algorithm_experiment_record_add'),
    path('admin_panel/model/experiments/<int:record_id>/edit/', modelViews.AlgorithmExperimentRecordFormView.as_view(), name='algorithm_experiment_record_edit'),
    path('admin_panel/model/self_check/', modelViews.ModelSelfCheckView.as_view(), name='model_self_check'),
    path('admin_panel/model/self_check/history/', modelViews.ModelSelfCheckHistoryView.as_view(), name='model_self_check_history'),
    path('admin_panel/model/self_check/history/<int:record_id>/', modelViews.ModelSelfCheckHistoryDetailView.as_view(), name='model_self_check_history_detail'),
    path('admin_panel/model/train/', trainViews.ModelTrainingHubView.as_view(), name='model_training'),
    path(
        'admin_panel/model/train/open_dataset_folder/',
        trainViews.OpenProjectFolderView.as_view(),
        {'folder_kind': 'datasets_nlp'},
        name='model_training_open_folder',
    ),
    path(
        'admin_panel/model/open_model_folder/',
        trainViews.OpenProjectFolderView.as_view(),
        {'folder_kind': 'model'},
        name='model_open_folder',
    ),
    path('admin_panel/model/train/run/', trainViews.ModelTrainingRunView.as_view(), name='model_training_run'),

    # 导出审计日志
    path('admin_panel/audit/export_logs/', auditViews.ExportLogListView.as_view(), name='export_log_list'),
    path('admin_panel/audit/export_logs/<int:log_id>/', auditViews.ExportLogDetailView.as_view(), name='export_log_detail'),

    # 旧路由兼容跳转（避免功能失效）
    path('workbench/', RedirectView.as_view(pattern_name='user_mgmt_list', permanent=False)),
    path('data/import/', RedirectView.as_view(pattern_name='rawtext_import', permanent=False)),
    path('data/rawtexts/', RedirectView.as_view(pattern_name='rawtext_list', permanent=False)),
    path('annotation/list/', RedirectView.as_view(pattern_name='annotation_list', permanent=False)),
    path('risk/pending_predict/', RedirectView.as_view(pattern_name='pending_predict_list', permanent=False)),
    path('risk/predictions/', RedirectView.as_view(pattern_name='prediction_result_list', permanent=False)),
    path('risk/alerts/', RedirectView.as_view(pattern_name='prediction_result_list', permanent=False)),
    path('model/list/', RedirectView.as_view(pattern_name='model_list', permanent=False)),
    path('model/config/', RedirectView.as_view(pattern_name='runtime_config', permanent=False)),
    path('model/evaluation/', RedirectView.as_view(pattern_name='model_evaluation', permanent=False)),
    path('model/self_check/', RedirectView.as_view(pattern_name='model_self_check', permanent=False)),
    path('model/train/', RedirectView.as_view(pattern_name='model_training', permanent=False)),
    path('audit/export_logs/', RedirectView.as_view(pattern_name='export_log_list', permanent=False)),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
