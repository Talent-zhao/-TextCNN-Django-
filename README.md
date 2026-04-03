# 贴吧抑郁 / 心理健康文本分析系统（Django）

面向**贴吧评论采集、数据管理、多算法预测与可视化**的一体化 Web 系统。后端为 **Django 2.2**，集成 **TF-IDF + 传统机器学习分类器**、**字符级 TextCNN / TextRCNN（PyTorch）**、**规则词典**，以及可选的 **Hugging Face Transformers** 序列分类模型（抑郁二分类 DistilBERT、心理健康四分类 BERT），支持**多模型加权融合**、管理员后台全流程（数据导入、标注、模型登记、训练入口、预测结果与导出审计等）。

> **声明**：本系统用于**学习与研究**，模型输出**不构成**临床诊断或心理咨询结论；对英文预训练模型应用于中文场景时需注意领域与语言差异。并且项目于本人学习参考记录使用，里面有很多逻辑上的漏洞，并且没有任何商用价值。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **用户端** | 注册 / 登录 / 个人资料 / 修改密码；支持按用户配置「可用算法类型」JSON，限制前台可选模型 |
| **贴吧爬取** | `scrawl/`：抓取帖子评论，输出至 `data/{tid}.csv` |
| **数据入库** | `init/`：将 CSV 导入数据库（与历史 `Comment` 流程配合） |
| **批量分类** | `trigger_fenlei/`：对评论进行情感 / 抑郁倾向分类（与 `Comment.emotion` 等字段相关） |
| **可视化** | `plot1`～`plot7`：词云、折线图、情感环形图、地域分布、柱状图、散点图等（ECharts 等） |
| **在线预测** | `predict/`：单条文本预测；模型列表受「模型登记 + 用户权限」控制 |
| **高风险列表** | `high_risk/`、`risk_detail/`：浏览与查看风险相关评论详情 |
| **管理后台 (`admin_panel`)** | 独立管理员登录；用户与套餐（ModelPlan / UserPlan）、原始文本（RawText）导入与列表、标注列表、待预测队列与预测结果、用量日志、模型列表与运行时融合配置、算法对比与实验记录、自检与训练中心、导出审计日志等 |

详细路由以 `DjangoWeb/urls.py` 为准；旧路径多已**重定向**到 `admin_panel/*` 或新名称，避免书签失效。

---

## 技术栈

- **Python**：建议 **3.7**（与 `requirements.txt` 中 Django、scikit-learn 等版本匹配；升级需自行回归）
- **Web**：Django **2.2.x**，模板 + **Bootstrap / Layui**，图表 **ECharts**（静态资源在 `static/`）
- **数据库**：默认 **SQLite**（`db/db.sqlite3`，可在 `DjangoWeb/config.py` 切换 **MySQL**，并配合 `utils/connect_mysql.py`）
- **机器学习**：**scikit-learn**、jieba、numpy、pandas、joblib 等
- **深度学习（项目内联预测）**：**PyTorch** 加载 `model/textcnn/*.pt`、`model/textrcnn/*.pt`（见 `algorithm/torch_nlp.py`）；`requirements.txt` 中 TensorFlow/Keras 为历史依赖，主预测链路以代码实际引用为准
- **可选 HF 模型**：`transformers`、`torch`、`safetensors`，预测逻辑见 `algorithm/hf_transformers_predict.py`，权重可放在 `Best_Modle/` 或通过脚本下载

---

## 算法与预测融合

- **Sklearn 角色**：`svm`、`knn`、`rf`、`dt`、`lr` — TF-IDF 向量器 + 对应分类器，产物路径约定见 `algorithm/model_paths.py`（如 `model/svm/model_svm.pkl`）。
- **深度学习**：`textcnn`、`textrcnn` — 字符 ID 序列 + `.pt` 权重 + 词表 JSON。
- **规则**：`algorithm/rules.py` 词典规则分。
- **融合**：`algorithm/fusion.py` 与 `PredictService`（`app/services/predict_service.py`）按 **SystemConfig / 运行时配置** 中的权重与阈值做加权融合，并写入 `PredictionResult`。
- **Hugging Face**：模型类型 `hf_depr_bin`（DistilBERT 抑郁二分类）、`hf_mh_4cls`（BERT 心理健康四分类）；本地目录说明见 `Best_Modle/README.txt`。

默认运行时配置示例（可在后台「运行时配置」中调整）包括：各算法开关、权重、`threshold_high_risk` / `threshold_alert`、TextCNN 超参数路径等 — 以 `PredictService.DEFAULT_RUNTIME_CONFIG` 为基准。

**模型登记（`ModelInfo`）**：`model_type` 涵盖上述类型及 `fusion`、`rule`；`listed_for_users` 控制是否出现在前台并对套餐/用户「可用算法」开放。

---

## 数据与业务对象（简述）

- **`Comment`**：贴吧评论业务表（楼层、昵称、内容、`emotion` 等）。
- **`RawText` / `TextSource` / `CleanText` / `AnnotationRecord` / `LabelDefinition`**：从接入、清洗、标注到标签定义的管线；`RawText` 状态流转见 `get_status_flow()`（未清洗 → … → 已预测 / 已复核）。
- **`PredictionResult`**：单条原始文本上某次预测的结果详情（含 `detail_json`）。
- **`SystemConfig`**：键值配置，承载融合权重、模型路径等。
- **`ModelInfo` / `ModelEvaluation` / `AlgorithmExperimentRecord` / `FusionConfigPreset`**：模型版本、评估、实验记录、融合配置快照。
- **套餐与配额**：`ModelPlan`、`UserPlan`、`QuotaUsageLog`、`PredictionUsageLog` 等（详见 `app/models.py`）。

数据集与目录约定见 `datasets/depression_nlp/README.txt`（中英文数据、`text,label` 列、splits 比例等）。

---

## 目录结构（核心）

| 路径 | 作用 |
|------|------|
| `DjangoWeb/` | 项目配置包：`settings.py`、`urls.py`、`config.py`、`wsgi.py` |
| `app/` | 主应用：模型、视图、服务、`management/commands/` |
| `algorithm/` | 预测、融合、路径约定、HF 推理、sklearn/torch 封装 |
| `templates/`、`static/` | 模板与静态资源 |
| `model/` | 训练产物：各算法子目录、`num2name.json`、`model_training_registry.json` |
| `Best_Modle/` | HF 权重（DistilBERT / BERT 等），`download_hf_models.py` |
| `data/` | 爬虫 CSV、导入用数据 |
| `datasets/depression_nlp/` | 组织好的 NLP 数据集（en/zh） |
| `scripts/` | 下载划分、训练（sklearn / char torch）、迁移与工具脚本 |
| `db/` | SQLite 文件默认位置 |
| `docs/` | 项目说明、算法与开发环境文档（与代码同步补充） |

训练成功后可由 `scripts/train_single_sklearn.py`、`scripts/train_char_torch.py` 等写回 `model/model_training_registry.json`，供后台展示训练次数与指标（见 `algorithm/model_training_registry.py`）。

---

## 环境与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

若需使用 **HF 目录预测**（`hf_depr_bin` / `hf_mh_4cls`）：

```bash
pip install transformers torch safetensors
```

Large 权重可设镜像后使用 `Best_Modle/download_hf_models.py`（说明见 `Best_Modle/README.txt`）。

### 2. 数据库迁移

```bash
python manage.py migrate
```

### 3. 启动开发服务器

```bash
python manage.py runserver 127.0.0.1:8000
```

或执行根目录 **`运行.py`**（内部调用同上命令）。

### 4. 访问入口

- 首页：`http://127.0.0.1:8000/`
- 用户登录：`/login/`
- 管理员登录：`/admin_panel/login/`

（生产环境请修改 `ALLOWED_HOSTS`、关闭 `DEBUG`、更换 `SECRET_KEY`，并勿将密钥提交到公开仓库。）

---

## 安全与合规提示

- `settings.py` 中 **CSRF 中间件默认被注释**，仅适合受信内网或本地调试；对外部署前应恢复并完整配置安全项。
- 用户密码等为**明文或弱哈希存储在自定义 `User` 模型**中，生产环境需改造为 Django 认证体系或安全哈希。
- 爬取贴吧须遵守平台规则与法律法规，本 README 不指导规避反爬或超范围采集。

---

## 大文件与 Git

部分数据与模型体积较大，仓库可能使用 **Git LFS**；克隆后若需完整数据请确保已安装 [Git LFS](https://git-lfs.github.com/) 并执行 `git lfs pull`。

---

## 文档与署名

- 更细的算法与配置说明见 **`docs/算法逻辑与配置说明.md`**。
- 文件结构与开发环境见 **`docs/开发环境与项目文件结构说明.md`**。
- `settings.py` 头部保留原作者联系说明（赵有才；邮箱 / 微信见该文件注释）。

---

## 许可证与第三方模型

- 项目整体许可证以仓库内声明为准（若未声明，使用前请自行评估）。
- `Best_Modle` 内各 HF 模型的许可证以对应 **Hugging Face 模型卡**为准（如 MIT、Apache-2.0 等）。

---

*本文档依据当前仓库中的 `urls.py`、`models.py`、`algorithm/`、`app/services/predict_service.py` 及 `docs/` 归纳；若代码变更，请以实际实现为准。*
