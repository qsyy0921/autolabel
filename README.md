# autolabel

视频自动标注平台设计与原型项目。

当前阶段范围收敛为：

- 输入数据为视频
- 自动标注任务优先支持：
  - 目标检测
  - 实例分割
- 平台形态为：
  - 模型预打标
  - 人工复核
  - 质量闭环
  - 可持续自进化

本文档同时作为项目总览和第一阶段架构设计说明。

## 1. 目标

设计一个面向视频数据的半自动标注平台，优先解决以下问题：

- 视频逐帧人工标注成本高，完全手工不可持续
- 纯模型自动标注结果不稳定，需要人工复核闭环
- 视频检测和实例分割链路高度相似，但不应实现成两套系统
- 后续需要随着标注数据增加，不断提升自动标注效果

第一阶段定位：

- 跑通视频目标检测自动标注工作流
- 跑通视频实例分割自动标注工作流
- 为后续自进化保留稳定的数据结构和系统边界

## 2. 设计原则

### 2.1 两条工作流，共用一套底座

目标检测与实例分割的大部分流程相同：

- 视频导入
- 镜头切分
- 关键帧选择
- 自动预测
- 人工审核
- 导出数据

真正的差异只有：

- 检测工作流输出 `box`
- 分割工作流输出 `box + mask`

因此平台应该采用：

`共享视频处理底座 + 模板化工作流 + 不同审核编辑器`

### 2.2 Prediction 和 Annotation 必须分离

系统必须分开保存：

- `Prediction`：模型自动预测结果
- `Annotation`：人工最终确认结果
- `CorrectionLog`：人工相对模型做的修改

这决定后续是否能做：

- 错误分析
- 主动学习
- 特征库构建
- 伪标签蒸馏
- 增量训练
- 模型回滚

### 2.3 先检索增强，再增量训练

第一阶段不建议直接做复杂训练平台。

更稳的路径是：

1. 记录人工修正
2. 构建特征库
3. 实现相似实例召回和难例优先
4. 样本足够后再微调项目专用模型

### 2.4 工作流可扩展，但第一版不追求复杂低代码

第一阶段建议只做：

- 预置工作流模板
- 参数化执行
- 节点状态展示
- 中间结果缓存

不建议一开始就做：

- 完整拖拽式低代码画布
- 任意循环和复杂分支
- 在线训练编排平台

## 3. 业务范围

### 3.1 输入

- 单个视频文件
- 视频目录
- 项目级视频批次

### 3.2 输出

针对关键帧或传播后的帧，输出：

- 目标检测：
  - 类别
  - 边界框
  - 置信度

- 实例分割：
  - 类别
  - 边界框
  - mask 或 polygon
  - 置信度

### 3.3 用户操作

- 上传视频
- 配置项目标签体系
- 选择工作流模板
- 查看自动预测结果
- 人工确认、修改、删除、补标
- 导出训练集

## 4. 端到端主流程

```text
Video Upload
-> Scene Split
-> KeyFrame Select
-> Detect
-> Optional Segment
-> Human Review
-> Final Annotation
-> Correction Log
-> Feature Store
-> Export / Learn
```

### 4.1 目标检测工作流

```text
视频
-> 镜头切分
-> 关键帧选择
-> 检测模型
-> 人工审核(Box Editor)
-> 最终标注
```

### 4.2 实例分割工作流

```text
视频
-> 镜头切分
-> 关键帧选择
-> 检测模型
-> SAM2 分割
-> 人工审核(Mask Editor)
-> 最终标注
```

## 5. 系统架构

建议按六层组织。

### 5.1 项目与数据层

负责：

- 用户与工作区
- 项目
- 视频资源
- 抽帧结果
- 关键帧集合
- 标签体系
- 导出数据集

核心原则：

- 原始视频统一进入 `VideoAsset`
- 下游工作流统一消费 `Frame` 和 `KeyFrame`
- 不让模型节点直接依赖原始文件路径

### 5.2 工作流引擎层

负责：

- 工作流模板定义
- 节点编排
- 参数注入
- 任务队列
- 中间结果缓存
- 失败重试
- 工作流版本管理

第一阶段只需要支持：

- 线性步骤
- 可选步骤
- 运行日志

### 5.3 模型运行层

负责：

- 调用镜头切分模型
- 调用关键帧特征模型
- 调用检测模型
- 调用分割模型
- 调用检索索引

这一层必须做成节点适配器，而不是散落在业务控制器里的脚本调用。

### 5.4 标注结果层

负责：

- Prediction 存储
- Annotation 存储
- CorrectionLog 存储
- 审核状态管理
- 导出格式转换

### 5.5 自进化与学习闭环层

负责：

- Feature Store
- Active Learning
- 伪标签蒸馏
- 项目专用微调
- 候选模型评估
- 模型注册与灰度发布

### 5.6 前端工作台

负责：

- 项目与数据管理
- 工作流运行控制
- 审核画布
- 结果筛选和回查
- 进化状态展示

## 6. 工作流设计

### 6.1 工作流模板一：视频目标检测

```text
Video
-> Scene Split
-> KeyFrame Select
-> Detect
-> Review
-> Export
```

节点建议：

- `scene_detect`
- `keyframe_select`
- `detect`
- `human_review_detection`
- `export_detection`

### 6.2 工作流模板二：视频实例分割

```text
Video
-> Scene Split
-> KeyFrame Select
-> Detect
-> Segment
-> Review
-> Export
```

节点建议：

- `scene_detect`
- `keyframe_select`
- `detect`
- `segment`
- `human_review_segmentation`
- `export_segmentation`

### 6.3 Workflow Schema

```yaml
name: video_instance_segmentation_v1
task_type: instance_segmentation
version: 1

steps:
  - id: split
    type: scene_detect.transnetv2

  - id: keyframes
    type: keyframe.select
    depends_on: [split]
    params:
      max_per_shot: 3

  - id: detect
    type: detect.groundingdino
    depends_on: [keyframes]
    params:
      prompts: ["gear", "bolt", "shaft"]

  - id: segment
    type: segment.sam2
    depends_on: [detect]

  - id: review
    type: review.mask_editor
    depends_on: [segment]
```

## 7. 模型选型

第一阶段建议只接角色清晰、复用价值高的模型。

### 7.1 镜头切分

- `PySceneDetect`
  - 稳定、成熟、部署轻

- `TransNetV2`
  - 复杂转场更稳

### 7.2 关键帧特征

- `DINOv2`
  - 视觉特征、聚类、多样性选择

- `CLIP`
  - 语义检索、后续 prompt 相关召回

### 7.3 检测

- `GroundingDINO`
  - 开放词汇检测
  - 适合按标签词或 prompt 找目标

- `YOLO / YOLO-World`
  - 已知类别检测更快
  - 适合批量推理和项目专用微调

### 7.4 分割

- `SAM2`
  - 根据框或点生成实例 mask
  - 作为实例分割默认分割器

### 7.5 检索

- `FAISS`
  - 管理相似实例搜索
  - 用于一标多找、去重、难例采样

## 8. 自进化设计

第一阶段必须把自进化底座设计好，即使训练能力后置。

### 8.1 自进化目标

平台需要随着标注数据增加，持续提升：

- 检测召回率
- 检测精度
- mask 质量
- 人工审核效率
- 相似实例补全能力

### 8.2 自进化闭环

```text
Prediction
-> Human Correction
-> Correction Log
-> Feature Store
-> Active Learning
-> Pseudo Label / Distill
-> Fine-tune Candidate
-> Evaluation
-> Registry / Rollout
-> Back to Workflow
```

### 8.3 学习信号

系统至少要记录三类信号：

1. 接受
   - 模型预测正确或基本正确

2. 修正
   - 框位置变了
   - mask 边界改了
   - 类别改了

3. 补标或删除
   - 模型漏检
   - 模型误检

这些信号决定：

- 哪些样本进入特征库
- 哪些样本成为难例
- 哪些样本进入训练集

### 8.4 第一阶段的进化重点

建议顺序：

1. `CorrectionLog`
2. `Feature Store`
3. `Active Learning`
4. `Pseudo Label / Distillation`
5. `Fine-tuning`

### 8.5 维护规则

必须遵守：

- 新模型只作为候选版本，不直接覆盖线上
- 训练与发布必须版本化
- 评估不只看 mAP，也看人工审核效率
- Feature Store 和训练数据必须带：
  - `project_id`
  - `workflow_version`
  - `model_version`

## 9. 数据模型

### 9.1 Project

```text
id
workspace_id
name
description
task_type
label_schema_json
created_at
updated_at
```

### 9.2 VideoAsset

```text
id
project_id
uri
hash
duration_ms
fps
width
height
created_at
```

### 9.3 Frame

```text
id
video_id
frame_index
timestamp_ms
image_uri
is_keyframe
created_at
```

### 9.4 WorkflowRun

```text
id
project_id
workflow_name
workflow_version
status
params_json
started_at
finished_at
```

### 9.5 Prediction

```text
id
run_id
frame_id
category
bbox_json
mask_uri
score
model_name
model_version
raw_output_json
created_at
```

### 9.6 Annotation

```text
id
frame_id
prediction_id
category
bbox_json
mask_uri
source
annotator_id
status
created_at
updated_at
```

### 9.7 CorrectionLog

```text
id
prediction_id
annotation_id
change_type
before_json
after_json
user_id
created_at
```

### 9.8 FeatureItem

```text
id
project_id
frame_id
annotation_id
category
crop_uri
mask_uri
clip_embedding_uri
dinov2_embedding_uri
quality_score
model_version
created_at
```

## 10. API 与服务边界

即使第一版先做单体，也建议按服务边界组织代码：

- `auth`
- `projects`
- `datasets`
- `workflow`
- `inference`
- `annotations`
- `features`
- `exports`
- `models`

推荐 API：

- `POST /projects`
- `POST /projects/{id}/videos`
- `POST /projects/{id}/workflow-runs`
- `GET /workflow-runs/{id}`
- `GET /frames/{id}/predictions`
- `POST /annotations`
- `POST /corrections`
- `POST /exports`

## 11. 前端架构

第一阶段建议三个主页面。

### 11.1 项目页

展示：

- 视频数量
- 已抽帧数量
- 已完成工作流数量
- 待审核帧数量
- 当前模型版本

### 11.2 工作流运行页

能力：

- 选择视频批次
- 选择工作流模板
- 配置节点参数
- 运行工作流
- 查看每个节点状态

### 11.3 标注工作台

统一布局：

- 左侧：帧列表和筛选条件
- 中间：画布
- 右侧：实例列表、类别编辑、模型建议

审核器按任务切换：

- detection 使用 `Box Editor`
- segmentation 使用 `Mask Editor`

## 12. 可维护性要求

### 12.1 统一节点接口

```python
class WorkflowNode:
    name: str
    version: str

    def run(self, context: dict, inputs: dict, params: dict) -> dict:
        ...
```

### 12.2 模型适配器插件化

禁止在业务控制器里直接写模型调用。

建议适配器：

- `GroundingDINOAdapter`
- `YOLOAdapter`
- `SAM2Adapter`
- `DINOv2Adapter`
- `CLIPAdapter`
- `FAISSAdapter`

### 12.3 中间结果可缓存

必须支持缓存：

- 镜头切分结果
- 关键帧结果
- 检测 proposal
- 分割 mask
- embedding

### 12.4 项目级隔离

默认要求：

- 项目只能看到自己的视频、结果、特征和工作流运行记录
- 特征是否跨项目共享必须显式配置

### 12.5 评估先于发布

任何新模型上线前都必须完成：

- 固定验证集评估
- 与现网模型对比
- 人工审核效率对比

## 13. 推荐落地顺序

### 阶段 1：最小可用链路

- 视频导入
- 镜头切分
- 关键帧选择
- 检测工作流
- Box 审核
- 导出检测数据

### 阶段 2：接入实例分割

- 在检测结果上接 SAM2
- Mask 审核
- 导出分割数据

### 阶段 3：补齐学习闭环

- CorrectionLog
- Feature Store
- Active Learning
- 候选模型评估与注册

### 阶段 4：项目专用优化

- 检测器微调
- 相似实例检索
- 伪标签蒸馏

## 14. 技术栈建议

如果从零开始：

- 前端：React + TypeScript + Vite
- 后端：FastAPI + Python
- 数据库：PostgreSQL，MVP 可先用 SQLite
- 队列：Celery/RQ，MVP 可先用 BackgroundTasks
- 文件存储：本地目录，后续接 MinIO/S3
- 向量索引：FAISS

## 15. 结论

第一阶段不应该把系统做成“一个标注页面接几个模型”，而应该做成：

`视频数据层 + 模板化工作流引擎 + 检测/分割节点适配器 + 审核工作台 + 自进化闭环底座`

这样既能快速跑通视频目标检测与实例分割自动标注，也能为后续持续提升自动标注能力留下稳定演进路径。
