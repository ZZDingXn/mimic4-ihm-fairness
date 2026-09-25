# MIMIC-IV 院内死亡预测与群体公平性

这个仓库整理了从 MIMIC-IV 3.1 队列构建、逻辑回归（LR）与 channel-wise LSTM 训练，到群体评价和偏差缓解的研究代码，以及已经完成的汇总结果和图表。想先看研究发现，可以从下面的“目前完成了什么”和“图表怎么读”开始；准备复现时，再看数据接口与运行入口。

这里最关心的问题是：**一个总体表现较好的模型，是否也能以相近的比例识别不同群体中的死亡患者；如果调整这种差异，又会增加多少误报、牺牲多少效用，或者改变概率校准？** 这些问题需要分别回答，不能用一个“公平性分数”概括。

本次整理只使用现有实验结果，没有重新训练或重新运行正式 bootstrap。英文新旧稿、中文稿及其 LaTeX 源文件和论文 PDF 留在原研究目录，不放进这个仓库。`figures/` 中的 PDF 是独立图表的矢量文件，不是论文 PDF。

## 目前完成了什么

正式队列有 **36,648 次 ICU stay、31,459 名患者、4,469 个死亡事件**。研究单位是 ICU stay；同一患者可以贡献多次符合条件的 stay，但不会跨越数据划分。

| 划分 | ICU stays | 患者 | 死亡事件 |
|---|---:|---:|---:|
| train | 25,502 | 21,927 | 3,072 |
| validation | 5,610 | 4,813 | 712 |
| test | 5,536 | 4,719 | 685 |

两种原模型、保险与族裔各自的四次重加权训练，以及 20 个评价条件均已完成。20 个条件来自“两模型 × 两个目标属性 × 五种决策方案”；其中原模型的固定阈值和统一阈值在两个目标属性下重复出现，是为了提供同一套对照，并不代表独立训练了 20 个模型。

| 原模型 test 结果 | AUROC | AUPRC | 0.5 阈值死亡召回率 |
|---|---:|---:|---:|
| LR | 0.8527 | 0.4789 | 24.09% |
| LSTM | 0.8646 | 0.5196 | 27.59% |

LSTM 的总体判别指标较高，但两个模型在 0.5 阈值下都漏掉了多数死亡事件。因此，纠偏比较首先加入验证集优化的统一阈值，避免把“整体降低阈值带来的检出增加”误认为公平约束独有的作用。

现有结果没有给出“某种纠偏方法普遍有效”的结论。12 项目标属性 TPR 差距变化的配对区间均包含零；这说明当前样本下的证据仍不确定，并不证明所有方法没有作用。机会均等后处理在四个模型—属性组合中有三个缩小了 TPR 差距点估计，但没有一个提高最差组 TPR。差距变小可能伴随着部分群体检出率下降，必须结合绝对表现、误报和效用一起读。完整数值保留在汇总表中，不只展示较好看的结果。

历史上使用过 500-stay 工程子集和两轮 LSTM 训练来验证管线；它们属于早期运行检查，不能代替上面的正式结果。旧 README 中“评价仍在进行”和以官方死亡标志作为最终标签的描述，也已按当前代码与冻结协议更新。

## 文件放在哪里

```text
mimic4-ihm-fairness/
├── README.md                     # 本文：研究背景、目录、图表说明与运行入口
├── requirements.txt              # Python 依赖，含 Fairlearn 0.13.0
├── pytest.ini
├── mimic4_ihm/                   # 队列、特征、原模型训练和基线评价
├── tests/                        # 基础管线测试，使用合成数据
└── fairness_study/
    ├── protocol.md               # 冻结纠偏协议；本机路径已替换为占位符
    ├── src/                      # 重加权、后处理、评价、审计与制图
    ├── tests/                    # 纠偏和统计计算的必要测试
    ├── results/                  # 经筛选的汇总数据，无患者级预测
    │   └── baseline/             # 原基线图表所用的历史汇总结果
    └── figures/                  # 正式图表；_en 英文，_zh 中文
        └── revised/              # 新版英文亚组图：同一数据，改善排版
```

运行代码后可能生成 `experiments/`、`tem/` 或 `results/plots/`。前两者用于本地模型、逐患者输出、临时检查和缓存；`results/plots/` 是评价器的制图输出位置之一，仓库中现成图表统一放在 `figures/`。这些用途不要混淆，也不要因为文件由本项目生成就直接上传。

核心模块中，`constants.py` 保存变量、item ID、类别与编码表；`prepare.py` 处理队列、标签、事件清洗和编码；`features.py` 生成 LR 特征；`model.py` 定义 LSTM；`train.py` 负责训练和预测导出；`evaluate.py` 负责原基线评价。新增方法分别在 `fairness_study/src/reweight.py`、`postprocess.py` 和 `evaluate.py`，图表入口是 `redraw_figures.py`。

## 图表怎么读

以下文件都在 [`fairness_study/figures/`](fairness_study/figures/)。通常每张图都有 `_en.png`、`_en.pdf`、`_zh.png`、`_zh.pdf` 四个版本；它们表达同一分析，PNG 适合预览，PDF 适合排版。表中的文件名省略语言和格式后缀。

| 图表 | 在什么情况下生成、数据来自哪里 | 具体看什么 |
|---|---|---|
| `cohort_flow` | 冻结队列完成纳排后，由 `results/baseline/cohort_flow.csv` 绘制 | 各步骤留下多少 stays，最终研究人群如何形成。它描述样本选择，不反映模型性能；也不要把 stay 数当成独立患者数。 |
| `baseline_performance` | 两个原模型在同一 test 上评价；来自 `results/baseline/overall_metrics.json`，简表见 `baseline_summary.csv` | 并列比较 AUROC、梯形积分 AUPRC 和固定 0.5 阈值的死亡召回率。前两项是排序能力，召回率依赖阈值；高 AUROC 不意味着默认阈值能识别多数死亡。AUROC/AUPRC 误差线来自原先 10,000 次 stay 行级 bootstrap；召回率没有补造区间。 |
| `baseline_subgroups` | 原模型按保险和族裔分组；来自 `results/baseline/subgroup_metrics.csv` 与 `calibration_in_large.csv` | 上排比较亚组 AUROC，下排比较平均预测风险减实际死亡率，以百分点表示。负值表示平均风险低估；接近零只表示平均偏差小，不能证明整条校准曲线良好。误差线沿用原基线 stay 级 bootstrap。读亚组 AUROC 时也应查对应人数、死亡数和事件率。 |
| `revised/baseline_subgroups_en` | 新英文稿修订时，使用上面完全相同的汇总估计重新安排字体、标签和面板；另提供 SVG | 内容和统计口径未变，只改善可读性。优先使用此版英文亚组图；旧版和中文版仍保留，便于核对。 |
| `group_tpr` | 五种冻结策略应用于 test 后生成，数据见 `policy_metrics.csv` 与 `bootstrap_confidence_intervals.csv` | 四个面板分别比较模型与纠偏目标；TPR 的分母是本组死亡事件。除了看组间距离，还要看最差组是否提高，是否只是较好组下降。后处理按提示概率计算期望混淆计数，点不代表某一次随机二分类结果；误差线来自新增的 10,000 次患者聚类 bootstrap。 |
| `group_fpr` | 与 TPR 图同一批策略、相同 test、相同统计口径 | FPR 的分母是本组未死亡 stays，表示这些样本被提示为高风险的比例。结合 TPR 图判断检出差异缩小时，误报是否增加、转移到哪些群体。均等化赔率约束在 validation 拟合，不能保证 test 上各组 FPR 或 TPR 完全重合。 |
| `fairness_utility_tradeoff` | 由 `fairness_gaps.csv` 和总体 `policy_metrics.csv` 汇总各策略 | 横轴为最大与最小组 TPR 的差距，越小表示所选标准下越接近；纵轴为 balanced accuracy，越大表示这一效用指标越高。这是离散策略点的比较，并非连续优化的 Pareto 前沿，也不含置信区间；不同面板的坐标范围需要分别读。 |
| `overall_calibration` | 原模型与两种重加权模型的 test 死亡风险分数；数据见公开版 `calibration_bins.csv` | 分箱点与 LOWESS 曲线比较预测风险和观察死亡率，理想状态在对角线上。分箱使用 10 个指数分位区间（分位边界 q^(1/5)），LOWESS 的 frac=0.5，导出为固定 0.005 风险网格上的显示值，不是逐患者风险坐标。曲线在对角线上方表示风险低估。图没有置信带；阈值后处理不改变死亡风险分数，所以没有把其提示概率画成新的风险校准曲线。 |
| `training_curves` | 保险和族裔重加权 LSTM 各训练 100 轮后，从轮次级损失提取 `training_curves.csv` | 对照加权训练 BCE 与未加权 validation BCE，竖线标记最低未加权验证损失对应的 checkpoint：保险第 22 轮、族裔第 14 轮。两条曲线使用不同权重，不能把它们的距离直接解释为通常意义的泛化差距；单次训练也不能说明训练种子间稳定性。 |

TPR/FPR 图上排是 LSTM、下排是 LR，左列为族裔、右列为保险。颜色区分五种策略；目标属性分开纠偏，不代表同时解决了其他属性的差异。性别、年龄等非目标属性的结果仍在汇总表中，便于检查影响有没有转移。

### 汇总表与图表的对应关系

所有表都在 [`fairness_study/results/`](fairness_study/results/)。`policy_metrics.csv` 按“模型 × 目标属性 × 策略 × 评价属性 × 群体”保存人数、死亡数、混淆计数、TPR、FPR、PPV、balanced accuracy、提示率、每千 stays 误报与漏报、AUROC、AUPRC、Brier score 和平均校准偏差。后处理的混淆计数可能是小数，因为它们是期望计数。

`fairness_gaps.csv` 保存 TPR/FPR 差距、均等化赔率差距、最差组 TPR 和提示率差距；`bootstrap_confidence_intervals.csv` 是新增患者聚类区间；`paired_differences_vs_global_threshold.csv` 将每种方法与原模型的优化统一阈值进行配对比较，读差值时先确认指标方向。`model_comparison_patient_cluster.csv` 比较两种原模型，`report_comparison_summary.csv` 是方便阅读的目标属性摘要，不能代替完整表。

`weight_audit_insurance_group.csv` 与 `weight_audit_ethnicity_group.csv` 保存 train 中“群体 × 结局”的人数、概率和对应权重，两模型使用同一套权重；这些表不含逐样本权重。`training_configuration.json` 保存四次训练的配置，已去掉本机输入输出路径。

`cohort_counts.csv` 和 `cohort_demographics.csv` 提供队列规模及各划分的人口构成；`baseline_summary.csv` 是原模型总体指标摘要。`evaluation_manifest.json` 记录抽样次数、种子、软件版本与评价口径，已移除本机路径。`temporal_label_audit.json` 只保留时间审计的汇总计数。公开版 `calibration_bins.csv` **仅保留总体层面**的分箱和曲线显示数据，不包含细分小组的校准导出；完整本地研究结果没有因此被修改。

## 研究口径与复现条件

### 数据怎样连接

`subject_id → hadm_id → stay_id` 分别对应患者、住院和 ICU stay。`hosp/patients` 提供患者信息，`hosp/admissions` 提供住院时间和人口属性，`icu/icustays` 定位 ICU 记录；`chartevents` 通过 stay 连接，`labevents` 通过住院及时间窗连接。数据根目录需要以下七张表，支持原始 `.csv.gz` 或 `.csv`，不必为了运行先删掉压缩文件：

```text
<MIMIC_DIR>/
├── hosp/{patients,admissions,d_labitems,labevents}.csv[.gz]
└── icu/{icustays,d_items,chartevents}.csv[.gz]
```

既有纳排规则保留成年人、住院期间仅一次 ICU stay、首末 careunit 相同、至少 48 小时 ICU stay，并要求有目标变量观测。首末 careunit 相同只是当前管线的转科筛选条件，不能证明中间从未转科。数据采用百万行分块、SQLite 中间存储与 NumPy memmap 处理，不要求把完整事件表一次读入内存。

主要死亡标签依据住院入院至出院的精确时间窗内是否记录死亡。官方 `hospital_expire_flag` 用于对账，不直接替代主标签：当前队列主标签有 4,469 个事件，官方标志有 4,785 个，存在 316 条差异。标签审计与原始记录留在受控本地环境；公开仓库只放汇总结果。

时间审计还发现 152 个死亡事件的记录时间不晚于 48 小时输入窗口结束（train 100、validation 28、test 24）。队列与标签没有事后重定义，因此本研究不能被理解为严格的“所有患者均存活到第 48 小时，再预测未来死亡”的 landmark 研究，更不是 ICU 入科即时预警或临床部署验证。

### 两种模型如何比较

两模型共享队列、标签和患者划分，但输入表示不同。LSTM 使用 17 个临床变量的小时序列，形状为 `(N, 48, 76)`，其中 59 个编码特征加 17 个缺失掩码；LR 从原始稀疏事件的 7 个时间片段计算 6 类统计量，共 `17 × 7 × 6 = 714` 个特征，不是简单展平 LSTM 输入。

变量映射保留 MIMIC-IV 的 item ID 适配（包括 227242），并处理 FiO2、SpO2、温度、身高、体重的单位转换，以及 GCS 回退和同时间总分。极端但可能具有临床意义的数值保留并审计，不根据 test 表现临时裁剪；详细定义见 `constants.py` 和 `prepare.py`。

患者划分种子为 49297：先抽取 15% 患者作 test，再从剩余患者抽取 18% 作 validation，不按结局分层。LR 使用 train 上拟合的均值填补与标准化，L2、C=0.001、lbfgs、max_iter=1000、random_state=42。LSTM 使用 dim=8、size_coef=4、dropout=0.3、batch_size=8、Adam 学习率 0.001，训练 100 轮；以最低未加权 validation BCE 选择 checkpoint，并列取较早轮次。

### 纠偏改了什么

五个条件依次为：原模型固定 0.5 阈值、原模型验证集优化统一阈值、重加权模型优化统一阈值、机会均等后处理、均等化赔率后处理。统一阈值遍历可实现的分割点，最大化 validation balanced accuracy，并列取较高阈值。

重加权只用 train 计算 `w(a,y)=P(a)P(y)/P(a,y)`，再归一化为样本均值 1；LR 通过拟合权重、LSTM 通过逐样本 BCE 实现，不另加过采样或类别权重。它调整组别—结局组合的训练贡献，并不直接优化 TPR 差距。

Fairlearn 0.13.0 的 `ThresholdOptimizer` 分别使用 `true_positive_rate_parity` 和 `equalized_odds`，共同参数为 `objective="balanced_accuracy_score"`、`grid_size=1000`、`flip=False`、`prefit=True`。规则只在 validation 拟合，冻结后评价 test；`MetricFrame` 用于分组评价。人口统计均等仅作为提示率差异报告，不作为主要优化目标。

死亡风险概率、策略提示概率与一次随机二分类决策是三个不同对象。主要策略指标从提示概率计算期望混淆计数；例如 PPV 是期望 TP 除以期望提示总数，不是所有随机实现 PPV 的平均值。后处理不改变原风险分数，所以不能宣称提高原分数 AUROC 或修复概率校准。

新增比较使用 10,000 次、种子 49297 的患者聚类配对 bootstrap：抽中患者时保留其全部 stays，各方法共享同轮抽样。模型和策略固定，不在每次抽样重新训练；这些区间不涵盖完整训练波动。原基线图仍保留历史 stay 级区间，新比较中的基线已经按患者聚类重算，两者不要混用。

原 test 基线在扩展协议冻结前已经被查看，validation 也重复用于模型与策略选择，因而这是已有研究的冻结扩展，不是完全未接触 test 的前瞻性验证。尚未进行多种子训练、外部验证或因果公平性分析；群体统计差异不能直接推导为歧视机制或可避免死亡数。更完整的预定规则见 [`protocol.md`](fairness_study/protocol.md)。

## 怎样运行

建议单独建 Python 环境，在仓库根目录执行下面的命令。`requirements.txt` 记录研究环境的依赖版本；PyTorch 的 CPU/CUDA 构建仍需与本机设备匹配。只浏览汇总结果不需要 MIMIC 数据访问权限；重新构建队列、训练或评价则需要自行取得并在本地配置合规的数据访问权限，仓库不提供原始数据下载。

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

上游 benchmark 的 header 对齐测试是可选项：设置环境变量 `MIMIC3_DISCRETIZER_CONFIG` 指向自己取得的 `discretizer_config.json` 才会运行，否则明确跳过。常规测试使用合成样本，不依赖真实患者文件。

**只重绘已有汇总图，不训练、不重算 bootstrap：**

```powershell
python -m fairness_study.src.redraw_figures
python -m fairness_study.src.baseline_figure_revised
```

中文图优先使用 SimHei/SimSun；其他系统需准备相应中文字体或调整绘图字体列表。新的图像文件会更新 `fairness_study/figures/`，缓存写入 `fairness_study/tem/`。

**完整复现的入口**如下。这些是供后续使用的命令，本次整理没有执行。先用 `--help` 查看数据构建参数，并把占位路径换成自己的本地目录：

```powershell
python -m mimic4_ihm.build_dataset --help
python -m mimic4_ihm.train --model logistic_regression --data-dir <DATA_DIR> --output-dir <DATA_DIR>/lr
python -m mimic4_ihm.train --model cw_lstm --data-dir <DATA_DIR> --output-dir <DATA_DIR>/cw_lstm --device cuda --epochs 100
python -m fairness_study.src.run_training --data-dir <DATA_DIR> --device cuda
python -m fairness_study.src.postprocess --data-dir <DATA_DIR> --study-dir fairness_study
python -m fairness_study.src.evaluate --study-dir fairness_study --iterations 10000 --seed 49297
```

`<DATA_DIR>` 是已完成准备的数据目录，包含 metadata、数组、config 和两种原模型的预测。重加权、后处理、统一评价的输出都留在本地研究目录。`finish_experiments.py` 是原冻结流程的串行收尾工具；其中 `align_baseline_train.py` 专门修复历史 LSTM train 导出顺序，并校验当时 checkpoint 的固定哈希。它不会更改原 validation/test 预测，也不是任意新训练模型都应调用的通用步骤。新数据复现应使用上面各入口，不要照搬历史 checkpoint 修复。

## 准备推送到云端时，哪些内容可以放进来

这个目录是面向后续云端仓库整理的公开材料区，**不是完整的本地研究目录备份**。本次沿用原仓库的数据保护边界：可以保留代码、合成测试、方法协议、经检查的群体汇总结果和由汇总结果生成的图；原始 MIMIC 数据及患者级派生产物继续留在受控本地位置。整理文件本身不等于批准所有未来生成文件都能公开，实际推送仍需遵守所使用数据的访问协议和机构要求。

不要推送以下内容：

- 原始 `hosp/`、`icu/` CSV/CSV.GZ，SQLite 中间库及其附属文件；
- 含 `subject_id`、`hadm_id`、`stay_id` 的 metadata、样本索引、逐患者特征/标签/风险/决策，NumPy 数组和逐样本训练权重；
- checkpoint、模型权重、序列化填补器/标准化器、pickle/joblib 文件；
- 原始训练日志、内部工作报告、本机路径配置、缓存、环境目录和凭据；
- 论文来源 PDF、第三方参考代码副本，以及本次明确排除的中英文论文源码、论文 PDF 和编译中间文件。

这里的“数据”仅指 `results/` 中有明确统计粒度的汇总导出；总体校准也只保留分箱与固定网格的曲线显示值，没有逐患者坐标。训练曲线只保留轮次损失和所选轮次，不保留原始日志。新增群体切分或导出细粒度分箱时，应重新检查披露风险，不能因为旧表已经通过检查就自动沿用。

`.gitignore` 默认排除 CSV、数组、数据库、模型和本地运行目录，只对本次检查过的汇总 CSV 按完整文件名逐项放行。它是一道便利的边界，不是内容审查器：如果将患者数据覆盖到一个已放行的汇总文件名下，Git 不会替你识别。推送前请检查 `git status --short`、`git diff --cached --stat` 及暂存内容，确认没有误加；不要使用 `git add -f` 绕过这些规则。本次只整理本地文件，没有自动提交或推送。

本 README 合并了原公开仓库的复现说明和研究工作目录的最新进展；完整说明以这里为准，论文及本地实验目录只保留检索入口，避免两份说明日后各自更新、彼此冲突。
