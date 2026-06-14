# STAT-X 项目说明

这个项目用于预测不同地区、不同时间段的急诊 overdose rate：`rate_per_10000_ed_visits`。代码已完成数据合并、建模数据生成、5 折 GroupKFold 模型比较、MAT 密度图像特征,并已定下最终方案(**Universal + Dataset C + 图像特征 + HistGradientBoosting**,详见「最终方案与选型理由」)。仅剩生成 `submission.csv` 一步。

## 目录结构

```text
STAT-X/
├── train/                         # 原始训练数据(含 images/mat_density 热力图)
├── val/                           # 原始验证/提交数据(target 隐藏)
├── outputs/                       # 中间数据、建模数据、模型比较结果
├── notebooks/                     # 01-08 数据处理与建模代码 + statx_helpers.py
├── Data_Description.md            # 官方数据说明
└── sample_submission.csv          # 官方提交格式
```

## 当前进度

已经完成：

1. 读取官方 `train` 和 `val` 数据，按 `period_id + jurisdiction` 合并 target 和 covariates。
2. 生成 universal 数据集（三类放同一张表）和 category-specific 数据集（三类分别建表）。
3. 对天气缺失值设计了三种处理方案 A/B/C。
4. 把模型评估从一开始的单次 80/20 拆分**统一升级为 5 折 GroupKFold（按 period 分组）**。
5. 从 MAT 密度热力图提取 8 个可解释图像特征，并验证其增益。
6. 完成全部选型对比，**定下最终方案**（见下文「最终方案与选型理由」）。
7. 用最终模型预测官方 `val/`，生成 `submission.csv` 并完成首次提交。

## 提交结果（Kaggle Public LB）

| 日期 | 方案 | Public LB (MAE) | 本地 5 折 MAE | 排名 |
|---|---|---|---|---|
| 2026-06-13 | Universal + Dataset C + 图像 + HistGB | **1.328** | 1.725 | 67 / 75 |

说明：

- 榜分(1.328)和本地 5 折(1.725)**同量级**,证实排行榜指标就是**原始 rate 的 MAE**(排除 log / 归一化的可能)。
- 榜分比本地 CV **更低(更好)**,说明 val/ 那 6 个 period 比训练分布略容易;本地 5 折是更保守的估计。
- 排名 67/75 偏后,主要差距仍在最难的 `all_drugs`——后续上分重点是用更强 NLP 挖 `state_doh_release` 文本(关键词计数已验证无效),以及模型集成 / 调参。

## 相对原始 fork 的改动总览

原始 fork 是竞赛 starter（`train/`、`val/`、`sample_submission.csv`、`Data_Description.md` 和最初的数据合并代码）。在其之上,团队和本分支做了如下改动：

### 团队前期（数据与建模框架）
- `01_prepare_merged_data.ipynb`：合并出 universal / category 基础表。
- `02` / `03`：生成 A/B/C × universal / category 的建模表。
- `statx_helpers.py`：集中公共逻辑（period 拆分、A/B/C 数据集、预处理、模型、评分）。
- `04` / `05`：搭建 baseline + Ridge + ElasticNet + HistGB + RandomForest 的比较框架。

### 本分支 `image-features`（本次新增/修改）
- **新增图像特征管线**：
  - `06_image_features.py`：把每张热力图压成 8 个去背景、可解释的密度特征 → `outputs/image_features.csv`。
  - `07_validate_image_features.py`：验证特征干净 merge 且在 5 折下有增益。
  - `08_image_feature_gains.py`：逐模型 × 逐数据集量化图像增益 → `outputs/image_feature_gains.csv`。
- **新增文本特征管线(重构自队友的 `STAIX-26.py`)**:
  - `09_text_features.py`：清洗版关键词文本特征,修掉原脚本所有 bug(硬编码路径、val 误用 train、train/val 关键词不一致、`overdose` 重复、死代码),用相对路径、能落盘 → `outputs/text_features.csv`。
  - `10_text_feature_gains.py`：5 折量化文本增益 → `outputs/text_feature_gains.csv`。结论是几乎零增益,故最终未采用(见「文本特征」)。
- **评估方法升级为 5 折**：在 `statx_helpers.py` 新增 `cross_val_by_period`（折内重算天气填补、防泄漏），把 `04` / `05` / `07` / `08` / `10` 全部改为调用它,取代原来的单次 80/20。
- **新增结果产物**：`outputs/universal_period_model_comparison.csv`、`category_period_model_comparison.csv`、`image_feature_gains.csv`、`text_features.csv`、`text_feature_gains.csv`。
- **README**：补充 5 折说明、图像/文本增益表、最终选型理由。

## 为什么按 period 拆分

项目里同一个 `period_id` 会包含多个州和多个 overdose 类别。如果随机拆分行，同一个 period 的信息可能同时出现在训练集和验证集里，验证分数会偏乐观。

现在的处理方式是：

- 一个 `period_id` 只能出现在训练集或验证集其中之一。
- 当前共有 77 个 period。

模型评估统一使用 **5 折 GroupKFold（按 `period_id` 分组）**，由 `statx_helpers.cross_val_by_period` 实现：

- 每一折里，同一个 `period_id` 只会落在训练或验证其中一侧，绝不跨折。
- 每个 period 都恰好被验证一次，最后取 5 折平均。
- 因为只有 77 个 period，单次 80/20 留出的验证集只有 16 个 period，估计方差大；5 折用上全部 period，结果更稳，且 04/05/07 同口径可比。
- Dataset B/C 的天气填补在**每一折内部**用该折训练数据重新计算（`reference=` 折内训练集），不会跨折泄漏。

这样更接近“用已有时间段预测未见时间段”的场景。

> 注：`split_by_period` 和 `outputs/*_train.csv` / `*_val.csv`（单次固定 80/20 拆分）仍然保留，可作为最终验收的固定 holdout，但模型比较已改用 5 折。

## Notebook 说明

建议按照下面顺序运行。

### 1. `notebooks/01_prepare_merged_data.ipynb`

作用：读取原始数据，并生成合并后的基础表。

主要输出：

- `outputs/train_universal_merged.csv`
- `outputs/train_all_drugs_merged.csv`
- `outputs/train_all_opioids_merged.csv`
- `outputs/train_all_stimulants_merged.csv`
- `outputs/val_universal_merged.csv`
- `outputs/val_all_drugs_merged.csv`
- `outputs/val_all_opioids_merged.csv`
- `outputs/val_all_stimulants_merged.csv`

这些表是后续处理的基础数据，建议保留。

### 2. `notebooks/02_make_universal_period_datasets.ipynb`

作用：基于 `train_universal_merged.csv` 生成 universal 建模数据。

Universal 表把三个 overdose 类别放在同一张表中，模型可以使用 `overdose_category` 作为一个特征。

当前数据规模：

- 原始 universal 数据：11781 行，77 个 period
- 训练集：9333 行，61 个 period
- 验证集：2448 行，16 个 period

### 3. `notebooks/03_make_category_period_datasets.ipynb`

作用：分别为三个预测类别生成建模数据。

三个类别分别是：

- `all_drugs`
- `all_opioids`
- `all_stimulants`

每个类别原始数据都是 3927 行、77 个 period。按 period 拆分后：

- 训练集：3111 行，61 个 period
- 验证集：816 行，16 个 period

### 4. `notebooks/04_compare_universal_period_models.ipynb`

作用：在 universal 数据上用 5 折 GroupKFold 比较不同天气处理方案和不同模型。

比较内容：

- Dataset A/B/C
- Mean baseline
- Ridge
- ElasticNet
- HistGradientBoosting
- RandomForest

评估指标：

- RMSE
- MAE
- R2

### 5. `notebooks/05_compare_category_period_models.ipynb`

作用：分别在 `all_drugs`、`all_opioids`、`all_stimulants` 三个类别上比较模型。

结果会保存到：

- `outputs/category_period_model_comparison.csv`

如果还没有这个文件，说明模型比较 notebook 还没有重新运行。

### 6. `notebooks/06_image_features.py`

作用：从每张 MAT 密度热力图(`{州}_{period_id}.png`)提取 8 个可解释的密度统计特征,以 `(period_id, jurisdiction)` 为键,可直接 merge 回建模表。

要点:

- 用 `max(R,G,B) < 20` 掩掉州外黑色背景,州大小不污染特征。
- viridis 配色亮度单调对应密度,用亮度当密度代理。
- 8 个特征:背景占比、密度均值/标准差/最大值/p90、高密度像素占比、Top10% 份额、空间分散度。
- 同一 (州, period) 三类共用一张图,去重后每张只算一次。
- 输出:`outputs/image_features.csv`(含 `split` 列区分竞赛 train/ 与 val/ 文件夹)。
- 之所以用手工特征而非 CNN:单特征信号弱(|Spearman ρ|≤0.22)且部分与州绑定,CNN 易过拟合。

### 7. `notebooks/07_validate_image_features.py`

作用：用与 04/05 同一套 5 折引擎,在 Dataset A/B/C 上比较 HistGB「加不加图像特征」,确认 merge 干净且有增益。

### 8. `notebooks/08_image_feature_gains.py`

作用：逐模型(Model 0–4)× 逐数据集(A/B/C)量化图像特征的改善幅度,输出 `outputs/image_feature_gains.csv`,并打印可直接贴进 README 的表格。

### 9. `notebooks/09_text_features.py`

作用：从 `state_doh_release` 文本提取关键词计数特征(危机/预警/应对三组计数 + 文本长度 + 有无发文 + 风险标签),以 `(period_id, jurisdiction)` 为键,输出 `outputs/text_features.csv`。是队友 `STAIX-26.py` 的清洗重构版(修了硬编码路径、val 误用 train、train/val 关键词不一致等 bug)。

### 10. `notebooks/10_text_feature_gains.py`

作用：在 universal + Dataset C 上用 5 折比较 cov / +text / +image / +image+text 四种特征组合,量化文本特征增益,输出 `outputs/text_feature_gains.csv`。结论:文本特征几乎零增益,最终未采用。

> 注:旧脚本 `notebooks/STAIX-26.py` 已被 `09` 取代,保留仅作历史参考,不在管线中。

### 11. `notebooks/11_make_submission.py`

作用：用最终方案(Universal + Dataset C + 图像特征 + HistGradientBoosting)在**全部**训练数据上训练,预测官方 `val/`,按 `sample_submission.csv` 的 `row_id` 对齐,生成根目录的 `submission.csv`(918 行,`row_id` + `rate_per_10000_ed_visits`)。文本特征不纳入(09/10 验证无增益)。

## 共享代码文件

### `notebooks/statx_helpers.py`

这个文件把重复代码集中到一起，避免每个 notebook 里复制同样的函数。

主要功能：

- `handle_text()`：处理 `state_doh_release` 文本缺失，并生成 `has_doh_release` 指示变量。
- `split_by_period()`：按 `period_id` 拆分训练集和验证集。
- `make_dataset()`：生成 A/B/C 三类建模数据。
- `save_all_datasets()`：保存 train/val 建模数据。
- `build_preprocessor()`：为模型创建数值和类别变量预处理流程。
- `get_models()`：定义要比较的模型。
- `fit_and_score_dataset()`：在单次 train/val 拆分上训练模型并计算 RMSE、MAE、R2。
- `cross_val_by_period()`：5 折 GroupKFold（按 period 分组）评估,折内重算天气填补防泄漏,返回每个模型的 5 折平均。04/05/07 统一调用它。

## A/B/C 数据集含义

项目里保留三种天气变量处理方式，用来比较哪一种更适合建模。

### Dataset A: no weather

文件名中包含：`A_no_weather`

处理方式：

- 删除 `temp_avg_f`
- 删除 `precip_in`

适用想法：不对天气缺失做假设，直接不用天气变量。

### Dataset B: weather period median

文件名中包含：`B_weather_period_median`

处理方式：

- 保留天气变量。
- 如果天气缺失，先用同一个 `period_id` 内的中位数填补。
- 如果同 period 仍然无法填补，再用训练集全局中位数填补。
- 添加 `weather_missing` 变量，标记天气是否原本缺失。

重要：验证集填补时只使用训练集统计量，避免验证集信息泄漏。

### Dataset C: weather similar state

文件名中包含：`C_weather_similar_state`

处理方式：

- 保留天气变量。
- 对 AK、HI、DC 等天气缺失较特殊的地区，优先用相似州同 period 的天气中位数填补。
- 如果相似州不可用，再退回到训练集 period 中位数。
- 最后再退回到训练集全局中位数。
- 添加 `weather_missing` 变量。

当前相似州设置：

```python
AK: WA, MT, ND, MN
HI: CA, FL
DC: MD, VA
```

## outputs 文件说明

### 基础 merged 表

这些表由 `01_prepare_merged_data.ipynb` 生成，是后续所有数据集的来源。

```text
train_universal_merged.csv
train_all_drugs_merged.csv
train_all_opioids_merged.csv
train_all_stimulants_merged.csv
val_universal_merged.csv
val_all_drugs_merged.csv
val_all_opioids_merged.csv
val_all_stimulants_merged.csv
```

### Universal 建模表

```text
universal_A_no_weather_train.csv
universal_A_no_weather_val.csv
universal_B_weather_period_median_train.csv
universal_B_weather_period_median_val.csv
universal_C_weather_similar_state_train.csv
universal_C_weather_similar_state_val.csv
```

### Category-specific 建模表

每个类别都有 A/B/C 三种版本，每种版本都有 train 和 val。

```text
all_drugs_A_no_weather_train.csv
all_drugs_A_no_weather_val.csv
all_drugs_B_weather_period_median_train.csv
all_drugs_B_weather_period_median_val.csv
all_drugs_C_weather_similar_state_train.csv
all_drugs_C_weather_similar_state_val.csv

all_opioids_A_no_weather_train.csv
all_opioids_A_no_weather_val.csv
all_opioids_B_weather_period_median_train.csv
all_opioids_B_weather_period_median_val.csv
all_opioids_C_weather_similar_state_train.csv
all_opioids_C_weather_similar_state_val.csv

all_stimulants_A_no_weather_train.csv
all_stimulants_A_no_weather_val.csv
all_stimulants_B_weather_period_median_train.csv
all_stimulants_B_weather_period_median_val.csv
all_stimulants_C_weather_similar_state_train.csv
all_stimulants_C_weather_similar_state_val.csv
```

## 当前建模策略

目前比较的模型是基础模型，目的是先建立可靠的验证流程，而不是一开始追求复杂模型。

模型包括：

- Model 0: Mean baseline
- Model 1: Ridge fixed effects
- Model 2: ElasticNet fixed effects
- Model 3: HistGradientBoosting
- Model 4: RandomForest

预处理方式：

- 数值变量使用 `StandardScaler`
- 类别变量使用 `OneHotEncoder`
- 原始长文本 `state_doh_release` 暂时不直接进入模型
- 由文本生成的 `has_doh_release` 会进入模型

## 5 折结果(universal 数据)

下面是 `04` 在 universal 数据上用 5 折 GroupKFold 跑出来的结果(`outputs/universal_period_model_comparison.csv`），按 RMSE 排序的前几名:

| Dataset | Model | RMSE | MAE | R2 |
|---|---|---|---|---|
| Dataset B | Model 3 HistGB | 2.670 | 1.732 | 0.926 |
| Dataset C | Model 3 HistGB | 2.672 | 1.732 | 0.925 |
| Dataset C | Model 4 RandomForest | 2.786 | 1.815 | 0.919 |
| Dataset A | Model 3 HistGB | 2.919 | 1.944 | 0.911 |
| Dataset B/C | Model 2 ElasticNet | 4.001 | 2.704 | 0.833 |
| Dataset \* | Model 0 baseline | 6.426 | 4.115 | 0.569 |

结论:树模型(Model 3 / 4)明显领先线性模型;**Dataset B 和 C 几乎打平且都优于 A**——说明保留天气并合理填补是有价值的。考虑到 C 对 AK/HI/DC 等天气特殊地区用相似州填补、假设更合理,**我们选 Dataset C 作为最终方案**。

## 图像特征增益(notebook 08)

把 `06` 生成的 8 个 MAT 密度图像特征(`outputs/image_features.csv`)merge 进 universal 数据后,对每个模型、每个数据集在同一套 5 折 GroupKFold 上重新评估,"有图像 vs 无图像"的改善幅度如下(完整数据见 `outputs/image_feature_gains.csv`,正号=变好):

| Dataset | Model | MAE(无图像) | MAE(+图像) | MAE 改善 | RMSE 改善 |
|---|---|---|---|---|---|
| Dataset A | Model 0 baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| Dataset A | Model 1 Ridge | 3.099 | 3.091 | +0.2% | +0.2% |
| Dataset A | Model 2 ElasticNet | 2.901 | 2.898 | +0.1% | +0.1% |
| Dataset A | Model 3 HistGB | 1.944 | 1.913 | +1.6% | +1.5% |
| Dataset A | Model 4 RandomForest | 1.924 | 1.834 | **+4.7%** | +4.3% |
| Dataset B | Model 0 baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| Dataset B | Model 1 Ridge | 3.102 | 3.096 | +0.2% | +0.1% |
| Dataset B | Model 2 ElasticNet | 2.704 | 2.706 | −0.1% | +0.0% |
| Dataset B | Model 3 HistGB | 1.732 | 1.723 | +0.5% | +0.4% |
| Dataset B | Model 4 RandomForest | 1.817 | 1.760 | **+3.1%** | +2.9% |
| **Dataset C** | Model 0 baseline | 4.115 | 4.115 | +0.0% | +0.0% |
| **Dataset C** | Model 1 Ridge | 3.096 | 3.090 | +0.2% | +0.1% |
| **Dataset C** | Model 2 ElasticNet | 2.705 | 2.704 | +0.0% | +0.1% |
| **Dataset C** | Model 3 HistGB | 1.732 | 1.725 | +0.4% | +0.2% |
| **Dataset C** | Model 4 RandomForest | 1.815 | 1.758 | **+3.2%** | +2.9% |

观察:

- **树模型吃得下图像特征,线性模型几乎吃不下**。Model 4 RandomForest 改善最大(三个数据集都有 +3%~+4.7% MAE),Model 3 HistGB 次之;Ridge/ElasticNet 改善都在 ±0.2% 噪声范围内,baseline 当然完全不变(均值预测用不上特征)。这符合图像特征与目标的关系是弱、非线性、部分与州绑定的判断——只有能自动建非线性交互的树模型才用得上。
- **在最终选定的 Dataset C 上**,图像特征让 RandomForest 的 MAE 从 1.815 降到 1.758(+3.2%),HistGB 从 1.732 降到 1.725(+0.4%)。注意:RF **涨幅**最大,但绝对分数仍是 **HistGB(1.725)< RF(1.758)**——最终模型选 HistGB,详见「最终方案与选型理由」。
- 增益虽不大但**方向稳定**(跨三个数据集一致为正),正是 `06` 当初选可解释特征而非 CNN 的预期:小而稳,不靠过拟合。

## 文本特征(notebook 09 / 10)—— 试过,没用上

`09_text_features.py` 把队友早期那份 `STAIX-26.py` 重构成干净版(修了所有 bug、用相对路径、train/val 共用一套关键词、能落盘),从 `state_doh_release` 提取关键词计数特征(危机 / 预警 / 应对三组计数 + 文本长度 + 有无发文)→ `outputs/text_features.csv`。

`10_text_feature_gains.py` 在最终设定(universal + Dataset C)上,对每个模型用同一套 5 折比较四种特征组合,结果(`outputs/text_feature_gains.csv`,MAE):

| Model | cov | +text | +image | +image+text | text 单独 vs cov |
|---|---|---|---|---|---|
| Model 0 baseline | 4.115 | 4.115 | 4.115 | 4.115 | +0.0% |
| Model 1 Ridge | 3.096 | 3.096 | 3.090 | 3.090 | +0.0% |
| Model 2 ElasticNet | 2.705 | 2.705 | 2.704 | 2.705 | −0.0% |
| Model 3 HistGB | 1.732 | 1.734 | 1.725 | 1.722 | −0.2% |
| Model 4 RandomForest | 1.815 | 1.819 | 1.758 | 1.759 | −0.2% |

> 上表的 `cov` 和 `+image` 两列,就是「未融入文本之前」的结果,可直接对照。

**结论:关键词计数文本特征几乎零贡献,树模型上甚至略微变差(−0.2%),所以最终模型不纳入文本特征。** "+image+text" 相对 "cov" 的改善几乎全部来自图像,不是文本。

原因分析:

- "有没有发文"这个信号 `has_doh_release` 早已在协变量里,细分成危机/预警/应对的计数没带来额外可分性。
- 关键词桶把文本压成几个计数,信息损失太大。文本里大概率有信号(尤其对最难的 `all_drugs`),但要用更强的 NLP(TF-IDF / 词嵌入 / 主题模型)才能榨出来——这是后续真正的上分方向,而不是关键词计数。

## 最终方案与选型理由

最终提交方案:**Universal（一个模型）+ Dataset C + 图像特征 + HistGradientBoosting**,用 5 折 GroupKFold 选型。

下面每一个选择都有数据支撑(均为 5 折结果)。

### 为什么用 5 折,而不是一开始的单次 80/20

- 全数据只有 **77 个 period**。单次 80/20 的验证集只有 16 个 period,估计方差大——例如单次拆分里 Dataset B(MAE 1.766)和 C(1.767)只差 0.001,排名完全在噪声里。
- 5 折让**每个 period 都被验证一次**,方差小、结论稳,而且 04/05/07/08 同口径可比。
- 关键防泄漏:B/C 的天气填补在**每折内部**重算,不跨折。

### 为什么选 Dataset C(天气处理)

| 数据集 | 最佳 MAE(HistGB) | 说明 |
|---|---|---|
| A 不用天气 | 1.944 | 信息最少,明显最差 |
| B period 中位数填补 | 1.732 | 与 C 几乎并列 |
| **C 相似州填补** | **1.732** | 与 B 并列,但对 AK/HI/DC 假设更合理 |

B 和 C 实质打平且都远好于 A → 保留天气有价值。C 对天气特殊地区(AK/HI/DC)用相似州填补,假设比"全局中位数"更可信,故选 C。

### 为什么选 HistGradientBoosting,而不是 RandomForest

虽然图像特征让 **RandomForest 涨幅最大**(+3.2% vs HistGB +0.4%),但"涨得多"不等于"最终最好"。看绝对分数(Dataset C,已加图像):

| 模型 | MAE | RMSE |
|---|---|---|
| **HistGradientBoosting** | **1.725** ✅ | **2.667** ✅ |
| RandomForest | 1.758 | 2.704 |

RF 起点差(无图像 1.815),补完也只到 1.758,仍输给 HistGB 的 1.725。HistGB 在 MAE 和 RMSE 上都是 Dataset C 的绝对最优。

### 为什么用一个模型预测三类,而不分别建模

竞赛按 `all_drugs` / `all_opioids` / `all_stimulants` 三类打分,但头对头比较(Dataset C + 图像 + HistGB)显示分开建模**没有收益**:

| 方案 | 合并 MAE | 合并 RMSE |
|---|---|---|
| **Universal(1 个模型)** | **1.7247** | **2.667** |
| Category-specific(3 个模型) | 1.735 | ~2.698 |

两者实质打平,universal 还略胜。原因:分开建模把每个模型的训练数据砍到 1/3(~3927 行),在只有 77 个 period 的小数据下更易过拟合。打平时选更简单、数据更充分的 universal。

### 为什么不纳入文本特征

试过(notebook 09/10),但**关键词计数文本特征在 5 折下几乎零增益,树模型上还略微变差**(见上文「文本特征」)。`has_doh_release` 已捕获"有无发文",细分计数没带来额外信息。故最终模型**不含文本特征**。文本仍是最大的潜在增量,但需要 TF-IDF / 词嵌入等更强 NLP,留作后续。

> 各类误差不均:`all_drugs`(率高)MAE 3.09 最难,`all_opioids` 1.39,`all_stimulants` 0.73 最好预测。后续想再上分,重点在啃 `all_drugs`——也是 `state_doh_release` 文本(用更强 NLP 时)最可能发力的地方。

## 如何继续推进

选型已定(见「最终方案与选型理由」),只剩生成提交:

1. 在**全部**训练数据上重训最终模型 **Universal + Dataset C + 图像特征 + HistGradientBoosting**(不留折,折只用于选型)。
2. 用同样的 Dataset C 填补逻辑处理官方 `val/`,并 merge `06` 已为 val 算好的图像特征(`image_features.csv` 中 `split=="val"` 的行)。
3. 预测并写出 `submission.csv`(仅 `row_id` + `rate_per_10000_ed_visits`,918 行)。
4. 提交,拿到真实榜分,再对照排行榜指标判断后续是否值得继续(例如启用 `state_doh_release` 文本特征)。

如需复现选型结果,运行 `04`、`05`、`08` 刷新 `outputs/*_comparison.csv` 和 `image_feature_gains.csv`。

## 注意事项

- 不要用随机行切分替代 period 切分，否则验证结果可能偏乐观。
- 验证集的天气填补不要使用验证集自己的统计量。
- `outputs` 里的 merged 表建议保留，因为它们是可复现后续数据集的基础。
- 如果重新运行 `02` 或 `03`，会覆盖对应的 train/val 建模表，这是正常的。
