# STAT-X 项目说明

这个项目用于预测不同地区、不同时间段的急诊 overdose rate：`rate_per_10000_ed_visits`。目前代码已经完成了数据合并、建模数据生成、按 `period_id` 拆分训练集和验证集，以及基础模型比较框架。

## 目录结构

```text
STAT-X/
├── train/                         # 原始训练数据
├── val/                           # 原始验证/提交数据
├── outputs/                       # 中间数据和建模数据
├── notebooks/                     # 数据处理和建模代码
├── Data_Description.md            # 官方数据说明
└── sample_submission.csv          # 官方提交格式
```

## 当前进度

已经完成：

1. 读取官方 `train` 和 `val` 数据。
2. 将 target 数据和 covariates 按 `period_id + jurisdiction` 合并。
3. 生成 universal 数据集，也就是三个预测类别放在同一张表里。
4. 生成 category-specific 数据集，也就是 `all_drugs`、`all_opioids`、`all_stimulants` 三类分别建表。
5. 对天气缺失值设计了三种处理方案 A/B/C。
6. 将建模数据按 `period_id` 拆成训练集和验证集。
7. 搭建了 baseline、Ridge、ElasticNet、HistGradientBoosting、RandomForest 的比较代码。

还可以继续做：

1. 运行 `04_compare_universal_period_models.ipynb` 和 `05_compare_category_period_models.ipynb`，更新模型比较结果。
2. 根据验证集表现选择最终方案。
3. 用最终模型预测官方 `val/` 数据，并生成 submission 文件。

## 为什么按 period 拆分

项目里同一个 `period_id` 会包含多个州和多个 overdose 类别。如果随机拆分行，同一个 period 的信息可能同时出现在训练集和验证集里，验证分数会偏乐观。

现在的处理方式是：

- 一个 `period_id` 只能出现在训练集或验证集其中之一。
- 当前切分比例是 80% 训练、20% 验证。
- 当前共有 77 个 period。
- 训练集包含 61 个 period。
- 验证集包含 16 个 period。
- 训练集和验证集的 period 没有交集。

这样更接近“用已有时间段预测未见时间段”的场景。

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

作用：在 universal 数据上比较不同天气处理方案和不同模型。

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
- `fit_and_score_dataset()`：训练模型并计算 RMSE、MAE、R2。

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

## 如何继续推进

推荐下一步：

1. 运行 `04_compare_universal_period_models.ipynb`。
2. 运行 `05_compare_category_period_models.ipynb`。
3. 查看 `outputs/category_period_model_comparison.csv` 和 notebook 里的结果表。
4. 比较 universal 建模和 category-specific 建模哪个验证表现更好。
5. 选择最终方案后，用同样的数据处理逻辑处理官方 `val/` 数据。
6. 生成最终 submission。

## 注意事项

- 不要用随机行切分替代 period 切分，否则验证结果可能偏乐观。
- 验证集的天气填补不要使用验证集自己的统计量。
- `outputs` 里的 merged 表建议保留，因为它们是可复现后续数据集的基础。
- 如果重新运行 `02` 或 `03`，会覆盖对应的 train/val 建模表，这是正常的。
