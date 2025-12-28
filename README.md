# GitHub仓库数据分析工具

## 项目简介

这是一个用于分析GitHub仓库数据的Python工具，支持批量分析多个仓库并生成详细的月度指标和总体统计报告。该工具能够从GitHub API获取数据，结合本地Git仓库分析，计算出多项开发协作指标。

## 主要功能

### 数据收集

- **PR数据**：收集拉取请求信息，包括创建时间、状态、合并状态、评论数、反应数等
- **Issue数据**：收集问题信息，包括解决周期、评论数、反应数等
- **Fork数据**：收集仓库的fork信息
- **RFC数据**：分析RFC相关issue和PR的采纳率
- **Git分析**：克隆仓库进行本地分析，获取提交数据、贡献者信息等

### 指标计算

- **PR合并率**：每月PR的合并比例
- **Issue解决周期**：问题从创建到解决的平均天数
- **RFC采纳率**：RFC相关issue和PR的采纳比例
- **开发者集中度**：提交量前10%的开发者占总提交的比例
- **多作者PR统计**：多人协作PR的次数和比例
- **贡献者统计**：每月活跃贡献者数量
- **交互数据**：评论和反应总数

## 输出文件

### 单个仓库输出

每个分析成功的仓库会在`output/`目录下创建独立的文件夹，包含两个文件：

- `{output_prefix}_{owner}_{name}_monthly.csv`：月度数据
- `{output_prefix}_{owner}_{name}_other.csv`：总体指标和仓库信息

### 汇总输出

- `github_analysis_results_summary.csv`：所有仓库的总体指标汇总
- `github_data.csv`：所有仓库月度数据的合并文件（包含中英文表头）

## 安装要求

### Python版本

- Python 3.7+

### 依赖包

bash

```
pip install requests python-dateutil gitpython
```

### GitHub令牌

需要GitHub个人访问令牌，需具有以下权限：

- `public_repo`（公开仓库）
- `repo`（私有仓库，如果需要）

## 使用方法

### 基本用法

bash

```
python main.py --file repos.csv --months 2023.11-2025.06 --token YOUR_GITHUB_TOKEN
```

### 参数说明

| 参数             | 说明                      | 必需 | 示例              |
| :--------------- | :------------------------ | :--- | :---------------- |
| `--file`         | 包含仓库列表的CSV文件路径 | 是   | `repos.csv`       |
| `--months`       | 分析的时间范围            | 是   | `2023.11-2025.06` |
| `--token`        | GitHub个人访问令牌        | 是   | `ghp_xxxxxxxx`    |
| `--output`       | 输出文件前缀              | 否   | `my_analysis`     |
| `--skip-collect` | 跳过API数据收集           | 否   | -                 |
| `--skip-git`     | 跳过Git克隆分析           | 否   | -                 |
| `--debug`        | 启用调试模式              | 否   | -                 |

### CSV文件格式

输入CSV文件需要包含以下列：

```
repo,eco_id,project_id_num,project_type
deepseek-ai/DeepSeek-LLM,1,1,1
meta-llama/llama-models,2,26,1
```

## 注意事项

### API限制

- GitHub API有速率限制（5000次/小时）
- 工具会自动检查剩余请求次数并适当延迟
- 建议在非高峰期运行批量分析

### 数据处理

1. **重复仓库处理**：自动识别并跳过重复仓库的API和Git分析
2. **数据抽样**：对于大量PR数据，采用智能抽样策略提高效率
3. **进度显示**：提供实时进度条和预计剩余时间
4. **错误处理**：完善的异常处理和重试机制

### 文件结构

```
项目目录/
├── main.py              # 主程序
├── repos.csv           # 输入文件（示例）
├── output/             # 输出目录
│   ├── owner1_repo1/  # 单个仓库输出
│   │   ├── my_analysis_owner1_repo1_monthly.csv
│   │   └── my_analysis_owner1_repo1_other.csv
│   ├── owner2_repo2/
│   ├── github_analysis_results_summary.csv
│   └── github_data.csv
└── README.md
```

## 输出格式说明

### 月度数据文件 (`*_monthly.csv`)

包含以下字段的中文表头：

- 项目名称、模型生态、项目数值标识、项目类型
- 创建时间、最后更新时间、月度时间
- 月度活跃贡献者数、PR合并率、Issue解决周期（倒数）
- 评论数、反应数、每月评论和反应总数
- RFC采纳率、提交量前10%开发者占比、月度非merge代码提交次数
- 月度发布新版本次数、月度Fork数、多作者PR次数、多作者PR比例

### 其他数据文件 (`*_other.csv`)

包含：

- 仓库基本信息
- 总体指标汇总
- 抽样统计信息

### 合并数据文件 (`github_data.csv`)

包含所有仓库月度数据的合并，采用两行表头：

- 第一行：英文字段名
- 第二行：中文说明
- 从第三行开始为数据

## 故障排除

### 常见问题

1. **API速率限制**：程序会暂停等待重置时间
2. **网络连接问题**：自动重试机制
3. **Git克隆失败**：跳过Git分析继续执行
4. **仓库不存在**：记录错误并继续处理下一个仓库

### 调试模式

使用`--debug`参数启用详细日志输出，有助于排查问题。

## 性能优化

1. **并发处理**：使用线程池并发获取PR详细数据
2. **智能抽样**：根据数据量动态调整抽样策略
3. **浅克隆**：Git仓库分析使用浅克隆提高速度
4. **缓存机制**：跳过重复仓库的分析
5. **分批处理**：避免内存溢出

## 许可证

本项目仅供学习和研究使用。使用GitHub API请遵守[GitHub服务条款](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)。

## 贡献

欢迎提交问题和改进建议。请确保在修改代码前充分测试。

## 更新日志

### v1.0

- 初始版本发布
- 支持批量仓库分析
- 提供月度指标和总体统计
- 智能数据抽样和错误处理
