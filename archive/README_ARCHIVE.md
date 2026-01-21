# 项目归档说明

本目录包含了项目中的一些旧文件和参考材料，已归档以保持项目结构清爽。

## 📁 归档内容

### 📄 文档文件
- `AGENTS.md` - OpenSpec代理使用说明（开发流程文档）
- `START_HERE.md` - 旧版快速开始指南
- `PROJECT_COMPLETION.md` - 项目生成完成清单
- `README_OLD.md` - 旧版README文件

### 📁 目录
- `openspec/` - OpenSpec项目规范和相关文档

## 🗂️ 为什么要归档？

1. **简化项目结构** - 删除不必要的文档，让用户快速找到核心功能
2. **避免混淆** - 新用户不会被旧的配置和说明误导
3. **保持参考** - 依然保存在archive中，需要时可以查阅

## 📚 当前项目结构

```
qwen3_nl2sql_grpo/
├── 📁 核心功能模块
│   ├── data/                    # 数据处理
│   ├── classifiers/             # 分类器
│   ├── generator/               # SQL生成器
│   ├── reward/                  # 奖励模型
│   ├── training/                # 训练脚本
│   ├── evaluation/              # 评估工具
│   └── deepseek_sql/            # 🆕 DeepSeek增强模块
├── 📁 配置和工具
│   ├── config.yaml              # 主配置
│   ├── requirements.txt         # 依赖
│   ├── scripts/                 # 辅助脚本
│   ├── QUICK_START.md           # 快速开始
│   └── INSTALL.md               # 安装说明
├── 📁 数据和输出
│   ├── DeepSeek-Math-V2/       # 参考实现（保留）
│   └── outputs/                 # 训练结果
└── 📁 归档内容
    └── archive/                 # 本目录
```

## 🔍 如何访问归档内容

如果需要查看归档的文件：

```bash
# 查看归档目录
ls -la archive/

# 查看特定文件
cat archive/README_OLD.md

# 恢复需要的内容（如果需要）
cp archive/needed_file.md ./
```

## 📝 注意事项

- 归档的内容**不会**影响项目的正常功能
- 所有核心功能都在主目录中，文档已更新
- 如需开发相关参考，可以查阅openspec目录中的规范

---

**项目现在更加清爽和专业！** 🎉