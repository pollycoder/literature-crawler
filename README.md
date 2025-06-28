# AI文献自动爬取助手

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ArXiv](https://img.shields.io/badge/ArXiv-Supported-red.svg)](https://arxiv.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/your-username/literature-crawler/graphs/commit-activity)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

一个基于AI的智能文献调研工具，能够自动理解用户需求，在ArXiv上爬取相关论文，进行智能分类分析，并生成高质量的文献综述。支持多种大语言模型，提供完整的从需求分析到综述生成的自动化流程。

## 演示视频

[![演示视频](https://img.shields.io/badge/Demo-YouTube-red)](https://youtube.com/demo-link)

## 目录

- [主要功能](#主要功能)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
- [项目架构](#项目架构)
- [配置说明](#配置说明)
- [维护指南](#维护指南)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 主要功能

### 智能需求理解
- 基于大语言模型分析用户输入的研究需求
- 自动提取研究领域、关键词和搜索策略
- 支持中英文需求描述

### ArXiv论文爬取
- 高效并发爬取ArXiv论文数据
- 智能查询优化和去重处理
- 支持时间范围和分类过滤

### 智能论文分类
- 多维度论文分析（研究类型、技术方法、应用领域）
- 相关性评分和新颖度评估
- 自动提取关键词和技术方法

### 文献综述生成
- 基于分析结果自动生成结构化综述
- 包含研究现状、技术对比、发展趋势等
- 支持Markdown、Word等多种输出格式

### 多模型路由
- 支持OpenAI GPT、Claude、Gemini、DeepSeek等多个LLM
- 智能模型选择（成本、速度、质量优化）
- 统一使用第三方代理提高稳定性和速度
- 按任务类型配置最优模型偏好
- 实时统计成本和性能指标

### 高级搜索功能
- 智能查询优化和扩展
- 支持时间范围过滤
- 自动去重和相关性排序
- 并发爬取提高效率

### 详细分析报告
- 论文领域分布统计
- 高质量论文自动筛选
- 技术方法趋势分析
- 可视化结果展示

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/literature-crawler.git
cd literature-crawler

# 安装依赖
pip install -r requirements.txt

# 或使用poetry
poetry install
```

### 配置

#### 方法一：快速配置第三方代理（推荐）

```bash
# 使用第三方代理服务（通常更稳定快速）
python main.py setup-proxy --api-key your_proxy_api_key --base-url https://your-proxy-domain.com/v1
```

#### 方法二：手动配置

1. 复制环境变量配置文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，添加你的API密钥：
```bash
# 第三方代理配置（推荐）
OPENAI_PROXY_API_KEY=your_proxy_api_key_here
OPENAI_PROXY_BASE_URL=https://your-proxy-domain.com/v1

# 或使用官方API
OPENAI_API_KEY=your_openai_api_key_here
```

3. 修改配置文件 `config/config.yaml`（可选）

### 基本使用

```bash
# 搜索和分析论文
python main.py search "深度学习在图像识别中的应用" --max-papers 30

# 生成文献综述
python main.py review --input ./papers.json --output ./review.md

# 测试模型连接
python main.py test-models

# 查看所有可用模型
python main.py list-models

# 查看使用统计
python main.py stats

# 快速配置代理（如果需要）
python main.py setup-proxy --api-key your_key --base-url https://your-proxy.com/v1
```

## 详细使用指南

### 搜索论文

```bash
# 基本搜索
python main.py search "机器学习优化算法"

# 指定论文数量和模型（使用新的模型格式）
python main.py search "神经网络" --max-papers 50 --model openai_proxy/gpt-4

# 自定义输出路径
python main.py search "计算机视觉" --output ./cv_papers.json

# 使用不同的代理模型
python main.py search "强化学习" --model claude_proxy/claude-3-5-sonnet-20241022
python main.py search "深度学习" --model deepseek_proxy/deepseek-chat
```

### 生成综述

```bash
# 基于搜索结果生成综述
python main.py review --input ./papers.json --output ./review.md

# 使用特定模型生成（新格式）
python main.py review --model openai_proxy/gpt-4 --input ./papers.json

# 使用高质量模型生成详细综述
python main.py review --model openai_proxy/o1-preview --input ./papers.json --output ./detailed_review.md
```

### 模型管理

```bash
# 测试所有配置的模型
python main.py test-models

# 查看所有可用模型和配置
python main.py list-models

# 查看模型使用统计和成本
python main.py stats
```

### 高级用法

```bash
# 批量处理多个研究主题
for topic in "深度学习" "强化学习" "计算机视觉"; do
  python main.py search "$topic" --output "${topic}_papers.json"
  python main.py review --input "${topic}_papers.json" --output "${topic}_review.md"
done

# 使用配置文件指定自定义设置
python main.py search "AI安全" --config ./custom_config.yaml
```

## 项目架构

```
literature-crawler/
├── src/
│   ├── core/                   # 核心数据模型
│   ├── llm/                    # 多模型路由系统
│   │   ├── providers/          # 各LLM提供商实现
│   │   ├── router.py           # 模型路由器
│   │   └── config.py           # 模型配置
│   ├── prompts/                # 提示词管理
│   ├── crawler/                # ArXiv爬虫
│   ├── analyzer/               # 文献分析
│   │   ├── requirement_analyzer.py  # 需求分析
│   │   └── classifier.py       # 论文分类
│   ├── reviewer/               # 综述生成
│   ├── integrations/           # 第三方集成
│   └── cli/                    # 命令行界面
├── config/                     # 配置文件
│   ├── config.yaml            # 主配置
│   └── prompts/               # 提示词模板
├── data/                      # 数据存储
├── tests/                     # 测试文件
└── requirements.txt           # 依赖列表
```

## 配置说明

### 模型配置

项目采用统一的第三方代理配置，在 `config/config.yaml` 中设置：

```yaml
# 提供商配置
providers:
  openai_proxy:
    provider_type: "openai"
    api_key: "${OPENAI_PROXY_API_KEY}"
    base_url: "${OPENAI_PROXY_BASE_URL}"
    timeout: 30
    retry_count: 3

# 模型配置
models:
  openai_proxy:
    - name: "gpt-4"
      display_name: "GPT-4 (代理)"
      cost_per_1k_input: 0.03
      cost_per_1k_output: 0.06
```

### 任务偏好

为不同任务设置模型偏好（按优先级排序）：

```yaml
task_preferences:
  requirement_analysis: 
    - "openai_proxy/gpt-4"
    - "claude_proxy/claude-3-5-sonnet-20241022"
    - "deepseek_proxy/deepseek-chat"
  
  paper_classification: 
    - "openai_proxy/gpt-3.5-turbo"
    - "deepseek_proxy/deepseek-chat"
    - "claude_proxy/claude-3-haiku-20240307"
  
  review_generation: 
    - "openai_proxy/gpt-4"
    - "claude_proxy/claude-3-5-sonnet-20241022"
    - "openai_proxy/o1-preview"
```

### 环境变量配置

在 `.env` 文件中设置API密钥：

```bash
# OpenAI 代理
OPENAI_PROXY_API_KEY=your_proxy_api_key
OPENAI_PROXY_BASE_URL=https://your-proxy-domain.com/v1

# Claude 代理
CLAUDE_PROXY_API_KEY=your_claude_proxy_key
CLAUDE_PROXY_BASE_URL=https://claude-proxy.com/v1

# Gemini 代理
GEMINI_PROXY_API_KEY=your_gemini_proxy_key
GEMINI_PROXY_BASE_URL=https://gemini-proxy.com/v1

# DeepSeek 代理
DEEPSEEK_PROXY_API_KEY=your_deepseek_proxy_key
DEEPSEEK_PROXY_BASE_URL=https://deepseek-proxy.com/v1
```

## 高级功能

### 自定义提示词

在 `config/prompts/` 目录下修改提示词模板：

```yaml
# config/prompts/requirement_analysis.yaml
name: "需求分析提示词"
task_type: "requirement_analysis"
template: |
  你是一个专业的学术文献调研助手...
  用户输入：{{ user_input }}
  请分析用户需求并返回JSON格式结果...
```

### 性能监控

```bash
# 查看详细的模型使用统计
python main.py stats

# 监控日志文件
tail -f logs/errors.log
tail -f logs/requests.log

# 检查缓存使用情况
du -sh data/cache/
```

### 批量处理脚本

```bash
#!/bin/bash
# batch_research.sh - 批量研究脚本

topics=("深度学习" "强化学习" "计算机视觉" "自然语言处理")

for topic in "${topics[@]}"; do
    echo "处理主题: $topic"
    python main.py search "$topic" --max-papers 30 --output "${topic}_papers.json"
    python main.py review --input "${topic}_papers.json" --output "${topic}_review.md"
    echo "完成: $topic"
done
```

## 示例输出

### 需求分析结果
```json
{
  "research_domains": ["深度学习", "图像识别", "计算机视觉"],
  "specific_topics": ["卷积神经网络", "注意力机制", "迁移学习"],
  "keywords": ["CNN", "attention", "transfer learning"],
  "search_queries": ["deep learning image recognition", "CNN computer vision"]
}
```

### 论文分类结果
```json
{
  "primary_domain": "计算机视觉",
  "research_type": "application",
  "relevance_score": 8.5,
  "novelty_level": "high",
  "technical_approaches": ["ResNet", "注意力机制"]
}
```

## 维护指南

### 日常维护任务

```bash
# 更新依赖
pip list --outdated
pip install --upgrade -r requirements.txt

# 清理日志文件 (保留最近30天)
find logs/ -name "*.log" -mtime +30 -delete

# 检查模型连接状态
python main.py test-models

# 查看使用统计和成本
python main.py stats
```

### 性能监控

定期检查以下指标：
- API调用成功率 (目标: >95%)
- 平均响应时间 (目标: <30秒)
- 每日成本 (通过 `stats` 命令查看)
- 错误日志频率 (`logs/errors.log`)

### 配置管理

```bash
# 备份重要配置
cp config/config.yaml config/config.yaml.backup
cp .env .env.backup

# 更新API密钥
python main.py setup-proxy --api-key new_key --base-url https://new-proxy.com/v1

# 测试新配置
python main.py test-models
```

## 故障排除

### 常见问题

**1. API调用失败**
```bash
# 检查网络连接和API密钥
python main.py test-models

# 查看详细错误信息
tail -n 50 logs/errors.log
```

**2. 爬虫超时**
```bash
# 调整配置中的超时时间
vim config/config.yaml
# 修改 crawler.timeout 和 crawler.request_delay
```

**3. 内存使用过高**
```bash
# 减少并发请求数
# 编辑 config/config.yaml 中的 max_concurrent_requests
# 减少单次处理的论文数量
python main.py search "主题" --max-papers 10
```

**4. 模型响应质量差**
```bash
# 切换到更高质量的模型
python main.py search "主题" --model openai_proxy/gpt-4
python main.py review --model openai_proxy/o1-preview --input papers.json
```

### 日志分析

```bash
# 分析错误频率
grep "ERROR" logs/errors.log | cut -d' ' -f1 | sort | uniq -c

# 查看API请求统计
grep "request" logs/requests.log | wc -l

# 监控实时日志
tail -f logs/errors.log logs/requests.log
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/your-username/literature-crawler.git
cd literature-crawler

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install black flake8 isort mypy pytest

# 设置pre-commit钩子
pre-commit install
```

### 代码贡献流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 编写代码并确保通过测试
   ```bash
   # 代码格式化
   black src/
   isort src/
   
   # 静态检查
   flake8 src/
   mypy src/
   
   # 运行测试
   pytest tests/
   ```
4. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
5. 推送到分支 (`git push origin feature/AmazingFeature`)
6. 打开 Pull Request

### 代码规范

- 使用 [Black](https://github.com/psf/black) 进行代码格式化
- 遵循 [PEP 8](https://pep8.org/) 编码规范
- 添加类型注解 (Type Hints)
- 编写单元测试
- 更新相关文档

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [ArXiv](https://arxiv.org/) - 提供开放的学术论文数据
- [OpenAI](https://openai.com/) - GPT模型支持
- [Anthropic](https://anthropic.com/) - Claude模型支持
- 所有开源依赖项的贡献者

## 联系方式

- 项目链接: [https://github.com/your-username/literature-crawler](https://github.com/your-username/literature-crawler)
- 问题反馈: [Issues](https://github.com/your-username/literature-crawler/issues)

## 未来计划

### 短期计划 (1-3个月)
- 添加完整的单元测试覆盖
- 实现结果缓存机制
- 优化内存使用和性能
- 增加更多论文数据库支持

### 中期计划 (3-6个月)
- 开发Web用户界面
- 添加数据可视化功能
- 支持协作式文献管理
- 集成Zotero和Mendeley

### 长期计划 (6个月+)
- 开发移动应用
- 支持多语言界面
- 添加AI论文质量评估
- 构建学术社区功能

## 技术栈

- **后端**: Python 3.8+, AsyncIO, Aiohttp
- **CLI**: Click, Rich (美观的命令行界面)
- **AI模型**: OpenAI GPT, Claude, Gemini, DeepSeek
- **数据处理**: Pandas, BeautifulSoup, Feedparser
- **配置管理**: YAML, Python-dotenv
- **日志**: 结构化日志系统
- **测试**: Pytest (计划中)

## 获取帮助

如果你遇到问题或需要帮助：

1. **查看文档**: 首先检查本 README 和 [故障排除](#-故障排除) 部分
2. **搜索Issues**: 在 [GitHub Issues](https://github.com/your-username/literature-crawler/issues) 中搜索类似问题
3. **提交Issue**: 如果没有找到答案，请创建新的 Issue
4. **社区讨论**: 加入我们的讨论区分享经验和技巧

### Issue 模板

报告问题时请包含：
- 操作系统和Python版本
- 错误的完整日志
- 复现步骤
- 预期行为vs实际行为

---

**如果这个项目对你有帮助，请给个 Star！**

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/literature-crawler&type=Date)](https://star-history.com/#your-username/literature-crawler&Date)