# 智鉴富策略模块 (O3Pro Strategy Action)

## 项目描述
这是一个基于 GPT-4o 模型的智能投资顾问系统，专注于美股日内交易策略分析和决策。系统集成了11个核心策略模块，目标命中率≥80%。

## 功能模块
- **模块A (Chronos)**: 策略评分引擎
- **模块B (Minerva)**: 行为验证器  
- **模块C (Aegis)**: 风控调度系统
- **模块D (Fenrir)**: 主力结构识别器
- **模块E (Hermes)**: 事件回溯引擎
- **模块F (AlphaForge)**: 因子扩展层
- **模块G (TerraFilter)**: 环境过滤器
- **模块S (Helios)**: 市场扫描与选股器
- **模块X (Cerberus)**: 异常防护模块
- **模块Y (Oracle)**: 全模块精度整合器(核心)
- **模块Z (EchoLog)**: 模拟与回测器

## API 端点
- `POST /analyze` - 分析单个股票
- `POST /scan` - 批量扫描股票池
- `GET /health` - 健康检查

## 环境配置
创建 `.env` 文件：
```
OPENAI_API_KEY=your_openai_key
POLYGON_API_KEY=your_polygon_key
```

## 运行方式
```bash
pip install -r requirements.txt
python main.py
```

## 模型要求
- 必须使用 OpenAI GPT o3-2025-04-16 模型
- 系统会自动验证模型版本

## 数据源
- Polygon API: 实时和历史市场数据
- OpenAI API: AI分析和决策 