import os
from typing import Dict, List

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "your_polygon_api_key_here")

# o3模型配置 (严格按照文档要求)
OPENAI_MODEL = "o3-2025-04-16"  # o3模型
MODEL_VALIDATION_REQUIRED = True
TARGET_HIT_RATE = 80.0  # 目标命中率≥80%

# Polygon Advanced API配置
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_ADVANCED_FEATURES = {
    "real_time_stocks": "/v2/aggs/ticker/{symbol}/prev",
    "ticker_details": "/v3/reference/tickers/{symbol}",
    "financials": "/vX/reference/financials",
    "market_news": "/v2/reference/news",
    "options_chain": "/v3/reference/options/contracts",
    "trades": "/v3/trades/{symbol}",
    "technical_indicators": "/v1/indicators",
    "market_status": "/v1/marketstatus/now",
    "corporate_actions": "/vX/reference/dividends",
    "related_companies": "/v1/related-companies/{symbol}",
    "historical_volatility": "/v1/indicators/sma/{symbol}",
    "level2_data": "/v2/snapshot/locale/us/markets/stocks/tickers"
}

# 模块优先级 (严格按照文档)
MODULE_PRIORITY = {
    "Y": 1,  # Oracle - 最高优先级
    "S": 2,  # Helios - 市场扫描
    "A": 3,  # Chronos - 策略评分
    "G": 4,  # TerraFilter - 环境过滤
    "B": 5,  # Minerva - 行为验证
    "F": 6,  # AlphaForge - 因子扩展
    "D": 7,  # Fenrir - 主力结构识别
    "E": 8,  # Hermes - 事件回溯
    "C": 9,  # Aegis - 风控调度
    "X": 10, # Cerberus - 异常防护
    "Z": 11  # EchoLog - 回测记录
}

# 可信度阈值配置
CONFIDENCE_THRESHOLDS = {
    "global_minimum": 75.0,    # 全局最低可信度
    "hit_rate_minimum": 80.0,  # 最低命中率要求
    "module_minimum": {
        "A": 70.0,  # Chronos
        "B": 70.0,  # Minerva  
        "C": 75.0,  # Aegis
        "D": 60.0,  # Fenrir
        "E": 50.0,  # Hermes
        "F": 60.0,  # AlphaForge
        "G": 65.0,  # TerraFilter
        "S": 60.0,  # Helios
        "X": 70.0,  # Cerberus
        "Y": 80.0,  # Oracle
        "Z": 65.0   # EchoLog
    }
}

# Helios扫描配置 (按文档S-Score机制)
HELIOS_CONFIG = {
    "min_premarket_volume": 500000,
    "min_premarket_change": 3.0,
    "min_gap_percentage": 2.0,
    "min_rvol": 3.0,
    "min_atr_ratio": 4.0,
    "min_stock_price": 10.0,
    "s_score_weights": {
        "rvol": 0.25,
        "atr_volatility": 0.20,
        "gap_size": 0.15,
        "ma_breakout": 0.20,
        "news_sentiment": 0.20
    },
    "s_score_thresholds": {
        "minimum": 60,
        "strong_pool": 80
    }
}

# 风险管理配置
RISK_CONFIG = {
    "max_position_risk": 3.0,  # 单笔最大风险3%
    "max_portfolio_risk": 10.0,
    "atr_multiplier": 1.5,
    "var_confidence": 95.0
}

# 回测配置
BACKTEST_CONFIG = {
    "periods": ["30d", "6m", "2y"],
    "min_hit_rate": 65.0,
    "min_profit_loss_ratio": 1.0,
    "max_drawdown": 5.0
}

# 系统身份配置 (严格按照文档)
SYSTEM_IDENTITY = {
    "name": "智鑑富專屬智能投資顧問",
    "role": "AI日內交易決策分析師",
    "target": "機構級水準並以目標命中率≥80%為基準",
    "scope": "美股日內選股, 多因子信號融合, 自動風險控制與全流程歷史回測"
}

# 输出格式模板
OUTPUT_TEMPLATES = {
    "hit_rate_format": "🏁 命中率: {:.2f}%",
    "confidence_format": "全局可信度: {:.2f}%",
    "timing_format": "⏱️分析时间: {:.2f}秒",
    "timestamp_format": "🕓分析时间戳: {timestamp}",
    "recommendation_levels": ["强烈买入", "买入", "持有", "卖出", "强烈卖出"]
} 