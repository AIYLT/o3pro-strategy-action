import os
from dotenv import load_dotenv

load_dotenv()

# API 配置 - 使用环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "your_polygon_api_key_here")

# 模型配置
REQUIRED_MODEL = "o3-2025-04-16"
TARGET_HIT_RATE = 0.80  # 目标命中率 ≥80%

# 系统配置
MAX_ANALYSIS_TIME = 1800  # 30分钟
ANALYSIS_TIMEOUT = 15 * 60  # 15分钟补全时间

# 模块优先级 (按照策略文件要求)
MODULE_PRIORITY = {
    'Y': 1,  # Oracle - 最高优先级
    'S': 2,  # Helios  
    'A': 3,  # Chronos
    'G': 4,  # TerraFilter
    'B': 5,  # Minerva
    'F': 6,  # AlphaForge
    'D': 7,  # Fenrir
    'E': 8,  # Hermes
    'C': 9,  # Aegis
    'X': 10, # Cerberus
    'Z': 11  # EchoLog
}

# 可信度阈值
MIN_CONFIDENCE_THRESHOLD = 0.75
MIN_MODULE_CONFIDENCE = {
    'A': 0.0,   # Chronos - 无最低要求
    'B': 0.70,  # Minerva - 70%
    'C': 0.75,  # Aegis - 75%
    'D': 0.0,   # Fenrir - 无最低要求
    'E': 0.0,   # Hermes - 无最低要求  
    'F': 0.60,  # AlphaForge - 60%
    'G': 0.0,   # TerraFilter - 无最低要求
    'S': 0.60,  # Helios - 60%
    'X': 0.0,   # Cerberus - 异常检测模块
    'Y': 0.80,  # Oracle - 80% (核心模块)
    'Z': 0.0    # EchoLog - 回测模块
}

# Polygon API 配置
POLYGON_BASE_URL = "https://api.polygon.io"

# 风控配置
MAX_POSITION_RISK = 0.03  # 单笔风险不超过3%
MAX_DRAWDOWN = 0.05       # 最大回撤5%
MIN_WIN_RATE = 0.65       # 最低胜率65%
MIN_PROFIT_RATIO = 1.0    # 最低盈亏比1:1 