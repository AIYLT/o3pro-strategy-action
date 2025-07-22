import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import asyncio
import json

class TimeTracker:
    """分析时间追踪器"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed_time
    
    @property
    def elapsed_time(self):
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return 0.0

def get_timestamp():
    """获取ISO-8601格式时间戳"""
    return datetime.now(timezone.utc).isoformat()

def validate_model_version(model_name: str, required_model: str) -> bool:
    """验证模型版本"""
    if model_name != required_model:
        raise ValueError(f"模型版本不符，封装无效。要求: {required_model}, 实际: {model_name}")
    return True

def calculate_confidence_score(modules_confidence: Dict[str, float], module_priority: Dict[str, int]) -> float:
    """计算全局可信度得分"""
    if not modules_confidence:
        return 0.0
    
    total_weight = 0
    weighted_sum = 0
    
    for module, confidence in modules_confidence.items():
        if module in module_priority:
            # 优先级越高(数字越小)，权重越大
            weight = 1.0 / module_priority[module]
            weighted_sum += confidence * weight
            total_weight += weight
    
    return round(weighted_sum / total_weight if total_weight > 0 else 0.0, 4)

def calculate_confidence(value1, value2=None) -> float:
    """简化的信心度计算函数 - 兼容性包装"""
    if value2 is None:
        return min(value1 / 100.0, 1.0) if isinstance(value1, (int, float)) else 0.8
    return min(value1 / value2, 1.0) if value2 > 0 else 0.8

def track_time():
    """创建时间追踪器"""
    return TimeTracker()

def format_price(price: float) -> str:
    """格式化价格显示"""
    return f"${price:.2f}"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:.2%}"

def create_signal_output(
    symbol: str,
    signal_type: str,
    confidence: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    position_size: int,
    modules_data: Dict[str, Any],
    analysis_time: float,
    hit_rate: float
) -> Dict[str, Any]:
    """创建标准信号输出格式"""
    
    timestamp = get_timestamp()
    
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "signal": {
            "type": signal_type,
            "confidence": confidence,
            "hit_rate": hit_rate
        },
        "prices": {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "expected_return": round((take_profit - entry_price) / entry_price, 4)
        },
        "position": {
            "size": position_size,
            "risk_percentage": round((entry_price - stop_loss) / entry_price, 4)
        },
        "modules": modules_data,
        "analysis": {
            "time_seconds": analysis_time,
            "timestamp": timestamp
        },
        "status": "valid" if confidence >= 0.75 and hit_rate >= 0.80 else "warning"
    }

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('strategy_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ModuleResult:
    """模块结果标准化类"""
    def __init__(self, module_name: str, confidence: float, data: Dict[str, Any], 
                 execution_time: float = 0.0, status: str = "success"):
        self.module_name = module_name
        self.confidence = confidence
        self.data = data
        self.execution_time = execution_time
        self.status = status
        self.timestamp = get_timestamp()
    
    def to_dict(self):
        return {
            "module": self.module_name,
            "confidence": self.confidence,
            "data": self.data,
            "execution_time": self.execution_time,
            "status": self.status,
            "timestamp": self.timestamp
        }
    
    def is_valid(self, min_confidence: float = 0.0) -> bool:
        return self.confidence >= min_confidence and self.status == "success"

async def run_with_timeout(coro, timeout_seconds: int):
    """运行协程，带超时处理"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"操作超时 ({timeout_seconds}秒)")
        return None 