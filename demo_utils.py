"""
演示模式工具函数
当没有真实API密钥时，提供模拟数据和响应
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from config import DEMO_DATA, DEMO_MODE, IS_DEMO_OPENAI, IS_DEMO_POLYGON

def get_demo_stock_data(symbol: str) -> Dict[str, Any]:
    """获取演示股票数据"""
    if symbol.upper() in DEMO_DATA:
        return DEMO_DATA[symbol.upper()]
    
    # 为未知股票生成随机数据
    base_price = random.uniform(50, 500)
    return {
        "price": round(base_price, 2),
        "volume": random.randint(1_000_000, 100_000_000),
        "change": round(random.uniform(-10, 10), 2),
        "change_percent": round(random.uniform(-5, 5), 2),
        "market_cap": random.randint(1_000_000_000, 3_000_000_000_000),
        "pe_ratio": round(random.uniform(15, 80), 1),
        "sector": random.choice(["Technology", "Healthcare", "Finance", "Consumer", "Energy"]),
        "beta": round(random.uniform(0.5, 2.5), 2)
    }

def get_demo_ai_analysis(symbol: str, modules_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成演示AI分析结果"""
    if not DEMO_MODE or not IS_DEMO_OPENAI:
        return None
    
    # 模拟延迟
    time.sleep(0.5)
    
    stock_data = get_demo_stock_data(symbol)
    
    # 基于模块数据生成智能决策
    avg_confidence = sum([m.get('confidence', 50) for m in modules_data.values()]) / len(modules_data)
    
    if avg_confidence >= 70:
        recommendation = "买入" if stock_data['change_percent'] > 0 else "持有"
        action = "BUY" if stock_data['change_percent'] > 0 else "HOLD"
    elif avg_confidence >= 50:
        recommendation = "持有"
        action = "HOLD"
    else:
        recommendation = "观望"
        action = "WAIT"
    
    return {
        "recommendation": recommendation,
        "action": action,
        "confidence": min(avg_confidence + random.uniform(-5, 15), 95),
        "reasoning": f"基于{len(modules_data)}个模块的综合分析，{symbol}当前技术面表现{'积极' if avg_confidence > 60 else '中性'}",
        "entry_price": stock_data['price'],
        "target_price": round(stock_data['price'] * random.uniform(1.02, 1.15), 2),
        "stop_loss": round(stock_data['price'] * random.uniform(0.92, 0.98), 2),
        "risk_level": "中等" if 50 <= avg_confidence <= 75 else ("低" if avg_confidence > 75 else "高")
    }

def get_demo_market_data() -> Dict[str, Any]:
    """获取演示市场数据"""
    return {
        "SPY": {"price": 445.67, "change": 1.23, "change_percent": 0.28},
        "QQQ": {"price": 375.89, "change": 2.45, "change_percent": 0.66},
        "VIX": {"price": 14.52, "change": -0.34, "change_percent": -2.29},
        "market_sentiment": random.choice(["乐观", "中性", "谨慎"]),
        "sector_rotation": random.choice(["科技", "金融", "医疗", "能源"])
    }

def generate_demo_historical_data(symbol: str, days: int = 60) -> List[Dict[str, Any]]:
    """生成演示历史数据"""
    data = []
    base_price = get_demo_stock_data(symbol)['price']
    current_price = base_price
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        
        # 模拟价格波动
        change_percent = random.uniform(-0.05, 0.05)  # 每日±5%波动
        current_price *= (1 + change_percent)
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(current_price * 0.99, 2),
            "high": round(current_price * 1.02, 2),
            "low": round(current_price * 0.98, 2),
            "close": round(current_price, 2),
            "volume": random.randint(10_000_000, 80_000_000)
        })
    
    return data

def should_use_demo_mode() -> bool:
    """检查是否应该使用演示模式"""
    return DEMO_MODE and (IS_DEMO_OPENAI or IS_DEMO_POLYGON)

def get_demo_backtest_results(symbol: str) -> Dict[str, Any]:
    """生成演示回测结果"""
    return {
        "symbol": symbol,
        "period": "2年",
        "total_trades": random.randint(45, 120),
        "win_rate": round(random.uniform(72, 88), 1),
        "avg_return": round(random.uniform(1.2, 3.5), 2),
        "max_drawdown": round(random.uniform(3, 8), 1),
        "sharpe_ratio": round(random.uniform(1.5, 2.8), 2),
        "profit_factor": round(random.uniform(1.8, 2.5), 2),
        "confidence": round(random.uniform(82, 94), 1)
    } 