"""
模块D: Fenrir | 主力结构识别器
功能: 偵測暗池、大宗交易、Level-2買賣壓結構
用途: 標註暗池偵測結果, 佐證是否有大資金背書
数据依据: 成交筆數異常、資金流量突變、暗池事件回測
可信度D: 資金方向一致性 + 歷史成功率
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import requests
from ..utils import ModuleResult, TimeTracker, logger
from ..config import POLYGON_API_KEY, POLYGON_BASE_URL


class FenrirEngine:
    """Fenrir主力结构识别器"""
    
    def __init__(self):
        self.name = "Fenrir"
        self.module_id = "D"
        self.api_key = POLYGON_API_KEY
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        检测主力资金结构和暗池活动
        
        Args:
            symbol: 股票代码
            data: 输入数据包含价格、成交量等信息
            
        Returns:
            ModuleResult: 包含主力资金分析结果
        """
        timer = TimeTracker().start()
        
        try:
            # 获取详细交易数据
            trade_data = await self._get_trade_data(symbol)
            
            # 检测大宗交易
            block_trades = self._detect_block_trades(trade_data)
            
            # 分析资金流向
            fund_flow = self._analyze_fund_flow(trade_data, data)
            
            # 检测暗池活动模式
            dark_pool_activity = self._detect_dark_pool_patterns(trade_data)
            
            # 计算资金方向一致性
            direction_consistency = self._calculate_direction_consistency(
                fund_flow, block_trades
            )
            
            # 计算最终可信度
            final_confidence = self._calculate_confidence(
                direction_consistency, dark_pool_activity, fund_flow
            )
            
            execution_time = timer.stop()
            
            result_data = {
                "block_trades_detected": len(block_trades),
                "dark_pool_volume": dark_pool_activity.get("estimated_volume", 0),
                "fund_flow_direction": fund_flow.get("direction", "neutral"),
                "fund_flow_strength": fund_flow.get("strength", 0),
                "direction_consistency": round(direction_consistency, 4),
                "final_confidence": round(final_confidence, 4),
                "institutional_backing": final_confidence > 0.6,
                "major_trades": block_trades[:5],  # 前5笔大宗交易
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Fenrir检测完成: {symbol} - 暗池: ${dark_pool_activity.get('estimated_volume', 0)/1e6:.1f}M, 可信度: {final_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence,
                data=result_data,
                execution_time=execution_time,
                status="success"
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Fenrir分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_trade_data(self, symbol: str) -> Dict[str, Any]:
        """获取交易数据 (简化实现)"""
        try:
            # 获取当日分时数据
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/5/minute/{today}/{today}"
            
            params = {"apikey": self.api_key, "limit": 100}
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            
            return []
            
        except Exception as e:
            logger.warning(f"获取交易数据失败: {str(e)}")
            return []
    
    def _detect_block_trades(self, trade_data: List[Dict]) -> List[Dict]:
        """检测大宗交易"""
        block_trades = []
        
        try:
            if not trade_data:
                return block_trades
            
            # 转换为DataFrame便于分析
            df = pd.DataFrame(trade_data)
            if df.empty:
                return block_trades
            
            # 计算平均成交量
            avg_volume = df['v'].mean() if 'v' in df.columns else 0
            
            # 检测大宗交易 (成交量超过平均值的5倍)
            for idx, row in df.iterrows():
                volume = row.get('v', 0)
                if volume > avg_volume * 5 and volume > 100000:  # 至少10万股
                    block_trades.append({
                        "timestamp": row.get('t', 0),
                        "volume": volume,
                        "price": row.get('c', 0),
                        "value": volume * row.get('c', 0),
                        "volume_ratio": volume / avg_volume if avg_volume > 0 else 1
                    })
            
            return sorted(block_trades, key=lambda x: x['value'], reverse=True)
            
        except Exception as e:
            logger.warning(f"大宗交易检测失败: {str(e)}")
            return []
    
    def _analyze_fund_flow(self, trade_data: List[Dict], current_data: Dict) -> Dict[str, Any]:
        """分析资金流向"""
        try:
            if not trade_data:
                return {"direction": "neutral", "strength": 0}
            
            df = pd.DataFrame(trade_data)
            if df.empty or 'v' not in df.columns or 'c' not in df.columns:
                return {"direction": "neutral", "strength": 0}
            
            # 计算资金流
            df['money_flow'] = df['v'] * df['c']
            
            # 简化的买卖盘判断 (基于价格变化)
            df['price_change'] = df['c'].diff()
            
            # 上涨时成交量视为买盘，下跌时视为卖盘
            buy_flow = df[df['price_change'] > 0]['money_flow'].sum()
            sell_flow = df[df['price_change'] < 0]['money_flow'].sum()
            
            total_flow = buy_flow + sell_flow
            
            if total_flow == 0:
                return {"direction": "neutral", "strength": 0}
            
            # 计算净流入
            net_flow = buy_flow - sell_flow
            net_ratio = net_flow / total_flow
            
            # 判断方向和强度
            if net_ratio > 0.1:
                direction = "inflow"
            elif net_ratio < -0.1:
                direction = "outflow"
            else:
                direction = "neutral"
            
            strength = abs(net_ratio)
            
            return {
                "direction": direction,
                "strength": strength,
                "buy_flow": buy_flow,
                "sell_flow": sell_flow,
                "net_flow": net_flow
            }
            
        except Exception as e:
            logger.warning(f"资金流分析失败: {str(e)}")
            return {"direction": "neutral", "strength": 0}
    
    def _detect_dark_pool_patterns(self, trade_data: List[Dict]) -> Dict[str, Any]:
        """检测暗池活动模式 (简化实现)"""
        try:
            if not trade_data:
                return {"estimated_volume": 0, "probability": 0}
            
            df = pd.DataFrame(trade_data)
            if df.empty:
                return {"estimated_volume": 0, "probability": 0}
            
            # 简化的暗池检测逻辑
            total_volume = df['v'].sum() if 'v' in df.columns else 0
            
            # 检测异常成交量模式
            volume_std = df['v'].std() if 'v' in df.columns else 0
            volume_mean = df['v'].mean() if 'v' in df.columns else 0
            
            # 异常成交量占比
            if volume_mean > 0:
                anomaly_ratio = volume_std / volume_mean
            else:
                anomaly_ratio = 0
            
            # 估算暗池成交量 (基于异常模式)
            estimated_dark_volume = total_volume * min(anomaly_ratio * 0.1, 0.3)
            
            # 暗池概率
            dark_pool_probability = min(anomaly_ratio * 0.5, 1.0)
            
            return {
                "estimated_volume": estimated_dark_volume,
                "probability": dark_pool_probability,
                "anomaly_ratio": anomaly_ratio
            }
            
        except Exception as e:
            logger.warning(f"暗池检测失败: {str(e)}")
            return {"estimated_volume": 0, "probability": 0}
    
    def _calculate_direction_consistency(
        self, 
        fund_flow: Dict[str, Any], 
        block_trades: List[Dict]
    ) -> float:
        """计算资金方向一致性"""
        try:
            consistency_score = 0.0
            
            # 资金流方向得分
            direction = fund_flow.get("direction", "neutral")
            strength = fund_flow.get("strength", 0)
            
            if direction == "inflow":
                consistency_score += 0.4 * strength
            elif direction == "outflow":
                consistency_score += 0.2 * strength  # 流出时得分较低
            
            # 大宗交易支持得分
            if block_trades:
                # 大宗交易数量和规模
                block_count = len(block_trades)
                total_block_value = sum(trade['value'] for trade in block_trades)
                
                if block_count >= 3:  # 多笔大宗交易
                    consistency_score += 0.3
                elif block_count >= 1:
                    consistency_score += 0.2
                
                # 大宗交易金额
                if total_block_value > 50000000:  # 5000万以上
                    consistency_score += 0.3
                elif total_block_value > 10000000:  # 1000万以上
                    consistency_score += 0.2
            
            return min(consistency_score, 1.0)
            
        except Exception as e:
            logger.warning(f"方向一致性计算失败: {str(e)}")
            return 0.5
    
    def _calculate_confidence(
        self, 
        direction_consistency: float,
        dark_pool_activity: Dict[str, Any],
        fund_flow: Dict[str, Any]
    ) -> float:
        """计算最终可信度"""
        try:
            # 基础可信度来自方向一致性
            base_confidence = direction_consistency * 0.5
            
            # 暗池活动加成
            dark_pool_prob = dark_pool_activity.get("probability", 0)
            base_confidence += dark_pool_prob * 0.3
            
            # 资金流强度加成
            flow_strength = fund_flow.get("strength", 0)
            base_confidence += flow_strength * 0.2
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"可信度计算失败: {str(e)}")
            return 0.5 