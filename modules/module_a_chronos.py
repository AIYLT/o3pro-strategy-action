"""
模块A: Chronos | 策略评分引擎
功能: 对每笔日内策略进行核心量化评分, 计算综合得分(0.00–1.00)
用途: 用于策略排序与优先级判定, 并将得分转换为可信度A(百分比)
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from utils import ModuleResult, TimeTracker, logger
from config import POLYGON_API_KEY, POLYGON_BASE_URL


class ChronosEngine:
    """Chronos策略评分引擎"""
    
    def __init__(self):
        self.name = "Chronos"
        self.module_id = "A"
        self.api_key = POLYGON_API_KEY
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        分析股票并生成策略评分
        
        Args:
            symbol: 股票代码
            data: 输入数据包含价格、成交量等信息
            
        Returns:
            ModuleResult: 包含评分和可信度的结果
        """
        timer = TimeTracker().start()
        
        try:
            # 获取历史数据用于评分计算
            historical_data = await self._get_historical_data(symbol)
            
            # 计算各项评分指标
            technical_score = self._calculate_technical_score(symbol, data, historical_data)
            volume_score = self._calculate_volume_score(data, historical_data)
            momentum_score = self._calculate_momentum_score(data, historical_data)
            volatility_score = self._calculate_volatility_score(historical_data)
            
            # 计算综合得分 (0.00-1.00)
            composite_score = self._calculate_composite_score(
                technical_score,
                volume_score, 
                momentum_score,
                volatility_score
            )
            
            # 转换为可信度百分比
            confidence_a = min(composite_score * 100, 100.0)
            
            # 多窗口回测精度计算
            backtest_accuracy = await self._calculate_backtest_accuracy(symbol, historical_data)
            
            # 最终调整得分
            final_score = self._adjust_score_with_backtest(composite_score, backtest_accuracy)
            final_confidence = min(final_score * 100, 100.0)
            
            execution_time = timer.stop()
            
            result_data = {
                "composite_score": round(final_score, 4),
                "confidence_percentage": round(final_confidence, 2),
                "technical_score": round(technical_score, 4),
                "volume_score": round(volume_score, 4),
                "momentum_score": round(momentum_score, 4),
                "volatility_score": round(volatility_score, 4),
                "backtest_accuracy": round(backtest_accuracy, 4),
                "ranking_eligible": final_confidence >= 0.0,  # Chronos无最低要求
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Chronos评分完成: {symbol} - 得分: {final_score:.4f}, 可信度: {final_confidence:.2f}%")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence / 100,  # 转换为0-1范围
                data=result_data,
                execution_time=execution_time,
                status="success"
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Chronos分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_historical_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取历史数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df.columns = ["volume", "volume_weighted_avg", "open", "close", "high", "low", "timestamp", "trades"]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            else:
                # 返回空DataFrame以避免错误
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"获取历史数据失败 {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_technical_score(self, symbol: str, current_data: Dict, historical_data: pd.DataFrame) -> float:
        """计算技术指标得分"""
        if historical_data.empty:
            return 0.5  # 默认中性得分
            
        try:
            # 获取当前价格
            current_price = current_data.get("price", historical_data["close"].iloc[-1])
            
            # 计算移动平均线
            ma_20 = historical_data["close"].rolling(20).mean().iloc[-1]
            ma_50 = historical_data["close"].rolling(50).mean().iloc[-1] if len(historical_data) >= 50 else ma_20
            
            # 价格相对位置得分
            price_position_score = 0.0
            if current_price > ma_20:
                price_position_score += 0.3
            if current_price > ma_50:
                price_position_score += 0.2
            if ma_20 > ma_50:
                price_position_score += 0.2
                
            # 计算RSI
            price_changes = historical_data["close"].diff()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            
            if len(gains) >= 14:
                avg_gain = gains.rolling(14).mean().iloc[-1]
                avg_loss = losses.rolling(14).mean().iloc[-1]
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # RSI得分 (30-70为理想区间)
                    if 30 <= rsi <= 70:
                        rsi_score = 0.3
                    elif rsi < 30:  # 超卖
                        rsi_score = 0.2
                    else:  # 超买
                        rsi_score = 0.1
                else:
                    rsi_score = 0.15
            else:
                rsi_score = 0.15
                
            return min(price_position_score + rsi_score, 1.0)
            
        except Exception as e:
            logger.warning(f"技术指标计算失败: {str(e)}")
            return 0.5
    
    def _calculate_volume_score(self, current_data: Dict, historical_data: pd.DataFrame) -> float:
        """计算成交量得分"""
        if historical_data.empty:
            return 0.5
            
        try:
            current_volume = current_data.get("volume", 0)
            avg_volume = historical_data["volume"].mean()
            
            if avg_volume == 0:
                return 0.5
                
            # 成交量比率
            volume_ratio = current_volume / avg_volume
            
            # 成交量得分
            if volume_ratio >= 3.0:  # 显著放量
                return 1.0
            elif volume_ratio >= 2.0:
                return 0.8
            elif volume_ratio >= 1.5:
                return 0.6
            elif volume_ratio >= 1.0:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"成交量计算失败: {str(e)}")
            return 0.5
    
    def _calculate_momentum_score(self, current_data: Dict, historical_data: pd.DataFrame) -> float:
        """计算动量得分"""
        if historical_data.empty or len(historical_data) < 5:
            return 0.5
            
        try:
            # 计算价格动量
            current_price = current_data.get("price", historical_data["close"].iloc[-1])
            price_5d_ago = historical_data["close"].iloc[-5]
            
            momentum_5d = (current_price - price_5d_ago) / price_5d_ago
            
            # 动量得分
            if momentum_5d >= 0.05:  # 5日涨幅≥5%
                return 1.0
            elif momentum_5d >= 0.03:  # 3%-5%
                return 0.8
            elif momentum_5d >= 0.01:  # 1%-3%
                return 0.6
            elif momentum_5d >= -0.01:  # -1%-1%
                return 0.4
            else:  # 跌幅>1%
                return 0.2
                
        except Exception as e:
            logger.warning(f"动量计算失败: {str(e)}")
            return 0.5
    
    def _calculate_volatility_score(self, historical_data: pd.DataFrame) -> float:
        """计算波动率得分"""
        if historical_data.empty or len(historical_data) < 20:
            return 0.5
            
        try:
            # 计算20日波动率
            returns = historical_data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 波动率得分 (适中波动率更好)
            if 0.15 <= volatility <= 0.35:  # 15%-35%年化波动率
                return 1.0
            elif 0.10 <= volatility <= 0.50:
                return 0.8
            elif volatility <= 0.60:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"波动率计算失败: {str(e)}")
            return 0.5
    
    def _calculate_composite_score(self, technical: float, volume: float, momentum: float, volatility: float) -> float:
        """计算综合得分"""
        # 权重分配
        weights = {
            "technical": 0.35,    # 技术指标35%
            "volume": 0.25,       # 成交量25%
            "momentum": 0.25,     # 动量25%
            "volatility": 0.15    # 波动率15%
        }
        
        composite = (
            technical * weights["technical"] +
            volume * weights["volume"] +
            momentum * weights["momentum"] +
            volatility * weights["volatility"]
        )
        
        return min(max(composite, 0.0), 1.0)
    
    async def _calculate_backtest_accuracy(self, symbol: str, historical_data: pd.DataFrame) -> float:
        """计算多窗口回测精度"""
        try:
            if historical_data.empty or len(historical_data) < 30:
                return 0.75  # 默认精度
                
            # 简化的回测模拟
            accuracy_scores = []
            
            # 30日窗口回测
            for i in range(min(30, len(historical_data) - 5)):
                end_idx = len(historical_data) - i - 1
                start_idx = max(0, end_idx - 5)
                
                if start_idx < end_idx:
                    price_change = (historical_data["close"].iloc[end_idx] - historical_data["close"].iloc[start_idx]) / historical_data["close"].iloc[start_idx]
                    
                    # 简化的信号判断: 价格上涨则视为成功
                    if price_change > 0:
                        accuracy_scores.append(1.0)
                    else:
                        accuracy_scores.append(0.0)
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.75
            
        except Exception as e:
            logger.warning(f"回测精度计算失败: {str(e)}")
            return 0.75
    
    def _adjust_score_with_backtest(self, base_score: float, backtest_accuracy: float) -> float:
        """根据回测精度调整得分"""
        # 回测精度权重15%
        adjusted_score = base_score * 0.85 + backtest_accuracy * 0.15
        return min(max(adjusted_score, 0.0), 1.0) 