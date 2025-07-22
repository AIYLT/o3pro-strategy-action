"""
模块B: Minerva | 行为验证器
功能: 分析该标的过去200笔以上类似结构是否有稳定命中表现
用途: 确认技术形态与资金行为是否一致, 作为信号真实性依据
数据依据: 近200笔同型K线/成交结构回测表现
可信度B: 命中率百分比; 若<70%, 自动降级
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import requests
from utils import ModuleResult, TimeTracker, logger
from config import POLYGON_API_KEY, POLYGON_BASE_URL, CONFIDENCE_THRESHOLDS


class MinervaEngine:
    """Minerva行为验证器"""
    
    def __init__(self):
        self.name = "Minerva"
        self.module_id = "B"
        self.api_key = POLYGON_API_KEY
        self.min_confidence = CONFIDENCE_THRESHOLDS['B']  # 70%
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        分析技术形态与资金行为一致性
        
        Args:
            symbol: 股票代码
            data: 输入数据包含价格、成交量等信息
            
        Returns:
            ModuleResult: 包含行为验证结果和可信度
        """
        timer = TimeTracker().start()
        
        try:
            # 获取扩展历史数据用于模式识别
            historical_data = await self._get_extended_historical_data(symbol)
            
            if historical_data.empty:
                logger.warning(f"Minerva: {symbol} 历史数据不足")
                return self._create_error_result("历史数据不足", timer.stop())
            
            # 识别当前K线模式
            current_pattern = self._identify_current_pattern(data, historical_data)
            
            # 查找历史相似模式
            similar_patterns = self._find_similar_patterns(
                current_pattern, historical_data
            )
            
            if len(similar_patterns) < 50:  # 至少需要50个相似模式
                logger.warning(f"Minerva: {symbol} 相似模式不足 ({len(similar_patterns)})")
                return self._create_insufficient_data_result(similar_patterns, timer.stop())
            
            # 计算历史模式命中率
            hit_rate = self._calculate_pattern_hit_rate(similar_patterns, historical_data)
            
            # 验证资金行为一致性
            fund_behavior_score = self._analyze_fund_behavior(
                current_pattern, similar_patterns, historical_data
            )
            
            # 计算最终可信度
            final_confidence = self._calculate_final_confidence(
                hit_rate, fund_behavior_score
            )
            
            # 判断是否达到最低可信度要求
            is_valid = final_confidence >= self.min_confidence
            status = "success" if is_valid else "degraded"
            
            execution_time = timer.stop()
            
            result_data = {
                "pattern_hit_rate": round(hit_rate, 4),
                "fund_behavior_score": round(fund_behavior_score, 4),
                "final_confidence": round(final_confidence, 4),
                "similar_patterns_count": len(similar_patterns),
                "pattern_type": current_pattern.get("type", "unknown"),
                "is_degraded": not is_valid,
                "min_confidence_threshold": self.min_confidence,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if is_valid:
                logger.info(f"Minerva验证通过: {symbol} - 命中率: {hit_rate:.2%}, 可信度: {final_confidence:.2%}")
            else:
                logger.warning(f"Minerva自动降级: {symbol} - 可信度: {final_confidence:.2%} < {self.min_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence,
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Minerva分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_extended_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """获取扩展历史数据(1年)用于模式识别"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 获取日线数据
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df.columns = ["volume", "vwap", "open", "close", "high", "low", "timestamp", "transactions"]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                # 计算技术指标
                df["rsi"] = self._calculate_rsi(df["close"])
                df["ma_20"] = df["close"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
                df["price_change"] = df["close"].pct_change()
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"获取扩展历史数据失败 {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _identify_current_pattern(self, current_data: Dict, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """识别当前K线模式"""
        try:
            if historical_data.empty:
                return {"type": "unknown"}
            
            # 获取最近价格数据
            recent_data = historical_data.tail(5)
            current_price = current_data.get("price", recent_data["close"].iloc[-1])
            current_volume = current_data.get("volume", recent_data["volume"].iloc[-1])
            
            # 计算模式特征
            pattern = {
                "type": "intraday_breakout",
                "price_momentum": (current_price - recent_data["close"].iloc[0]) / recent_data["close"].iloc[0],
                "volume_surge": current_volume / recent_data["volume"].mean() if recent_data["volume"].mean() > 0 else 1.0,
                "ma_position": 1 if current_price > recent_data["ma_20"].iloc[-1] else 0,
                "rsi_level": recent_data["rsi"].iloc[-1] if not pd.isna(recent_data["rsi"].iloc[-1]) else 50,
                "volatility": recent_data["price_change"].std(),
                "trend_strength": self._calculate_trend_strength(recent_data)
            }
            
            # 分类模式类型
            if pattern["price_momentum"] > 0.03 and pattern["volume_surge"] > 2.0:
                pattern["type"] = "breakout_with_volume"
            elif pattern["price_momentum"] > 0.02:
                pattern["type"] = "momentum_surge"
            elif pattern["volume_surge"] > 3.0:
                pattern["type"] = "volume_anomaly"
            else:
                pattern["type"] = "consolidation"
            
            return pattern
            
        except Exception as e:
            logger.warning(f"模式识别失败: {str(e)}")
            return {"type": "unknown"}
    
    def _find_similar_patterns(self, current_pattern: Dict, historical_data: pd.DataFrame) -> List[Dict]:
        """查找历史相似模式"""
        try:
            similar_patterns = []
            
            if historical_data.empty or current_pattern.get("type") == "unknown":
                return similar_patterns
            
            # 滑动窗口查找相似模式
            for i in range(5, len(historical_data) - 5):  # 保留前后5天用于计算
                window_data = historical_data.iloc[i-5:i+1]
                
                # 计算窗口模式特征
                window_momentum = (window_data["close"].iloc[-1] - window_data["close"].iloc[0]) / window_data["close"].iloc[0]
                window_volume_surge = window_data["volume"].iloc[-1] / window_data["volume"].iloc[:-1].mean() if window_data["volume"].iloc[:-1].mean() > 0 else 1.0
                window_ma_position = 1 if window_data["close"].iloc[-1] > window_data["ma_20"].iloc[-1] else 0
                window_rsi = window_data["rsi"].iloc[-1] if not pd.isna(window_data["rsi"].iloc[-1]) else 50
                
                # 计算相似度
                similarity = self._calculate_pattern_similarity(
                    current_pattern,
                    {
                        "momentum": window_momentum,
                        "volume_surge": window_volume_surge,
                        "ma_position": window_ma_position,
                        "rsi": window_rsi
                    }
                )
                
                # 相似度阈值 (可调整)
                if similarity > 0.7:
                    # 计算未来5日表现
                    future_data = historical_data.iloc[i+1:i+6]
                    if len(future_data) >= 5:
                        future_return = (future_data["close"].iloc[-1] - window_data["close"].iloc[-1]) / window_data["close"].iloc[-1]
                        
                        similar_patterns.append({
                            "index": i,
                            "similarity": similarity,
                            "future_return": future_return,
                            "success": future_return > 0.01,  # 1%收益阈值
                            "timestamp": window_data.index[-1]
                        })
            
            # 按相似度排序，返回最相似的200个
            similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_patterns[:200]
            
        except Exception as e:
            logger.warning(f"查找相似模式失败: {str(e)}")
            return []
    
    def _calculate_pattern_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """计算模式相似度"""
        try:
            # 各特征权重
            weights = {
                "momentum": 0.3,
                "volume_surge": 0.25,
                "ma_position": 0.2,
                "rsi": 0.25
            }
            
            total_similarity = 0.0
            
            # 动量相似度
            momentum_diff = abs(pattern1.get("price_momentum", 0) - pattern2.get("momentum", 0))
            momentum_similarity = max(0, 1 - momentum_diff / 0.1)  # 10%差异归一化
            total_similarity += momentum_similarity * weights["momentum"]
            
            # 成交量相似度
            volume_diff = abs(pattern1.get("volume_surge", 1) - pattern2.get("volume_surge", 1))
            volume_similarity = max(0, 1 - volume_diff / 5.0)  # 5倍差异归一化
            total_similarity += volume_similarity * weights["volume_surge"]
            
            # 均线位置相似度
            ma_similarity = 1.0 if pattern1.get("ma_position", 0) == pattern2.get("ma_position", 0) else 0.0
            total_similarity += ma_similarity * weights["ma_position"]
            
            # RSI相似度
            rsi_diff = abs(pattern1.get("rsi_level", 50) - pattern2.get("rsi", 50))
            rsi_similarity = max(0, 1 - rsi_diff / 50)  # 50点差异归一化
            total_similarity += rsi_similarity * weights["rsi"]
            
            return min(total_similarity, 1.0)
            
        except Exception as e:
            logger.warning(f"相似度计算失败: {str(e)}")
            return 0.0
    
    def _calculate_pattern_hit_rate(self, similar_patterns: List[Dict], historical_data: pd.DataFrame) -> float:
        """计算历史模式命中率"""
        try:
            if not similar_patterns:
                return 0.0
            
            success_count = sum(1 for pattern in similar_patterns if pattern.get("success", False))
            total_count = len(similar_patterns)
            
            return success_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"命中率计算失败: {str(e)}")
            return 0.0
    
    def _analyze_fund_behavior(self, current_pattern: Dict, similar_patterns: List[Dict], historical_data: pd.DataFrame) -> float:
        """分析资金行为一致性"""
        try:
            if not similar_patterns:
                return 0.5
            
            # 分析成功模式的资金特征
            successful_patterns = [p for p in similar_patterns if p.get("success", False)]
            
            if not successful_patterns:
                return 0.3
            
            # 计算成功模式的平均成交量特征
            avg_volume_surge = np.mean([
                historical_data.iloc[p["index"]]["volume_ratio"] 
                for p in successful_patterns 
                if not pd.isna(historical_data.iloc[p["index"]]["volume_ratio"])
            ])
            
            current_volume_surge = current_pattern.get("volume_surge", 1.0)
            
            # 资金行为一致性得分
            if avg_volume_surge > 0:
                consistency_score = min(current_volume_surge / avg_volume_surge, 2.0) / 2.0
            else:
                consistency_score = 0.5
            
            # 额外加分项
            if current_volume_surge > 2.0:  # 当前成交量显著放大
                consistency_score += 0.2
            
            return min(consistency_score, 1.0)
            
        except Exception as e:
            logger.warning(f"资金行为分析失败: {str(e)}")
            return 0.5
    
    def _calculate_final_confidence(self, hit_rate: float, fund_behavior_score: float) -> float:
        """计算最终可信度"""
        # 命中率权重70%, 资金行为权重30%
        final_confidence = hit_rate * 0.7 + fund_behavior_score * 0.3
        return min(max(final_confidence, 0.0), 1.0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.warning(f"RSI计算失败: {str(e)}")
            return pd.Series(index=prices.index, data=50)  # 默认中性RSI
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """计算趋势强度"""
        try:
            if len(data) < 3:
                return 0.0
            
            # 计算价格趋势的一致性
            price_changes = data["close"].diff().dropna()
            
            if len(price_changes) == 0:
                return 0.0
            
            positive_changes = (price_changes > 0).sum()
            total_changes = len(price_changes)
            
            trend_consistency = positive_changes / total_changes
            
            # 归一化到0-1范围，0.5为中性
            return abs(trend_consistency - 0.5) * 2
            
        except Exception as e:
            logger.warning(f"趋势强度计算失败: {str(e)}")
            return 0.0
    
    def _create_error_result(self, error_msg: str, execution_time: float) -> ModuleResult:
        """创建错误结果"""
        return ModuleResult(
            module_name=self.name,
            confidence=0.0,
            data={"error": error_msg, "is_degraded": True},
            execution_time=execution_time,
            status="error"
        )
    
    def _create_insufficient_data_result(self, patterns: List, execution_time: float) -> ModuleResult:
        """创建数据不足结果"""
        return ModuleResult(
            module_name=self.name,
            confidence=0.3,  # 低可信度
            data={
                "warning": "相似模式数量不足",
                "pattern_count": len(patterns),
                "is_degraded": True,
                "min_required": 50
            },
            execution_time=execution_time,
            status="degraded"
        ) 