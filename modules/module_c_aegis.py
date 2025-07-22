"""
模块C: Aegis | 风控调度系统
功能: 动态计算入场、止损、止盈区间, 结合ATR-20、VaR与期权隐含波动率
用途: 控制单笔风险不超过总资金1–3%
数据依据: 分时波动、历史ATR、IV曲面
可信度C: 区间有效性计算, 低于75%不得发信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import requests
from ..utils import ModuleResult, TimeTracker, logger
from ..config import (
    POLYGON_API_KEY, POLYGON_BASE_URL, MIN_MODULE_CONFIDENCE,
    MAX_POSITION_RISK, MAX_DRAWDOWN
)


class AegisEngine:
    """Aegis风控调度系统"""
    
    def __init__(self):
        self.name = "Aegis"
        self.module_id = "C"
        self.api_key = POLYGON_API_KEY
        self.min_confidence = MIN_MODULE_CONFIDENCE['C']  # 75%
        self.max_position_risk = MAX_POSITION_RISK  # 3%
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        风控分析并计算入场、止损、止盈区间
        
        Args:
            symbol: 股票代码
            data: 输入数据包含价格、成交量等信息
            
        Returns:
            ModuleResult: 包含风控参数和可信度
        """
        timer = TimeTracker().start()
        
        try:
            # 获取历史数据用于ATR计算
            historical_data = await self._get_historical_data(symbol)
            
            if historical_data.empty:
                logger.warning(f"Aegis: {symbol} 历史数据不足")
                return self._create_error_result("历史数据不足", timer.stop())
            
            # 计算ATR-20
            atr_20 = self._calculate_atr(historical_data, period=20)
            
            # 获取当前价格
            current_price = data.get("price", historical_data["close"].iloc[-1])
            
            # 计算VaR (Value at Risk)
            var_95 = self._calculate_var(historical_data, confidence_level=0.95)
            
            # 获取隐含波动率 (简化计算)
            implied_volatility = await self._get_implied_volatility(symbol, historical_data)
            
            # 计算风控区间
            risk_params = self._calculate_risk_parameters(
                current_price, atr_20, var_95, implied_volatility
            )
            
            # 计算建议仓位
            position_size = self._calculate_position_size(
                current_price, risk_params["stop_loss"], data.get("account_value", 100000)
            )
            
            # 验证区间有效性
            interval_validity = self._validate_risk_intervals(risk_params, current_price)
            
            # 计算最终可信度
            final_confidence = self._calculate_final_confidence(
                interval_validity, atr_20, var_95, implied_volatility
            )
            
            # 判断是否达到最低可信度要求
            is_valid = final_confidence >= self.min_confidence
            status = "success" if is_valid else "invalid"
            
            execution_time = timer.stop()
            
            result_data = {
                "entry_price": round(current_price, 2),
                "stop_loss": round(risk_params["stop_loss"], 2),
                "take_profit": round(risk_params["take_profit"], 2),
                "position_size": position_size,
                "risk_percentage": round(risk_params["risk_percentage"], 4),
                "atr_20": round(atr_20, 4),
                "var_95": round(var_95, 4),
                "implied_volatility": round(implied_volatility, 4),
                "interval_validity": round(interval_validity, 4),
                "final_confidence": round(final_confidence, 4),
                "is_valid": is_valid,
                "min_confidence_threshold": self.min_confidence,
                "expected_return": round((risk_params["take_profit"] - current_price) / current_price, 4),
                "risk_reward_ratio": round(risk_params.get("risk_reward_ratio", 0), 2),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if is_valid:
                logger.info(f"Aegis风控通过: {symbol} - 入场: ${current_price:.2f}, 止损: ${risk_params['stop_loss']:.2f}, 止盈: ${risk_params['take_profit']:.2f}")
            else:
                logger.warning(f"Aegis风控拒绝: {symbol} - 可信度: {final_confidence:.2%} < {self.min_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence,
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Aegis分析失败 {symbol}: {str(e)}")
            
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
                df.columns = ["volume", "vwap", "open", "close", "high", "low", "timestamp", "transactions"]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                # 计算每日收益率
                df["returns"] = df["close"].pct_change()
                df["true_range"] = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"] - df["close"].shift(1))
                    )
                )
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"获取历史数据失败 {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 20) -> float:
        """计算ATR (Average True Range)"""
        try:
            if "true_range" not in data.columns or len(data) < period:
                return 0.0
            
            atr = data["true_range"].rolling(window=period).mean().iloc[-1]
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.warning(f"ATR计算失败: {str(e)}")
            return 0.0
    
    def _calculate_var(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """计算VaR (Value at Risk)"""
        try:
            if "returns" not in data.columns or len(data) < 30:
                return 0.02  # 默认2%风险
            
            returns = data["returns"].dropna()
            
            if len(returns) == 0:
                return 0.02
            
            # 计算指定置信水平的VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return abs(var)  # 返回正值
            
        except Exception as e:
            logger.warning(f"VaR计算失败: {str(e)}")
            return 0.02
    
    async def _get_implied_volatility(self, symbol: str, historical_data: pd.DataFrame) -> float:
        """获取或估算隐含波动率"""
        try:
            # 由于期权数据API复杂，这里使用历史波动率作为IV的代理
            if "returns" not in historical_data.columns or len(historical_data) < 30:
                return 0.25  # 默认25%波动率
            
            returns = historical_data["returns"].dropna()
            
            if len(returns) == 0:
                return 0.25
            
            # 计算30日历史波动率作为IV代理
            historical_vol = returns.std() * np.sqrt(252)  # 年化
            
            # 通常IV会高于历史波动率
            implied_vol = historical_vol * 1.2
            
            return min(max(implied_vol, 0.1), 1.0)  # 限制在10%-100%范围
            
        except Exception as e:
            logger.warning(f"隐含波动率计算失败: {str(e)}")
            return 0.25
    
    def _calculate_risk_parameters(self, current_price: float, atr: float, var: float, iv: float) -> Dict[str, float]:
        """计算风控参数"""
        try:
            # 止损距离 = max(1.5*ATR, VaR*价格, IV*价格*0.5)
            atr_stop_distance = 1.5 * atr
            var_stop_distance = var * current_price
            iv_stop_distance = iv * current_price * 0.5
            
            stop_distance = max(atr_stop_distance, var_stop_distance, iv_stop_distance)
            
            # 确保止损距离不超过5%
            max_stop_distance = current_price * 0.05
            stop_distance = min(stop_distance, max_stop_distance)
            
            # 计算止损价格
            stop_loss = current_price - stop_distance
            
            # 止盈价格 = 入场价 + 2*止损距离 (1:2风险收益比)
            take_profit = current_price + (2 * stop_distance)
            
            # 风险百分比
            risk_percentage = stop_distance / current_price
            
            # 风险收益比
            profit_distance = take_profit - current_price
            risk_reward_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
            
            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_distance": stop_distance,
                "risk_percentage": risk_percentage,
                "risk_reward_ratio": risk_reward_ratio
            }
            
        except Exception as e:
            logger.warning(f"风控参数计算失败: {str(e)}")
            return {
                "stop_loss": current_price * 0.97,
                "take_profit": current_price * 1.06,
                "stop_distance": current_price * 0.03,
                "risk_percentage": 0.03,
                "risk_reward_ratio": 2.0
            }
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, account_value: float) -> int:
        """计算建议仓位大小"""
        try:
            # 单笔风险不超过账户资金的3%
            max_risk_amount = account_value * self.max_position_risk
            
            # 每股风险
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0:
                return 0
            
            # 计算最大股数
            max_shares = int(max_risk_amount / risk_per_share)
            
            # 确保不超过账户总价值的30%
            max_position_value = account_value * 0.3
            max_shares_by_value = int(max_position_value / entry_price)
            
            # 取两者最小值
            position_size = min(max_shares, max_shares_by_value)
            
            return max(position_size, 0)
            
        except Exception as e:
            logger.warning(f"仓位计算失败: {str(e)}")
            return 0
    
    def _validate_risk_intervals(self, risk_params: Dict, current_price: float) -> float:
        """验证风控区间有效性"""
        try:
            validity_score = 0.0
            
            # 检查止损价格合理性 (不超过5%，不少于1%)
            stop_loss_pct = (current_price - risk_params["stop_loss"]) / current_price
            if 0.01 <= stop_loss_pct <= 0.05:
                validity_score += 0.3
            elif 0.005 <= stop_loss_pct <= 0.08:
                validity_score += 0.2
            else:
                validity_score += 0.1
            
            # 检查风险收益比 (目标 >= 1.5)
            risk_reward = risk_params.get("risk_reward_ratio", 0)
            if risk_reward >= 2.0:
                validity_score += 0.3
            elif risk_reward >= 1.5:
                validity_score += 0.25
            elif risk_reward >= 1.0:
                validity_score += 0.15
            else:
                validity_score += 0.05
            
            # 检查价格区间合理性
            price_range_pct = (risk_params["take_profit"] - risk_params["stop_loss"]) / current_price
            if 0.05 <= price_range_pct <= 0.15:  # 5%-15%总波动范围
                validity_score += 0.25
            elif 0.03 <= price_range_pct <= 0.20:
                validity_score += 0.2
            else:
                validity_score += 0.1
            
            # 检查风险百分比
            risk_pct = risk_params.get("risk_percentage", 0)
            if risk_pct <= 0.03:  # 不超过3%
                validity_score += 0.15
            elif risk_pct <= 0.05:
                validity_score += 0.1
            else:
                validity_score += 0.05
            
            return min(validity_score, 1.0)
            
        except Exception as e:
            logger.warning(f"区间验证失败: {str(e)}")
            return 0.5
    
    def _calculate_final_confidence(self, interval_validity: float, atr: float, var: float, iv: float) -> float:
        """计算最终可信度"""
        try:
            # 基础可信度来自区间有效性
            base_confidence = interval_validity
            
            # 数据质量调整
            data_quality = 0.0
            
            # ATR数据质量
            if atr > 0:
                data_quality += 0.3
            
            # VaR数据质量
            if 0.005 <= var <= 0.1:  # 合理的VaR范围
                data_quality += 0.3
            elif var > 0:
                data_quality += 0.2
            
            # IV数据质量
            if 0.1 <= iv <= 0.8:  # 合理的IV范围
                data_quality += 0.4
            elif iv > 0:
                data_quality += 0.2
            
            # 综合可信度
            final_confidence = base_confidence * 0.7 + data_quality * 0.3
            
            return min(max(final_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"可信度计算失败: {str(e)}")
            return 0.5
    
    def _create_error_result(self, error_msg: str, execution_time: float) -> ModuleResult:
        """创建错误结果"""
        return ModuleResult(
            module_name=self.name,
            confidence=0.0,
            data={"error": error_msg, "is_valid": False},
            execution_time=execution_time,
            status="error"
        ) 