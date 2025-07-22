"""
模块X: Cerberus | 异常防护模块
功能: 偵測波動率過高、市場逆轉等風險事件
用途: 中斷信號或降級可信度
数据依据: VIX、跳空、重大宏觀事件
特性: 異常觸發將自動調降全局命中率5%並標註紅色警示
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
from ..utils import ModuleResult, TimeTracker, logger


class CerberusEngine:
    """Cerberus异常防护模块"""
    
    def __init__(self):
        self.name = "Cerberus"
        self.module_id = "X"
        
        # 异常检测阈值
        self.thresholds = {
            "vix_high": 30,      # VIX高恐慌阈值
            "vix_extreme": 40,   # VIX极度恐慌阈值
            "gap_large": 0.05,   # 5%以上跳空
            "gap_extreme": 0.10, # 10%以上跳空
            "volume_spike": 5.0, # 成交量异常倍数
            "volatility_high": 0.06  # 日内波动率6%以上
        }
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        检测市场异常和风险事件
        
        Args:
            symbol: 股票代码
            data: 输入数据
            
        Returns:
            ModuleResult: 包含异常检测结果
        """
        timer = TimeTracker().start()
        
        try:
            # 检测VIX异常
            vix_anomaly = await self._detect_vix_anomaly()
            
            # 检测价格跳空
            gap_anomaly = await self._detect_gap_anomaly(symbol, data)
            
            # 检测成交量异常
            volume_anomaly = await self._detect_volume_anomaly(symbol, data)
            
            # 检测波动率异常
            volatility_anomaly = await self._detect_volatility_anomaly(symbol)
            
            # 检测宏观事件
            macro_events = await self._detect_macro_events()
            
            # 综合异常评估
            anomaly_assessment = self._assess_overall_anomaly(
                vix_anomaly, gap_anomaly, volume_anomaly, 
                volatility_anomaly, macro_events
            )
            
            # 确定状态
            is_anomaly_detected = anomaly_assessment["risk_level"] in ["high", "extreme"]
            status = "anomaly_detected" if is_anomaly_detected else "normal"
            
            execution_time = timer.stop()
            
            result_data = {
                "anomaly_detected": is_anomaly_detected,
                "risk_level": anomaly_assessment["risk_level"],
                "confidence_penalty": anomaly_assessment["confidence_penalty"],
                "vix_anomaly": vix_anomaly,
                "gap_anomaly": gap_anomaly,
                "volume_anomaly": volume_anomaly,
                "volatility_anomaly": volatility_anomaly,
                "macro_events": macro_events,
                "warning_message": anomaly_assessment.get("warning", ""),
                "recommended_action": anomaly_assessment.get("action", "continue"),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if is_anomaly_detected:
                logger.warning(f"🚨 Cerberus异常警报: {symbol} - 风险等级: {anomaly_assessment['risk_level']}")
            else:
                logger.info(f"Cerberus检测正常: {symbol} - 未发现显著异常")
            
            return ModuleResult(
                module_name=self.name,
                confidence=1.0 - anomaly_assessment["confidence_penalty"],
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Cerberus检测失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _detect_vix_anomaly(self) -> Dict[str, Any]:
        """检测VIX恐慌指数异常"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="5d")
            
            if vix_hist.empty:
                return {"detected": False, "current_vix": 0, "severity": "unknown"}
            
            current_vix = vix_hist['Close'].iloc[-1]
            
            # 判断VIX水平
            if current_vix >= self.thresholds["vix_extreme"]:
                severity = "extreme"
                detected = True
            elif current_vix >= self.thresholds["vix_high"]:
                severity = "high"
                detected = True
            else:
                severity = "normal"
                detected = False
            
            return {
                "detected": detected,
                "current_vix": round(current_vix, 2),
                "severity": severity,
                "threshold_high": self.thresholds["vix_high"],
                "threshold_extreme": self.thresholds["vix_extreme"]
            }
            
        except Exception as e:
            logger.warning(f"VIX检测失败: {str(e)}")
            return {"detected": False, "current_vix": 0, "severity": "unknown"}
    
    async def _detect_gap_anomaly(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """检测价格跳空异常"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if len(hist) < 2:
                return {"detected": False, "gap_percent": 0, "severity": "unknown"}
            
            # 计算今日开盘与昨日收盘的跳空
            today_open = hist['Open'].iloc[-1]
            yesterday_close = hist['Close'].iloc[-2]
            
            gap_percent = abs(today_open - yesterday_close) / yesterday_close
            
            # 判断跳空严重程度
            if gap_percent >= self.thresholds["gap_extreme"]:
                severity = "extreme"
                detected = True
            elif gap_percent >= self.thresholds["gap_large"]:
                severity = "high"
                detected = True
            else:
                severity = "normal"
                detected = False
            
            gap_direction = "up" if today_open > yesterday_close else "down"
            
            return {
                "detected": detected,
                "gap_percent": round(gap_percent, 4),
                "gap_direction": gap_direction,
                "severity": severity,
                "today_open": round(today_open, 2),
                "yesterday_close": round(yesterday_close, 2)
            }
            
        except Exception as e:
            logger.warning(f"跳空检测失败 {symbol}: {str(e)}")
            return {"detected": False, "gap_percent": 0, "severity": "unknown"}
    
    async def _detect_volume_anomaly(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """检测成交量异常"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty or len(hist) < 10:
                return {"detected": False, "volume_ratio": 1, "severity": "unknown"}
            
            # 计算平均成交量
            avg_volume = hist['Volume'].iloc[:-1].mean()  # 排除今日
            current_volume = data.get("volume", hist['Volume'].iloc[-1])
            
            if avg_volume == 0:
                return {"detected": False, "volume_ratio": 1, "severity": "unknown"}
            
            volume_ratio = current_volume / avg_volume
            
            # 判断成交量异常
            if volume_ratio >= self.thresholds["volume_spike"] * 2:  # 10倍以上
                severity = "extreme"
                detected = True
            elif volume_ratio >= self.thresholds["volume_spike"]:  # 5倍以上
                severity = "high"
                detected = True
            else:
                severity = "normal"
                detected = False
            
            return {
                "detected": detected,
                "volume_ratio": round(volume_ratio, 2),
                "current_volume": current_volume,
                "avg_volume": round(avg_volume, 0),
                "severity": severity
            }
            
        except Exception as e:
            logger.warning(f"成交量异常检测失败 {symbol}: {str(e)}")
            return {"detected": False, "volume_ratio": 1, "severity": "unknown"}
    
    async def _detect_volatility_anomaly(self, symbol: str) -> Dict[str, Any]:
        """检测波动率异常"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="5m")
            
            if hist.empty or len(hist) < 10:
                return {"detected": False, "intraday_volatility": 0, "severity": "unknown"}
            
            # 计算日内波动率
            high_price = hist['High'].max()
            low_price = hist['Low'].min()
            open_price = hist['Open'].iloc[0]
            
            intraday_volatility = (high_price - low_price) / open_price
            
            # 判断波动率异常
            if intraday_volatility >= self.thresholds["volatility_high"] * 2:  # 12%以上
                severity = "extreme"
                detected = True
            elif intraday_volatility >= self.thresholds["volatility_high"]:  # 6%以上
                severity = "high"
                detected = True
            else:
                severity = "normal"
                detected = False
            
            return {
                "detected": detected,
                "intraday_volatility": round(intraday_volatility, 4),
                "high_price": round(high_price, 2),
                "low_price": round(low_price, 2),
                "volatility_threshold": self.thresholds["volatility_high"],
                "severity": severity
            }
            
        except Exception as e:
            logger.warning(f"波动率异常检测失败 {symbol}: {str(e)}")
            return {"detected": False, "intraday_volatility": 0, "severity": "unknown"}
    
    async def _detect_macro_events(self) -> Dict[str, Any]:
        """检测宏观事件 (简化实现)"""
        try:
            # 简化的宏观事件检测
            # 在实际应用中，这里应该连接新闻API或经济日历API
            
            # 检查主要市场是否有大幅波动
            major_indices = ["^GSPC", "^IXIC", "^DJI"]
            macro_stress = False
            
            for index in major_indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        daily_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                        
                        if abs(daily_change) > 0.03:  # 3%以上变动
                            macro_stress = True
                            break
                            
                except Exception:
                    continue
            
            return {
                "macro_stress_detected": macro_stress,
                "stress_level": "high" if macro_stress else "normal",
                "description": "Major market volatility detected" if macro_stress else "Normal market conditions"
            }
            
        except Exception as e:
            logger.warning(f"宏观事件检测失败: {str(e)}")
            return {"macro_stress_detected": False, "stress_level": "unknown"}
    
    def _assess_overall_anomaly(
        self,
        vix_anomaly: Dict[str, Any],
        gap_anomaly: Dict[str, Any],
        volume_anomaly: Dict[str, Any],
        volatility_anomaly: Dict[str, Any],
        macro_events: Dict[str, Any]
    ) -> Dict[str, Any]:
        """综合异常评估"""
        try:
            risk_score = 0
            warnings = []
            
            # VIX异常评分
            if vix_anomaly.get("detected", False):
                if vix_anomaly.get("severity") == "extreme":
                    risk_score += 3
                    warnings.append("极度恐慌(VIX)")
                elif vix_anomaly.get("severity") == "high":
                    risk_score += 2
                    warnings.append("高恐慌(VIX)")
            
            # 跳空异常评分
            if gap_anomaly.get("detected", False):
                if gap_anomaly.get("severity") == "extreme":
                    risk_score += 2
                    warnings.append("极端跳空")
                elif gap_anomaly.get("severity") == "high":
                    risk_score += 1
                    warnings.append("大幅跳空")
            
            # 成交量异常评分
            if volume_anomaly.get("detected", False):
                if volume_anomaly.get("severity") == "extreme":
                    risk_score += 2
                    warnings.append("极端放量")
                elif volume_anomaly.get("severity") == "high":
                    risk_score += 1
                    warnings.append("异常放量")
            
            # 波动率异常评分
            if volatility_anomaly.get("detected", False):
                if volatility_anomaly.get("severity") == "extreme":
                    risk_score += 2
                    warnings.append("极端波动")
                elif volatility_anomaly.get("severity") == "high":
                    risk_score += 1
                    warnings.append("高波动")
            
            # 宏观事件评分
            if macro_events.get("macro_stress_detected", False):
                risk_score += 1
                warnings.append("宏观压力")
            
            # 确定风险等级和可信度惩罚
            if risk_score >= 6:
                risk_level = "extreme"
                confidence_penalty = 0.15  # 15%惩罚
                action = "suspend_trading"
            elif risk_score >= 4:
                risk_level = "high"
                confidence_penalty = 0.10  # 10%惩罚
                action = "reduce_position"
            elif risk_score >= 2:
                risk_level = "medium"
                confidence_penalty = 0.05  # 5%惩罚 (符合要求)
                action = "proceed_with_caution"
            else:
                risk_level = "low"
                confidence_penalty = 0.0
                action = "continue"
            
            warning_message = "; ".join(warnings) if warnings else ""
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence_penalty": confidence_penalty,
                "warning": warning_message,
                "action": action
            }
            
        except Exception as e:
            logger.warning(f"综合异常评估失败: {str(e)}")
            return {
                "risk_level": "unknown",
                "confidence_penalty": 0.05,
                "warning": "评估系统异常",
                "action": "proceed_with_caution"
            } 