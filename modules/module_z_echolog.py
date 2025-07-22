"""
模块Z: EchoLog | 模拟与回测器
功能: 记录策略逐筆回测與實盤績效
用途: 為模組Y提供歷史數據
数据依据: 30日回测 + 實盤同步
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from utils import ModuleResult, TimeTracker, logger
from config import BACKTEST_CONFIG, MIN_PROFIT_RATIO, RISK_CONFIG


class EchoLogEngine:
    """EchoLog模拟与回测器"""
    
    def __init__(self):
        self.name = "EchoLog"
        self.module_id = "Z"
        self.min_win_rate = BACKTEST_CONFIG  # 65%
        self.min_profit_ratio = MIN_PROFIT_RATIO  # 1.0
        self.max_drawdown = RISK_CONFIG  # 5%
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        执行策略回测和绩效分析
        
        Args:
            symbol: 股票代码
            data: 输入数据包含策略参数
            
        Returns:
            ModuleResult: 包含回测结果和绩效统计
        """
        timer = TimeTracker().start()
        
        try:
            # 获取历史数据用于回测
            historical_data = await self._get_historical_data(symbol)
            
            if historical_data.empty:
                logger.warning(f"EchoLog: {symbol} 历史数据不足")
                return self._create_insufficient_data_result(timer.stop())
            
            # 执行30日回测
            backtest_30d = await self._run_backtest(symbol, historical_data, days=30)
            
            # 执行6个月回测
            backtest_6m = await self._run_backtest(symbol, historical_data, days=180)
            
            # 执行2年回测 (如果数据足够)
            backtest_2y = await self._run_backtest(symbol, historical_data, days=730)
            
            # 综合回测结果
            combined_results = self._combine_backtest_results(
                backtest_30d, backtest_6m, backtest_2y
            )
            
            # 验证回测质量
            quality_check = self._validate_backtest_quality(combined_results)
            
            # 计算最终可信度
            final_confidence = self._calculate_backtest_confidence(
                combined_results, quality_check
            )
            
            execution_time = timer.stop()
            
            result_data = {
                "hit_rate": combined_results.get("hit_rate", 0),
                "profit_ratio": combined_results.get("profit_ratio", 0),
                "max_drawdown": combined_results.get("max_drawdown", 0),
                "total_trades": combined_results.get("total_trades", 0),
                "winning_trades": combined_results.get("winning_trades", 0),
                "backtest_30d": backtest_30d,
                "backtest_6m": backtest_6m,
                "backtest_2y": backtest_2y,
                "quality_passed": quality_check["passed"],
                "quality_issues": quality_check.get("issues", []),
                "final_confidence": round(final_confidence, 4),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"EchoLog回测完成: {symbol} - 命中率: {combined_results.get('hit_rate', 0):.2%}, 盈亏比: {combined_results.get('profit_ratio', 0):.2f}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence,
                data=result_data,
                execution_time=execution_time,
                status="success"
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"EchoLog回测失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_historical_data(self, symbol: str, max_days: int = 800) -> pd.DataFrame:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")  # 获取2年数据
            
            if hist.empty:
                return pd.DataFrame()
            
            # 计算技术指标
            hist['Returns'] = hist['Close'].pct_change()
            hist['MA_20'] = hist['Close'].rolling(20).mean()
            hist['MA_50'] = hist['Close'].rolling(50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['Volume_MA'] = hist['Volume'].rolling(20).mean()
            hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA']
            
            return hist.dropna()
            
        except Exception as e:
            logger.warning(f"获取历史数据失败 {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _run_backtest(self, symbol: str, data: pd.DataFrame, days: int) -> Dict[str, Any]:
        """运行回测"""
        try:
            if data.empty or len(data) < days:
                return {"period": f"{days}d", "trades": 0, "hit_rate": 0, "profit_ratio": 0}
            
            # 使用最近N天数据
            backtest_data = data.tail(days).copy()
            
            # 生成交易信号 (简化策略)
            signals = self._generate_trading_signals(backtest_data)
            
            # 模拟交易执行
            trades = self._simulate_trades(backtest_data, signals)
            
            # 计算绩效指标
            performance = self._calculate_performance(trades)
            
            return {
                "period": f"{days}d",
                "trades": len(trades),
                "hit_rate": performance.get("hit_rate", 0),
                "profit_ratio": performance.get("profit_ratio", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "total_return": performance.get("total_return", 0),
                "winning_trades": performance.get("winning_trades", 0),
                "losing_trades": performance.get("losing_trades", 0),
                "avg_win": performance.get("avg_win", 0),
                "avg_loss": performance.get("avg_loss", 0),
                "trade_details": trades[:10]  # 保留前10笔交易详情
            }
            
        except Exception as e:
            logger.warning(f"回测执行失败 {days}天: {str(e)}")
            return {"period": f"{days}d", "trades": 0, "hit_rate": 0, "profit_ratio": 0}
    
    def _generate_trading_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成交易信号 (简化策略)"""
        try:
            signals = []
            
            for i in range(20, len(data) - 5):  # 保留前20天和后5天
                current_data = data.iloc[i]
                
                # 简化的买入信号
                # 1. 价格突破20日均线
                # 2. RSI在30-70之间
                # 3. 成交量放大
                
                price_above_ma20 = current_data['Close'] > current_data['MA_20']
                rsi_range = 30 <= current_data['RSI'] <= 70
                volume_surge = current_data['Volume_Ratio'] > 1.5
                
                if price_above_ma20 and rsi_range and volume_surge:
                    # 计算止损止盈位
                    entry_price = current_data['Close']
                    atr = self._calculate_atr_at_point(data, i)
                    
                    stop_loss = entry_price - (1.5 * atr)
                    take_profit = entry_price + (3 * atr)  # 1:2 风险收益比
                    
                    signals.append({
                        "entry_date": current_data.name,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "signal_strength": self._calculate_signal_strength(data, i)
                    })
            
            return signals
            
        except Exception as e:
            logger.warning(f"信号生成失败: {str(e)}")
            return []
    
    def _simulate_trades(self, data: pd.DataFrame, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """模拟交易执行"""
        trades = []
        
        try:
            for signal in signals:
                entry_date = signal["entry_date"]
                entry_price = signal["entry_price"]
                stop_loss = signal["stop_loss"]
                take_profit = signal["take_profit"]
                
                # 寻找出场点
                exit_result = self._find_exit_point(
                    data, entry_date, entry_price, stop_loss, take_profit
                )
                
                if exit_result:
                    trade = {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_result["exit_date"],
                        "exit_price": exit_result["exit_price"],
                        "exit_reason": exit_result["reason"],
                        "return": (exit_result["exit_price"] - entry_price) / entry_price,
                        "holding_days": exit_result["holding_days"],
                        "success": exit_result["exit_price"] > entry_price
                    }
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.warning(f"交易模拟失败: {str(e)}")
            return []
    
    def _find_exit_point(
        self, 
        data: pd.DataFrame, 
        entry_date: pd.Timestamp, 
        entry_price: float,
        stop_loss: float, 
        take_profit: float,
        max_holding_days: int = 10
    ) -> Dict[str, Any]:
        """寻找交易出场点"""
        try:
            # 找到入场日期在数据中的位置
            entry_idx = data.index.get_loc(entry_date)
            
            # 检查后续几天的价格
            for i in range(1, min(max_holding_days + 1, len(data) - entry_idx)):
                current_idx = entry_idx + i
                if current_idx >= len(data):
                    break
                    
                current_row = data.iloc[current_idx]
                
                # 检查止损
                if current_row['Low'] <= stop_loss:
                    return {
                        "exit_date": current_row.name,
                        "exit_price": stop_loss,
                        "reason": "stop_loss",
                        "holding_days": i
                    }
                
                # 检查止盈
                if current_row['High'] >= take_profit:
                    return {
                        "exit_date": current_row.name,
                        "exit_price": take_profit,
                        "reason": "take_profit",
                        "holding_days": i
                    }
            
            # 如果没有触发止损止盈，按最大持有期出场
            final_idx = min(entry_idx + max_holding_days, len(data) - 1)
            final_row = data.iloc[final_idx]
            
            return {
                "exit_date": final_row.name,
                "exit_price": final_row['Close'],
                "reason": "time_exit",
                "holding_days": max_holding_days
            }
            
        except Exception as e:
            logger.warning(f"出场点查找失败: {str(e)}")
            return None
    
    def _calculate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算交易绩效"""
        try:
            if not trades:
                return {"hit_rate": 0, "profit_ratio": 0, "max_drawdown": 0}
            
            # 计算基本统计
            returns = [trade["return"] for trade in trades]
            winning_trades = [r for r in returns if r > 0]
            losing_trades = [r for r in returns if r <= 0]
            
            hit_rate = len(winning_trades) / len(trades)
            
            # 计算盈亏比
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
            profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = np.cumprod([1 + r for r in returns])
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # 总收益
            total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "profit_ratio": profit_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_trades": len(trades)
            }
            
        except Exception as e:
            logger.warning(f"绩效计算失败: {str(e)}")
            return {"hit_rate": 0, "profit_ratio": 0, "max_drawdown": 0}
    
    def _combine_backtest_results(
        self, 
        result_30d: Dict[str, Any],
        result_6m: Dict[str, Any],
        result_2y: Dict[str, Any]
    ) -> Dict[str, Any]:
        """综合多期回测结果"""
        try:
            # 加权综合结果 (最近30天权重最高)
            weights = {"30d": 0.5, "6m": 0.3, "2y": 0.2}
            
            results = [result_30d, result_6m, result_2y]
            periods = ["30d", "6m", "2y"]
            
            combined_hit_rate = 0
            combined_profit_ratio = 0
            combined_max_drawdown = 0
            total_trades = 0
            total_winning = 0
            
            for result, period in zip(results, periods):
                if result.get("trades", 0) > 0:
                    weight = weights[period]
                    combined_hit_rate += result.get("hit_rate", 0) * weight
                    combined_profit_ratio += result.get("profit_ratio", 0) * weight
                    combined_max_drawdown += result.get("max_drawdown", 0) * weight
                    total_trades += result.get("trades", 0)
                    total_winning += result.get("winning_trades", 0)
            
            return {
                "hit_rate": combined_hit_rate,
                "profit_ratio": combined_profit_ratio,
                "max_drawdown": combined_max_drawdown,
                "total_trades": total_trades,
                "winning_trades": total_winning
            }
            
        except Exception as e:
            logger.warning(f"结果综合失败: {str(e)}")
            return {"hit_rate": 0, "profit_ratio": 0, "max_drawdown": 0}
    
    def _validate_backtest_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证回测质量"""
        try:
            issues = []
            
            hit_rate = results.get("hit_rate", 0)
            profit_ratio = results.get("profit_ratio", 0)
            max_drawdown = results.get("max_drawdown", 0)
            total_trades = results.get("total_trades", 0)
            
            # 检查命中率
            if hit_rate < self.min_win_rate:
                issues.append(f"命中率{hit_rate:.1%} < {self.min_win_rate:.1%}")
            
            # 检查盈亏比
            if profit_ratio < self.min_profit_ratio:
                issues.append(f"盈亏比{profit_ratio:.2f} < {self.min_profit_ratio:.2f}")
            
            # 检查最大回撤
            if max_drawdown > self.max_drawdown:
                issues.append(f"最大回撤{max_drawdown:.1%} > {self.max_drawdown:.1%}")
            
            # 检查交易数量
            if total_trades < 10:
                issues.append(f"交易数量过少({total_trades})")
            
            passed = len(issues) == 0
            
            return {"passed": passed, "issues": issues}
            
        except Exception as e:
            logger.warning(f"质量验证失败: {str(e)}")
            return {"passed": False, "issues": ["质量验证异常"]}
    
    def _calculate_backtest_confidence(
        self, 
        results: Dict[str, Any], 
        quality_check: Dict[str, Any]
    ) -> float:
        """计算回测可信度"""
        try:
            if not quality_check.get("passed", False):
                return 0.3  # 质量不过关时低可信度
            
            hit_rate = results.get("hit_rate", 0)
            profit_ratio = results.get("profit_ratio", 0)
            total_trades = results.get("total_trades", 0)
            
            # 基础可信度
            base_confidence = min(hit_rate, 1.0) * 0.6
            
            # 盈亏比加成
            profit_bonus = min(profit_ratio / 2.0, 0.3)
            base_confidence += profit_bonus
            
            # 交易数量加成
            trade_bonus = min(total_trades / 50.0, 0.1)
            base_confidence += trade_bonus
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"可信度计算失败: {str(e)}")
            return 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr_at_point(self, data: pd.DataFrame, index: int, period: int = 14) -> float:
        """计算指定点的ATR"""
        try:
            if index < period:
                return data.iloc[index]['Close'] * 0.02  # 默认2%
            
            subset = data.iloc[index-period:index]
            high_low = subset['High'] - subset['Low']
            high_close = abs(subset['High'] - subset['Close'].shift(1))
            low_close = abs(subset['Low'] - subset['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.mean()
            
            return atr if not pd.isna(atr) else data.iloc[index]['Close'] * 0.02
            
        except Exception as e:
            logger.warning(f"ATR计算失败: {str(e)}")
            return data.iloc[index]['Close'] * 0.02
    
    def _calculate_signal_strength(self, data: pd.DataFrame, index: int) -> float:
        """计算信号强度"""
        try:
            current = data.iloc[index]
            
            # 多个因子综合评分
            strength = 0.0
            
            # 价格位置
            if current['Close'] > current['MA_20']:
                strength += 0.3
            if current['Close'] > current['MA_50']:
                strength += 0.2
            
            # RSI
            rsi = current['RSI']
            if 40 <= rsi <= 60:
                strength += 0.3
            elif 30 <= rsi <= 70:
                strength += 0.2
            
            # 成交量
            if current['Volume_Ratio'] > 2.0:
                strength += 0.2
            elif current['Volume_Ratio'] > 1.5:
                strength += 0.1
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.warning(f"信号强度计算失败: {str(e)}")
            return 0.5
    
    def _create_insufficient_data_result(self, execution_time: float) -> ModuleResult:
        """创建数据不足结果"""
        return ModuleResult(
            module_name=self.name,
            confidence=0.0,
            data={
                "error": "历史数据不足",
                "hit_rate": 0,
                "profit_ratio": 0,
                "max_drawdown": 0
            },
            execution_time=execution_time,
            status="error"
        )
    
    async def run_backtest(self, symbol: str, params: dict = None) -> dict:
        """运行回测分析"""
        try:
            from config import DEMO_MODE, IS_DEMO_POLYGON
            from demo_utils import get_demo_backtest_results, should_use_demo_mode
            
            # 演示模式
            if should_use_demo_mode():
                logger.info(f"💻 EchoLog演示模式回测: {symbol}")
                return get_demo_backtest_results(symbol)
            
            # 实际回测逻辑
            start_time = time.time()
            result = await self.analyze(symbol)
            
            backtest_data = {
                "symbol": symbol,
                "period": "60天",
                "confidence": result.confidence,
                "status": result.status,
                "execution_time": time.time() - start_time
            }
            
            if result.status == "success":
                backtest_data.update(result.data)
            
            return backtest_data
            
        except Exception as e:
            logger.error(f"回测失败: {str(e)}")
            return {"error": f"回测失败: {str(e)}", "symbol": symbol} 