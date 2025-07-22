"""
模块G: TerraFilter | 环境过滤器
功能: 判定市场总体趋势与板块动能
用途: 决定是否顺勢交易
数据依据: 20日指数趋势一致性 + ETF趋势
可信度G: 同步指数比例
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
from utils import ModuleResult, TimeTracker, logger


class TerraFilterEngine:
    """TerraFilter环境过滤器"""
    
    def __init__(self):
        self.name = "TerraFilter"
        self.module_id = "G"
        
        # 主要市场指数
        self.market_indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ 100", 
            "DIA": "Dow Jones",
            "IWM": "Russell 2000"
        }
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        分析市场环境和板块趋势
        
        Args:
            symbol: 股票代码
            data: 输入数据
            
        Returns:
            ModuleResult: 包含环境分析结果
        """
        timer = TimeTracker().start()
        
        try:
            # 获取市场指数趋势
            market_trends = await self._analyze_market_trends()
            
            # 获取股票所属板块
            sector_info = await self._get_sector_info(symbol)
            
            # 分析板块ETF趋势
            sector_trend = await self._analyze_sector_trend(sector_info)
            
            # 计算市场同步性
            market_sync = self._calculate_market_synchronization(market_trends)
            
            # 计算整体环境得分
            environment_score = self._calculate_environment_score(
                market_trends, sector_trend, market_sync
            )
            
            # 生成交易建议
            trading_recommendation = self._generate_trading_recommendation(
                environment_score, market_trends, sector_trend
            )
            
            execution_time = timer.stop()
            
            result_data = {
                "environment_score": round(environment_score, 4),
                "market_sync_ratio": round(market_sync.get("sync_ratio", 0), 4),
                "sector_trend": sector_trend.get("direction", "neutral"),
                "sector_strength": round(sector_trend.get("strength", 0), 4),
                "market_trends": market_trends,
                "trading_recommendation": trading_recommendation,
                "favorable_environment": environment_score > 0.6,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"TerraFilter分析完成: {symbol} - 环境得分: {environment_score:.2%}, 板块: {sector_trend.get('direction', 'neutral')}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=environment_score,
                data=result_data,
                execution_time=execution_time,
                status="success"
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"TerraFilter分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _analyze_market_trends(self) -> Dict[str, Dict[str, Any]]:
        """分析主要市场指数趋势"""
        trends = {}
        
        try:
            for symbol, name in self.market_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="30d")
                    
                    if not hist.empty:
                        # 计算20日均线趋势
                        ma_20 = hist['Close'].rolling(20).mean()
                        current_price = hist['Close'].iloc[-1]
                        ma_20_current = ma_20.iloc[-1]
                        
                        # 趋势方向
                        above_ma20 = current_price > ma_20_current
                        
                        # 均线斜率
                        ma_slope = (ma_20.iloc[-1] - ma_20.iloc[-5]) / ma_20.iloc[-5] if len(ma_20) >= 5 else 0
                        
                        # 20日涨跌幅
                        price_change_20d = (current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                        
                        trends[symbol] = {
                            "name": name,
                            "above_ma20": above_ma20,
                            "ma_slope": ma_slope,
                            "price_change_20d": price_change_20d,
                            "trend_direction": "bullish" if above_ma20 and ma_slope > 0 else "bearish" if not above_ma20 and ma_slope < 0 else "neutral"
                        }
                        
                except Exception as e:
                    logger.warning(f"获取{symbol}数据失败: {str(e)}")
                    trends[symbol] = {"name": name, "trend_direction": "unknown"}
            
            return trends
            
        except Exception as e:
            logger.warning(f"市场趋势分析失败: {str(e)}")
            return {}
    
    async def _get_sector_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票板块信息"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # 主要板块ETF映射
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV', 
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Consumer Defensive': 'XLP',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                'Basic Materials': 'XLB',
                'Communication Services': 'XLC'
            }
            
            sector_etf = sector_etfs.get(sector, 'SPY')  # 默认使用SPY
            
            return {
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf
            }
            
        except Exception as e:
            logger.warning(f"获取板块信息失败 {symbol}: {str(e)}")
            return {"sector": "Unknown", "industry": "Unknown", "sector_etf": "SPY"}
    
    async def _analyze_sector_trend(self, sector_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析板块ETF趋势"""
        try:
            sector_etf = sector_info.get("sector_etf", "SPY")
            
            ticker = yf.Ticker(sector_etf)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return {"direction": "neutral", "strength": 0}
            
            # 计算趋势指标
            current_price = hist['Close'].iloc[-1]
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            
            # 20日涨跌幅
            price_change_20d = (current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            
            # 相对强度 (vs SPY)
            spy_ticker = yf.Ticker("SPY")
            spy_hist = spy_ticker.history(period="30d")
            
            relative_strength = 0
            if not spy_hist.empty:
                spy_change = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) / spy_hist['Close'].iloc[0]
                relative_strength = price_change_20d - spy_change
            
            # 判断趋势方向和强度
            if price_change_20d > 0.02 and current_price > ma_20:
                direction = "bullish"
                strength = min(abs(price_change_20d) * 2, 1.0)
            elif price_change_20d < -0.02 and current_price < ma_20:
                direction = "bearish"
                strength = min(abs(price_change_20d) * 2, 1.0)
            else:
                direction = "neutral"
                strength = 0.5
            
            return {
                "direction": direction,
                "strength": strength,
                "price_change_20d": price_change_20d,
                "relative_strength": relative_strength,
                "above_ma20": current_price > ma_20,
                "sector_etf": sector_etf
            }
            
        except Exception as e:
            logger.warning(f"板块趋势分析失败: {str(e)}")
            return {"direction": "neutral", "strength": 0}
    
    def _calculate_market_synchronization(self, market_trends: Dict[str, Any]) -> Dict[str, Any]:
        """计算市场同步性"""
        try:
            if not market_trends:
                return {"sync_ratio": 0, "bullish_count": 0, "total_count": 0}
            
            bullish_count = 0
            bearish_count = 0
            total_count = 0
            
            for symbol, trend in market_trends.items():
                if trend.get("trend_direction") != "unknown":
                    total_count += 1
                    if trend.get("trend_direction") == "bullish":
                        bullish_count += 1
                    elif trend.get("trend_direction") == "bearish":
                        bearish_count += 1
            
            sync_ratio = max(bullish_count, bearish_count) / total_count if total_count > 0 else 0
            
            return {
                "sync_ratio": sync_ratio,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "total_count": total_count,
                "market_consensus": "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "mixed"
            }
            
        except Exception as e:
            logger.warning(f"市场同步性计算失败: {str(e)}")
            return {"sync_ratio": 0, "bullish_count": 0, "total_count": 0}
    
    def _calculate_environment_score(
        self, 
        market_trends: Dict[str, Any],
        sector_trend: Dict[str, Any],
        market_sync: Dict[str, Any]
    ) -> float:
        """计算环境得分"""
        try:
            score = 0.0
            
            # 市场同步性得分 (40%)
            sync_ratio = market_sync.get("sync_ratio", 0)
            score += sync_ratio * 0.4
            
            # 板块趋势得分 (35%)
            sector_direction = sector_trend.get("direction", "neutral")
            sector_strength = sector_trend.get("strength", 0)
            
            if sector_direction == "bullish":
                score += sector_strength * 0.35
            elif sector_direction == "neutral":
                score += 0.5 * 0.35
            else:  # bearish
                score += (1 - sector_strength) * 0.35
            
            # 整体市场方向得分 (25%)
            market_consensus = market_sync.get("market_consensus", "mixed")
            if market_consensus == "bullish":
                score += 0.25
            elif market_consensus == "mixed":
                score += 0.15
            else:  # bearish
                score += 0.05
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"环境得分计算失败: {str(e)}")
            return 0.5
    
    def _generate_trading_recommendation(
        self,
        environment_score: float,
        market_trends: Dict[str, Any],
        sector_trend: Dict[str, Any]
    ) -> str:
        """生成交易建议"""
        try:
            if environment_score >= 0.8:
                return "highly_favorable"
            elif environment_score >= 0.6:
                return "favorable"
            elif environment_score >= 0.4:
                return "neutral"
            elif environment_score >= 0.2:
                return "unfavorable"
            else:
                return "highly_unfavorable"
                
        except Exception as e:
            logger.warning(f"交易建议生成失败: {str(e)}")
            return "neutral" 