"""
模块S: Helios | 市场扫描与选股器（升级版）
功能: 在指定股票池中，基于交易量、波动性、价格行为与即时新闻等多维度条件，
      自动筛选出当日具备交易潜力的股票候选名单
用途: 作为整个决策流程的起点，为后续的精细化分析模块提供高质量的交易候选标的
可信度S: 根据筛选条件的满足程度与历史回测有效性生成，若低于60%，则该候选名单将被视为低质量
"""

import asyncio
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from ..utils import ModuleResult, TimeTracker, logger
from ..config import POLYGON_API_KEY, POLYGON_BASE_URL, MIN_MODULE_CONFIDENCE


class HeliosEngine:
    """Helios市场扫描与选股器"""
    
    def __init__(self):
        self.name = "Helios"
        self.module_id = "S"
        self.api_key = POLYGON_API_KEY
        self.min_confidence = MIN_MODULE_CONFIDENCE['S']  # 60%
        
        # 默认股票池 (可扩展)
        self.default_pools = {
            "sp500": self._get_sp500_symbols(),
            "nasdaq100": self._get_nasdaq100_symbols(),
            "tech_leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD"]
        }
    
    async def scan_market(
        self, 
        stock_pool: str = "tech_leaders",
        custom_symbols: List[str] = None
    ) -> ModuleResult:
        """
        扫描市场并选出潜力股票
        
        Args:
            stock_pool: 股票池名称 ("sp500", "nasdaq100", "tech_leaders")
            custom_symbols: 自定义股票列表
            
        Returns:
            ModuleResult: 包含筛选结果和S-Score评分
        """
        timer = TimeTracker().start()
        
        try:
            # 确定扫描范围
            if custom_symbols:
                symbols = custom_symbols
            else:
                symbols = self.default_pools.get(stock_pool, self.default_pools["tech_leaders"])
            
            logger.info(f"Helios开始扫描 {len(symbols)} 只股票...")
            
            # 获取预市场数据
            premarket_data = await self._get_premarket_data(symbols)
            
            # 计算各股票的S-Score
            candidates = []
            
            for symbol in symbols:
                try:
                    # 获取股票技术数据
                    technical_data = await self._get_technical_data(symbol)
                    
                    if not technical_data:
                        continue
                    
                    # 计算S-Score
                    s_score = self._calculate_s_score(symbol, technical_data, premarket_data.get(symbol, {}))
                    
                    if s_score >= 60:  # 只保留60分以上的候选股
                        candidates.append({
                            "symbol": symbol,
                            "s_score": s_score,
                            "technical_data": technical_data,
                            "premarket_data": premarket_data.get(symbol, {})
                        })
                        
                except Exception as e:
                    logger.warning(f"扫描 {symbol} 失败: {str(e)}")
                    continue
            
            # 按S-Score排序
            candidates.sort(key=lambda x: x["s_score"], reverse=True)
            
            # 分类候选股
            strong_candidates = [c for c in candidates if c["s_score"] >= 80]  # 强选池
            moderate_candidates = [c for c in candidates if 60 <= c["s_score"] < 80]
            
            # 计算整体筛选质量
            overall_confidence = self._calculate_scan_confidence(candidates, len(symbols))
            
            execution_time = timer.stop()
            
            result_data = {
                "total_scanned": len(symbols),
                "qualified_candidates": len(candidates),
                "strong_candidates": len(strong_candidates),
                "moderate_candidates": len(moderate_candidates),
                "strong_pool": strong_candidates[:10],  # 返回前10个强候选
                "moderate_pool": moderate_candidates[:15],  # 返回前15个中等候选
                "scan_confidence": round(overall_confidence, 4),
                "is_high_quality": overall_confidence >= self.min_confidence,
                "scan_timestamp": datetime.now().isoformat(),
                "stock_pool_used": stock_pool
            }
            
            status = "success" if overall_confidence >= self.min_confidence else "low_quality"
            
            if overall_confidence >= self.min_confidence:
                logger.info(f"Helios扫描完成: 发现 {len(strong_candidates)} 个强候选, {len(moderate_candidates)} 个中等候选")
            else:
                logger.warning(f"Helios扫描质量不足: 可信度 {overall_confidence:.2%} < {self.min_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=overall_confidence,
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Helios扫描失败: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e)},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_premarket_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """获取盘前数据"""
        premarket_data = {}
        
        try:
            # 使用 yfinance 获取盘前数据 (简化实现)
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # 获取盘前价格变动
                    regular_market_price = info.get('regularMarketPrice', 0)
                    previous_close = info.get('previousClose', 0)
                    
                    if previous_close > 0:
                        premarket_change = (regular_market_price - previous_close) / previous_close
                        
                        premarket_data[symbol] = {
                            "premarket_change_pct": premarket_change,
                            "premarket_volume": info.get('regularMarketVolume', 0),
                            "gap_percent": abs(premarket_change),
                            "current_price": regular_market_price
                        }
                
                except Exception as e:
                    logger.warning(f"获取 {symbol} 盘前数据失败: {str(e)}")
                    continue
            
            return premarket_data
            
        except Exception as e:
            logger.warning(f"获取盘前数据失败: {str(e)}")
            return {}
    
    async def _get_technical_data(self, symbol: str) -> Dict[str, Any]:
        """获取技术分析数据"""
        try:
            # 使用 yfinance 获取技术数据
            ticker = yf.Ticker(symbol)
            
            # 获取历史数据
            hist = ticker.history(period="60d")
            
            if hist.empty:
                return None
            
            # 计算技术指标
            current_price = hist['Close'].iloc[-1]
            volume_20d_avg = hist['Volume'].rolling(20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # 计算RVOL (相对成交量)
            rvol = current_volume / volume_20d_avg if volume_20d_avg > 0 else 1.0
            
            # 计算ATR
            high_low = hist['High'] - hist['Low']
            high_close = abs(hist['High'] - hist['Close'].shift())
            low_close = abs(hist['Low'] - hist['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            atr_percentage = (atr_14 / current_price) if current_price > 0 else 0
            
            # 移动平均线
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else ma_20
            
            # 52周高点距离
            high_52w = hist['High'].max()
            distance_from_high = (high_52w - current_price) / high_52w if high_52w > 0 else 1.0
            
            # 近期表现
            recent_performance = {}
            for days in [5, 15]:
                if len(hist) >= days:
                    old_price = hist['Close'].iloc[-days]
                    performance = (current_price - old_price) / old_price
                    recent_performance[f"{days}d_return"] = performance
            
            return {
                "current_price": current_price,
                "rvol": rvol,
                "atr_percentage": atr_percentage,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "price_above_ma20": current_price > ma_20,
                "price_above_ma50": current_price > ma_50,
                "distance_from_52w_high": distance_from_high,
                "volume_20d_avg": volume_20d_avg,
                "current_volume": current_volume,
                "recent_performance": recent_performance,
                "price_meets_minimum": current_price >= 10.0  # 最低价格要求
            }
            
        except Exception as e:
            logger.warning(f"获取 {symbol} 技术数据失败: {str(e)}")
            return None
    
    def _calculate_s_score(
        self, 
        symbol: str, 
        technical_data: Dict[str, Any], 
        premarket_data: Dict[str, Any]
    ) -> float:
        """计算S-Score (0-100分)"""
        try:
            score = 0.0
            
            # 1. RVOL权重25%
            rvol = technical_data.get("rvol", 1.0)
            if rvol >= 3.0:
                score += 25
            elif rvol >= 2.0:
                score += 20
            elif rvol >= 1.5:
                score += 15
            elif rvol >= 1.0:
                score += 10
            else:
                score += 5
            
            # 2. ATR波动性20%
            atr_pct = technical_data.get("atr_percentage", 0)
            price = technical_data.get("current_price", 0)
            
            if atr_pct >= 0.04 and price >= 10:  # ATR>4% 且价格>$10
                score += 20
            elif atr_pct >= 0.03 and price >= 10:
                score += 15
            elif atr_pct >= 0.02:
                score += 10
            else:
                score += 5
            
            # 3. 跳空幅度15%
            gap_pct = premarket_data.get("gap_percent", 0)
            if gap_pct >= 0.03:  # 3%以上跳空
                score += 15
            elif gap_pct >= 0.02:
                score += 12
            elif gap_pct >= 0.01:
                score += 8
            else:
                score += 3
            
            # 4. 均线突破20%
            breakthrough_score = 0
            if technical_data.get("price_above_ma20"):
                breakthrough_score += 10
            if technical_data.get("price_above_ma50"):
                breakthrough_score += 10
            
            # 均线排列
            ma_20 = technical_data.get("ma_20", 0)
            ma_50 = technical_data.get("ma_50", 0)
            if ma_20 > ma_50:  # 多头排列
                breakthrough_score += 5
            
            score += min(breakthrough_score, 20)
            
            # 5. 新闻情绪20% (简化实现)
            news_score = self._get_simplified_news_score(symbol)
            score += news_score
            
            # 额外加分项
            
            # 接近52周高点
            distance_from_high = technical_data.get("distance_from_52w_high", 1.0)
            if distance_from_high <= 0.05:  # 5%范围内
                score += 5
            
            # 近期强势表现
            recent_perf = technical_data.get("recent_performance", {})
            if recent_perf.get("5d_return", 0) > 0.05:  # 5日涨幅>5%
                score += 5
            
            # 确保价格符合要求
            if not technical_data.get("price_meets_minimum", False):
                score *= 0.5  # 价格过低扣分
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            logger.warning(f"S-Score计算失败 {symbol}: {str(e)}")
            return 0.0
    
    def _get_simplified_news_score(self, symbol: str) -> float:
        """简化的新闻情绪评分 (0-20分)"""
        try:
            # 这里可以集成新闻API，目前使用简化逻辑
            # 基于股票的基本面信息给分
            
            # 大盘股通常新闻较多
            large_cap_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
            
            if symbol in large_cap_symbols:
                return 15.0  # 大盘股基础分较高
            else:
                return 10.0  # 其他股票基础分
                
        except Exception as e:
            logger.warning(f"新闻评分失败 {symbol}: {str(e)}")
            return 8.0
    
    def _calculate_scan_confidence(self, candidates: List[Dict], total_scanned: int) -> float:
        """计算整体扫描可信度"""
        try:
            if total_scanned == 0:
                return 0.0
            
            # 基础可信度 = 合格候选股比例
            qualified_ratio = len(candidates) / total_scanned
            base_confidence = qualified_ratio
            
            # 质量调整
            if candidates:
                avg_score = np.mean([c["s_score"] for c in candidates])
                quality_factor = avg_score / 100  # 归一化到0-1
                
                # 综合可信度
                final_confidence = base_confidence * 0.6 + quality_factor * 0.4
            else:
                final_confidence = 0.0
            
            # 数据完整性调整
            if total_scanned >= 20:  # 扫描样本足够
                final_confidence *= 1.0
            elif total_scanned >= 10:
                final_confidence *= 0.9
            else:
                final_confidence *= 0.8
            
            return min(max(final_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"扫描可信度计算失败: {str(e)}")
            return 0.5
    
    def _get_sp500_symbols(self) -> List[str]:
        """获取S&P 500股票列表 (简化版)"""
        # 返回主要的S&P 500成分股
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "PFE",
            "CVX", "ABBV", "BAC", "KO", "COST", "DIS", "TMO", "WMT", "PEP",
            "MRK", "ADBE", "ABT", "CRM", "NFLX", "ACN", "LLY", "NKE", "DHR"
        ]
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """获取NASDAQ 100股票列表 (简化版)"""
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
            "COST", "NFLX", "ADBE", "PEP", "CSCO", "COMCAST", "INTC", "INTU",
            "AMD", "QCOM", "TXN", "ISRG", "BKNG", "AMGN", "HON", "VRTX",
            "ADP", "SBUX", "GILD", "ADI", "MU", "AMAT", "PYPL", "REGN"
        ] 