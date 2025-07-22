"""
模組S: Helios | 市场扫描与选股器 (升级版)
基于Polygon Advanced API的智能选股系统
实现S-Score评分机制，确保高质量候选股票筛选
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import json

from config import (
    POLYGON_API_KEY, POLYGON_BASE_URL, POLYGON_ADVANCED_FEATURES,
    HELIOS_CONFIG, CONFIDENCE_THRESHOLDS, OPENAI_API_KEY, OPENAI_MODEL
)
from utils import calculate_confidence, get_timestamp

logger = logging.getLogger(__name__)

class HeliosEngine:
    """
    Helios市场扫描与选股器
    
    功能:
    1. 基于交易量、波动性、价格行为的多维度筛选
    2. S-Score评分机制 (0-100分)
    3. Polygon Advanced API数据获取
    4. 预判性筛选与趋势加速点检测
    """
    
    def __init__(self):
        self.name = "Helios"
        self.description = "市场扫描与选股器(升级版)"
        self.polygon_session = None
        self.openai_client = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.polygon_session = aiohttp.ClientSession()
        try:
            import openai
            self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            logger.warning("OpenAI库未安装，AI新闻解读功能将被禁用")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.polygon_session:
            await self.polygon_session.close()
    
    async def scan_market(self, market: str = "SP500", min_volume: float = 500000, 
                         min_price: float = 10.0, max_results: int = 10) -> Dict[str, Any]:
        """
        市场扫描主函数
        
        Args:
            market: 扫描范围 (SP500, NASDAQ100, ALL)
            min_volume: 最小成交量
            min_price: 最小股价
            max_results: 最大返回结果数
            
        Returns:
            扫描结果字典
        """
        try:
            scan_start = datetime.now()
            logger.info(f"🔍 开始扫描{market}市场...")
            
            # 1. 获取股票池
            stock_pool = await self._get_stock_pool(market)
            logger.info(f"📊 获取股票池: {len(stock_pool)}只股票")
            
            # 2. 并行获取所有股票的市场数据
            candidates = []
            
            # 分批处理以避免API限制
            batch_size = 50
            for i in range(0, len(stock_pool), batch_size):
                batch = stock_pool[i:i+batch_size]
                batch_results = await self._process_stock_batch(
                    batch, min_volume, min_price
                )
                candidates.extend(batch_results)
                
                # 防止API限制
                if i + batch_size < len(stock_pool):
                    await asyncio.sleep(1)
            
            # 3. 计算S-Score并排序
            scored_candidates = []
            for candidate in candidates:
                s_score = await self._calculate_s_score(candidate)
                if s_score >= HELIOS_CONFIG["s_score_thresholds"]["minimum"]:
                    candidate["s_score"] = s_score
                    candidate["confidence"] = self._score_to_confidence(s_score)
                    scored_candidates.append(candidate)
            
            # 按S-Score排序
            scored_candidates.sort(key=lambda x: x["s_score"], reverse=True)
            
            # 4. 限制返回数量
            final_candidates = scored_candidates[:max_results]
            
            scan_time = (datetime.now() - scan_start).total_seconds()
            
            result = {
                "scan_time": get_timestamp(),
                "scan_duration": f"{scan_time:.2f}秒",
                "total_scanned": len(stock_pool),
                "qualified_count": len(scored_candidates),
                "candidates": final_candidates,
                "strong_pool_count": len([c for c in scored_candidates 
                                        if c["s_score"] >= HELIOS_CONFIG["s_score_thresholds"]["strong_pool"]]),
                "confidence": calculate_confidence(len(final_candidates), max_results)
            }
            
            logger.info(f"✅ 扫描完成: {len(final_candidates)}个高质量候选")
            return result
            
        except Exception as e:
            logger.error(f"❌ 市场扫描失败: {str(e)}")
            return {
                "error": str(e),
                "scan_time": get_timestamp(),
                "candidates": [],
                "confidence": 0
            }
    
    async def scan_symbol(self, symbol: str, portfolio_value: float) -> Dict[str, Any]:
        """
        单个股票扫描分析
        
        Args:
            symbol: 股票代码
            portfolio_value: 投资组合价值
            
        Returns:
            分析结果
        """
        try:
            logger.info(f"🔍 Helios扫描分析: {symbol}")
            
            # 获取股票数据
            stock_data = await self._get_stock_data(symbol)
            if not stock_data:
                return {"error": f"无法获取{symbol}数据", "confidence": 0}
            
            # 计算S-Score
            s_score = await self._calculate_s_score(stock_data)
            confidence = self._score_to_confidence(s_score)
            
            # 趋势加速点检测
            acceleration_signal = await self._detect_acceleration_point(symbol)
            
            result = {
                "symbol": symbol,
                "s_score": s_score,
                "confidence": confidence,
                "acceleration_detected": acceleration_signal["detected"],
                "acceleration_reason": acceleration_signal["reason"],
                "polygon_data": stock_data,
                "recommendation": self._get_recommendation(s_score),
                "analysis_timestamp": get_timestamp()
            }
            
            logger.info(f"✅ {symbol} S-Score: {s_score:.1f}, 可信度: {confidence:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"❌ {symbol}扫描失败: {str(e)}")
            return {"error": str(e), "confidence": 0}
    
    async def _get_stock_pool(self, market: str) -> List[str]:
        """获取指定市场的股票池"""
        # 简化版本，实际应从Polygon API获取
        stock_pools = {
            "SP500": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "PYPL", "DIS", "ADBE",
                "CRM", "NFLX", "INTC", "AMD", "QCOM", "T", "VZ", "PFE", "KO", "PEP"
            ],
            "NASDAQ100": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ADBE",
                "CRM", "NFLX", "INTC", "AMD", "QCOM", "PYPL", "COST", "AVGO", "TXN"
            ],
            "ALL": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "PYPL", "DIS", "ADBE",
                "CRM", "NFLX", "INTC", "AMD", "QCOM", "T", "VZ", "PFE", "KO", "PEP",
                "COST", "AVGO", "TXN", "LLY", "ABBV", "MRK", "ORCL", "ACN", "TMO"
            ]
        }
        return stock_pools.get(market, stock_pools["SP500"])
    
    async def _process_stock_batch(self, symbols: List[str], min_volume: float, 
                                 min_price: float) -> List[Dict[str, Any]]:
        """批量处理股票数据"""
        batch_results = []
        
        for symbol in symbols:
            try:
                stock_data = await self._get_stock_data(symbol)
                if stock_data and self._meets_basic_criteria(stock_data, min_volume, min_price):
                    batch_results.append(stock_data)
                    
            except Exception as e:
                logger.warning(f"⚠️ 处理{symbol}失败: {str(e)}")
                continue
        
        return batch_results
    
    async def _get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """从Polygon API获取股票数据"""
        try:
            async with aiohttp.ClientSession() as session:
                # 1. 获取基本行情数据
                price_url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/prev"
                headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
                
                async with session.get(price_url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    price_data = await resp.json()
                
                # 2. 获取实时快照
                snapshot_url = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
                async with session.get(snapshot_url, headers=headers) as resp:
                    snapshot_data = await resp.json() if resp.status == 200 else {}
                
                # 整合数据
                if price_data.get("results"):
                    result = price_data["results"][0]
                    return {
                        "symbol": symbol,
                        "price": result.get("c", 0),
                        "volume": result.get("v", 0),
                        "open": result.get("o", 0),
                        "high": result.get("h", 0),
                        "low": result.get("l", 0),
                        "change": result.get("c", 0) - result.get("o", 0),
                        "change_percent": ((result.get("c", 0) - result.get("o", 0)) / result.get("o", 1)) * 100,
                        "snapshot": snapshot_data.get("results", {})
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"❌ 获取{symbol}数据失败: {str(e)}")
            return None
    
    def _meets_basic_criteria(self, stock_data: Dict[str, Any], min_volume: float, 
                            min_price: float) -> bool:
        """检查是否满足基本筛选条件"""
        return (
            stock_data.get("price", 0) >= min_price and
            stock_data.get("volume", 0) >= min_volume and
            stock_data.get("price", 0) > 0
        )
    
    async def _calculate_s_score(self, stock_data: Dict[str, Any]) -> float:
        """
        计算S-Score (0-100)
        基于文档中的加权评分机制
        """
        try:
            score = 0.0
            weights = HELIOS_CONFIG["s_score_weights"]
            
            # 1. RVOL得分 (25%)
            rvol_score = await self._calculate_rvol_score(stock_data)
            score += rvol_score * weights["rvol"]
            
            # 2. ATR波动性得分 (20%)
            atr_score = await self._calculate_atr_score(stock_data)
            score += atr_score * weights["atr_volatility"]
            
            # 3. 跳空幅度得分 (15%)
            gap_score = self._calculate_gap_score(stock_data)
            score += gap_score * weights["gap_size"]
            
            # 4. 均线突破得分 (20%)
            ma_score = await self._calculate_ma_breakout_score(stock_data)
            score += ma_score * weights["ma_breakout"]
            
            # 5. 新闻情绪得分 (20%)
            news_score = await self._calculate_news_sentiment_score(stock_data["symbol"])
            score += news_score * weights["news_sentiment"]
            
            return min(100.0, max(0.0, score * 100))
            
        except Exception as e:
            logger.error(f"❌ S-Score计算失败: {str(e)}")
            return 0.0
    
    async def _calculate_rvol_score(self, stock_data: Dict[str, Any]) -> float:
        """计算相对成交量得分"""
        try:
            current_volume = stock_data.get("volume", 0)
            # 简化版本：假设平均成交量为当前成交量的70%
            avg_volume = current_volume * 0.7
            
            if avg_volume > 0:
                rvol = current_volume / avg_volume
                # RVOL > 3.0 = 满分, 线性递减
                return min(1.0, max(0.0, (rvol - 1.0) / 2.0))
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _calculate_atr_score(self, stock_data: Dict[str, Any]) -> float:
        """计算ATR波动性得分"""
        try:
            price = stock_data.get("price", 0)
            high = stock_data.get("high", price)
            low = stock_data.get("low", price)
            
            if price > 0:
                daily_range = high - low
                atr_ratio = (daily_range / price) * 100
                
                # ATR比率 > 4% = 满分
                return min(1.0, max(0.0, atr_ratio / 4.0))
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_gap_score(self, stock_data: Dict[str, Any]) -> float:
        """计算跳空缺口得分"""
        try:
            # 简化版本：基于当日涨跌幅
            change_percent = abs(stock_data.get("change_percent", 0))
            
            # 跳空 > 2% = 满分
            return min(1.0, max(0.0, change_percent / 2.0))
            
        except Exception:
            return 0.0
    
    async def _calculate_ma_breakout_score(self, stock_data: Dict[str, Any]) -> float:
        """计算均线突破得分"""
        try:
            # 简化版本：基于价格位置
            price = stock_data.get("price", 0)
            high = stock_data.get("high", price)
            
            # 假设突破信号
            breakout_signal = price >= high * 0.98  # 接近最高价
            
            return 1.0 if breakout_signal else 0.3
            
        except Exception:
            return 0.0
    
    async def _calculate_news_sentiment_score(self, symbol: str) -> float:
        """计算新闻情绪得分"""
        try:
            if not self.openai_client:
                return 0.5  # 默认中性得分
            
            # 获取最新新闻
            news_data = await self._get_latest_news(symbol)
            if not news_data:
                return 0.5
            
            # 使用GPT-4o分析情绪
            sentiment = await self._analyze_news_sentiment(news_data)
            
            return sentiment
            
        except Exception as e:
            logger.warning(f"⚠️ 新闻情绪分析失败: {str(e)}")
            return 0.5
    
    async def _get_latest_news(self, symbol: str) -> Optional[List[Dict]]:
        """获取最新新闻"""
        try:
            url = f"{POLYGON_BASE_URL}/v2/reference/news"
            params = {
                "ticker": symbol,
                "limit": 5,
                "order": "desc"
            }
            headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("results", [])
            return None
            
        except Exception:
            return None
    
    async def _analyze_news_sentiment(self, news_data: List[Dict]) -> float:
        """使用GPT-4o分析新闻情绪"""
        try:
            news_text = "\n".join([
                f"标题: {article.get('title', '')}\n描述: {article.get('description', '')}"
                for article in news_data[:3]
            ])
            
            prompt = f"""
            分析以下新闻的整体情绪，返回0-1的分数：
            - 0.0: 极度负面
            - 0.5: 中性
            - 1.0: 极度正面
            
            新闻内容：
            {news_text}
            
            只返回数字分数，不要其他解释。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception:
            return 0.5
    
    async def _detect_acceleration_point(self, symbol: str) -> Dict[str, Any]:
        """检测趋势加速点"""
        try:
            # 简化版本的加速点检测
            return {
                "detected": False,
                "reason": "需要更多实时数据进行加速点检测"
            }
            
        except Exception:
            return {"detected": False, "reason": "检测失败"}
    
    def _score_to_confidence(self, s_score: float) -> float:
        """将S-Score转换为可信度百分比"""
        return min(95.0, max(10.0, s_score * 0.95))
    
    def _get_recommendation(self, s_score: float) -> str:
        """基于S-Score给出建议"""
        if s_score >= HELIOS_CONFIG["s_score_thresholds"]["strong_pool"]:
            return "强选池候选"
        elif s_score >= HELIOS_CONFIG["s_score_thresholds"]["minimum"]:
            return "合格候选"
        else:
            return "不合格" 