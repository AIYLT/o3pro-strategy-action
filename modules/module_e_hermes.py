"""
模块E: Hermes | 事件回溯引擎
功能: 回溯近24小时财报、新闻, 判断利多/利空与市场反应
用途: 调整信号强度
数据依据: 至少2个一致新闻来源
可信度E: 根据事件一致性评估
"""

import requests
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf
from utils import ModuleResult, TimeTracker, logger


class HermesEngine:
    """Hermes事件回溯引擎"""
    
    def __init__(self):
        self.name = "Hermes"
        self.module_id = "E"
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        分析近期新闻事件和财报影响
        
        Args:
            symbol: 股票代码
            data: 输入数据
            
        Returns:
            ModuleResult: 包含事件分析结果
        """
        timer = TimeTracker().start()
        
        try:
            # 获取近期新闻和事件
            news_events = await self._get_recent_news(symbol)
            
            # 获取财报信息
            earnings_info = await self._get_earnings_info(symbol)
            
            # 分析事件情绪和影响
            sentiment_analysis = self._analyze_sentiment(news_events)
            
            # 评估市场反应一致性
            market_reaction = self._evaluate_market_reaction(
                sentiment_analysis, earnings_info, data
            )
            
            # 计算事件一致性
            event_consistency = self._calculate_event_consistency(
                news_events, sentiment_analysis
            )
            
            # 计算最终可信度
            final_confidence = self._calculate_confidence(
                event_consistency, market_reaction, sentiment_analysis
            )
            
            execution_time = timer.stop()
            
            result_data = {
                "news_count": len(news_events),
                "sentiment_score": sentiment_analysis.get("score", 0),
                "sentiment_direction": sentiment_analysis.get("direction", "neutral"),
                "event_consistency": round(event_consistency, 4),
                "market_reaction_score": round(market_reaction.get("score", 0), 4),
                "earnings_impact": earnings_info.get("impact", "none"),
                "final_confidence": round(final_confidence, 4),
                "signal_adjustment": self._get_signal_adjustment(sentiment_analysis),
                "key_events": news_events[:3],  # 前3个关键事件
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Hermes分析完成: {symbol} - 情绪: {sentiment_analysis.get('direction')}, 可信度: {final_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=final_confidence,
                data=result_data,
                execution_time=execution_time,
                status="success"
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Hermes分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_recent_news(self, symbol: str) -> List[Dict[str, Any]]:
        """获取近期新闻 (使用yfinance简化实现)"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            recent_news = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for item in news[:10]:  # 最多10条新闻
                news_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if news_time > cutoff_time:
                    recent_news.append({
                        "title": item.get('title', ''),
                        "summary": item.get('summary', ''),
                        "publisher": item.get('publisher', ''),
                        "timestamp": news_time.isoformat(),
                        "link": item.get('link', '')
                    })
            
            return recent_news
            
        except Exception as e:
            logger.warning(f"获取新闻失败 {symbol}: {str(e)}")
            return []
    
    async def _get_earnings_info(self, symbol: str) -> Dict[str, Any]:
        """获取财报信息"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            earnings_info = {"impact": "none", "next_date": None}
            
            if calendar is not None and not calendar.empty:
                # 检查是否有即将到来的财报
                next_earnings = calendar.index[0] if len(calendar.index) > 0 else None
                
                if next_earnings:
                    days_to_earnings = (next_earnings - datetime.now()).days
                    
                    if -2 <= days_to_earnings <= 7:  # 前后一周内
                        if days_to_earnings <= 0:
                            earnings_info["impact"] = "recent_release"
                        else:
                            earnings_info["impact"] = "upcoming"
                        
                        earnings_info["next_date"] = next_earnings.isoformat()
            
            return earnings_info
            
        except Exception as e:
            logger.warning(f"获取财报信息失败 {symbol}: {str(e)}")
            return {"impact": "none", "next_date": None}
    
    def _analyze_sentiment(self, news_events: List[Dict]) -> Dict[str, Any]:
        """分析新闻情绪 (简化实现)"""
        try:
            if not news_events:
                return {"score": 0, "direction": "neutral", "confidence": 0}
            
            # 简化的情绪分析 - 基于关键词
            positive_keywords = [
                'upgrade', 'buy', 'outperform', 'beat', 'exceeds', 'strong',
                'growth', 'partnership', 'acquisition', 'positive', 'bullish',
                'rally', 'surge', 'gain', 'rise', 'breakthrough'
            ]
            
            negative_keywords = [
                'downgrade', 'sell', 'underperform', 'miss', 'weak', 'decline',
                'loss', 'bearish', 'fall', 'drop', 'concern', 'risk',
                'investigation', 'lawsuit', 'warning', 'cut'
            ]
            
            total_score = 0
            total_weight = 0
            
            for news in news_events:
                text = (news.get('title', '') + ' ' + news.get('summary', '')).lower()
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                
                # 计算单条新闻得分
                news_score = positive_count - negative_count
                total_score += news_score
                total_weight += 1
            
            if total_weight == 0:
                return {"score": 0, "direction": "neutral", "confidence": 0}
            
            # 平均情绪分数
            avg_score = total_score / total_weight
            
            # 确定方向
            if avg_score > 0.5:
                direction = "positive"
            elif avg_score < -0.5:
                direction = "negative"
            else:
                direction = "neutral"
            
            # 计算置信度
            confidence = min(abs(avg_score) / 2.0, 1.0)
            
            return {
                "score": avg_score,
                "direction": direction,
                "confidence": confidence,
                "news_count": len(news_events)
            }
            
        except Exception as e:
            logger.warning(f"情绪分析失败: {str(e)}")
            return {"score": 0, "direction": "neutral", "confidence": 0}
    
    def _evaluate_market_reaction(
        self, 
        sentiment: Dict[str, Any], 
        earnings: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估市场反应一致性"""
        try:
            reaction_score = 0.0
            
            # 情绪与价格表现一致性
            sentiment_direction = sentiment.get("direction", "neutral")
            
            # 简化的价格表现判断 (需要价格数据)
            price_change = current_data.get("price_change_pct", 0)
            
            if sentiment_direction == "positive" and price_change > 0:
                reaction_score += 0.5
            elif sentiment_direction == "negative" and price_change < 0:
                reaction_score += 0.5
            elif sentiment_direction == "neutral":
                reaction_score += 0.3
            
            # 财报影响
            earnings_impact = earnings.get("impact", "none")
            if earnings_impact in ["recent_release", "upcoming"]:
                reaction_score += 0.3
            
            # 新闻数量影响
            news_count = sentiment.get("news_count", 0)
            if news_count >= 3:
                reaction_score += 0.2
            elif news_count >= 1:
                reaction_score += 0.1
            
            return {
                "score": min(reaction_score, 1.0),
                "sentiment_price_consistency": sentiment_direction,
                "earnings_factor": earnings_impact
            }
            
        except Exception as e:
            logger.warning(f"市场反应评估失败: {str(e)}")
            return {"score": 0.5}
    
    def _calculate_event_consistency(
        self, 
        news_events: List[Dict], 
        sentiment: Dict[str, Any]
    ) -> float:
        """计算事件一致性"""
        try:
            if len(news_events) < 2:
                return 0.3  # 新闻数量不足，一致性较低
            
            # 基于情绪置信度
            sentiment_confidence = sentiment.get("confidence", 0)
            
            # 新闻来源多样性
            publishers = set(news.get('publisher', '') for news in news_events)
            source_diversity = min(len(publishers) / 3.0, 1.0)  # 最多3个不同来源满分
            
            # 时间集中度 (24小时内)
            time_concentration = 1.0 if len(news_events) >= 2 else 0.5
            
            # 综合一致性
            consistency = (
                sentiment_confidence * 0.5 +
                source_diversity * 0.3 +
                time_concentration * 0.2
            )
            
            return min(max(consistency, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"事件一致性计算失败: {str(e)}")
            return 0.5
    
    def _calculate_confidence(
        self,
        event_consistency: float,
        market_reaction: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> float:
        """计算最终可信度"""
        try:
            # 基础可信度来自事件一致性
            base_confidence = event_consistency * 0.4
            
            # 市场反应得分
            reaction_score = market_reaction.get("score", 0)
            base_confidence += reaction_score * 0.4
            
            # 情绪强度
            sentiment_confidence = sentiment.get("confidence", 0)
            base_confidence += sentiment_confidence * 0.2
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"可信度计算失败: {str(e)}")
            return 0.5
    
    def _get_signal_adjustment(self, sentiment: Dict[str, Any]) -> str:
        """获取信号调整建议"""
        direction = sentiment.get("direction", "neutral")
        confidence = sentiment.get("confidence", 0)
        
        if direction == "positive" and confidence > 0.7:
            return "strengthen_buy"
        elif direction == "negative" and confidence > 0.7:
            return "weaken_buy"
        else:
            return "neutral" 