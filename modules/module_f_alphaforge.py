"""
模块F: AlphaForge | 因子扩展层
功能: 引入外部因子(EPS增长率、ROE等)判断基本面健康度
用途: 补强短线技术信号
数据依据: 最新季度财报与统计异动
可信度F: 整合后若<60%不纳入评分
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ..utils import ModuleResult, TimeTracker, logger
from ..config import MIN_MODULE_CONFIDENCE


class AlphaForgeEngine:
    """AlphaForge因子扩展层"""
    
    def __init__(self):
        self.name = "AlphaForge"
        self.module_id = "F"
        self.min_confidence = MIN_MODULE_CONFIDENCE['F']  # 60%
        
    async def analyze(self, symbol: str, data: Dict[str, Any]) -> ModuleResult:
        """
        分析基本面因子健康度
        
        Args:
            symbol: 股票代码
            data: 输入数据
            
        Returns:
            ModuleResult: 包含基本面分析结果
        """
        timer = TimeTracker().start()
        
        try:
            # 获取基本面数据
            fundamental_data = await self._get_fundamental_data(symbol)
            
            # 计算财务健康度指标
            financial_health = self._calculate_financial_health(fundamental_data)
            
            # 计算成长性指标
            growth_metrics = self._calculate_growth_metrics(fundamental_data)
            
            # 计算估值合理性
            valuation_metrics = self._calculate_valuation_metrics(fundamental_data)
            
            # 综合评分
            composite_score = self._calculate_composite_score(
                financial_health, growth_metrics, valuation_metrics
            )
            
            # 判断是否达到最低可信度要求
            is_valid = composite_score >= self.min_confidence
            status = "success" if is_valid else "insufficient"
            
            execution_time = timer.stop()
            
            result_data = {
                "composite_score": round(composite_score, 4),
                "financial_health_score": round(financial_health.get("score", 0), 4),
                "growth_score": round(growth_metrics.get("score", 0), 4),
                "valuation_score": round(valuation_metrics.get("score", 0), 4),
                "is_valid": is_valid,
                "min_confidence_threshold": self.min_confidence,
                "key_metrics": {
                    "eps_growth": fundamental_data.get("eps_growth", 0),
                    "roe": fundamental_data.get("roe", 0),
                    "debt_to_equity": fundamental_data.get("debt_to_equity", 0),
                    "pe_ratio": fundamental_data.get("pe_ratio", 0),
                    "peg_ratio": fundamental_data.get("peg_ratio", 0)
                },
                "health_indicators": financial_health,
                "growth_indicators": growth_metrics,
                "valuation_indicators": valuation_metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if is_valid:
                logger.info(f"AlphaForge评估通过: {symbol} - 综合得分: {composite_score:.2%}")
            else:
                logger.warning(f"AlphaForge评估不足: {symbol} - 得分: {composite_score:.2%} < {self.min_confidence:.2%}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=composite_score,
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"AlphaForge分析失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """获取基本面数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            
            # 提取关键指标
            fundamental_data = {
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "peg_ratio": info.get('pegRatio', 0),
                "price_to_book": info.get('priceToBook', 0),
                "roe": info.get('returnOnEquity', 0),
                "debt_to_equity": info.get('debtToEquity', 0),
                "current_ratio": info.get('currentRatio', 0),
                "gross_margin": info.get('grossMargins', 0),
                "profit_margin": info.get('profitMargins', 0),
                "eps_current": info.get('trailingEps', 0),
                "eps_forward": info.get('forwardEps', 0),
                "revenue_growth": info.get('revenueGrowth', 0),
                "earnings_growth": info.get('earningsGrowth', 0),
                "book_value": info.get('bookValue', 0),
                "cash_per_share": info.get('totalCashPerShare', 0)
            }
            
            # 计算EPS增长率
            if fundamental_data["eps_current"] and fundamental_data["eps_forward"]:
                fundamental_data["eps_growth"] = (
                    fundamental_data["eps_forward"] - fundamental_data["eps_current"]
                ) / abs(fundamental_data["eps_current"])
            else:
                fundamental_data["eps_growth"] = fundamental_data.get("earnings_growth", 0)
            
            return fundamental_data
            
        except Exception as e:
            logger.warning(f"获取基本面数据失败 {symbol}: {str(e)}")
            return {}
    
    def _calculate_financial_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算财务健康度"""
        try:
            health_score = 0.0
            indicators = {}
            
            # ROE (净资产收益率)
            roe = data.get("roe", 0)
            if roe > 0.15:  # >15%
                health_score += 0.25
                indicators["roe_rating"] = "excellent"
            elif roe > 0.10:  # >10%
                health_score += 0.20
                indicators["roe_rating"] = "good"
            elif roe > 0.05:  # >5%
                health_score += 0.10
                indicators["roe_rating"] = "fair"
            else:
                indicators["roe_rating"] = "poor"
            
            # 债务股本比
            debt_to_equity = data.get("debt_to_equity", 0)
            if debt_to_equity < 0.3:  # <30%
                health_score += 0.25
                indicators["debt_rating"] = "excellent"
            elif debt_to_equity < 0.6:  # <60%
                health_score += 0.20
                indicators["debt_rating"] = "good"
            elif debt_to_equity < 1.0:  # <100%
                health_score += 0.10
                indicators["debt_rating"] = "fair"
            else:
                indicators["debt_rating"] = "poor"
            
            # 流动比率
            current_ratio = data.get("current_ratio", 0)
            if current_ratio >= 2.0:
                health_score += 0.25
                indicators["liquidity_rating"] = "excellent"
            elif current_ratio >= 1.5:
                health_score += 0.20
                indicators["liquidity_rating"] = "good"
            elif current_ratio >= 1.0:
                health_score += 0.10
                indicators["liquidity_rating"] = "fair"
            else:
                indicators["liquidity_rating"] = "poor"
            
            # 利润率
            profit_margin = data.get("profit_margin", 0)
            if profit_margin > 0.20:  # >20%
                health_score += 0.25
                indicators["margin_rating"] = "excellent"
            elif profit_margin > 0.10:  # >10%
                health_score += 0.20
                indicators["margin_rating"] = "good"
            elif profit_margin > 0.05:  # >5%
                health_score += 0.10
                indicators["margin_rating"] = "fair"
            else:
                indicators["margin_rating"] = "poor"
            
            return {
                "score": health_score,
                "indicators": indicators,
                "roe": roe,
                "debt_to_equity": debt_to_equity,
                "current_ratio": current_ratio,
                "profit_margin": profit_margin
            }
            
        except Exception as e:
            logger.warning(f"财务健康度计算失败: {str(e)}")
            return {"score": 0.0, "indicators": {}}
    
    def _calculate_growth_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算成长性指标"""
        try:
            growth_score = 0.0
            indicators = {}
            
            # EPS增长率
            eps_growth = data.get("eps_growth", 0)
            if eps_growth > 0.25:  # >25%
                growth_score += 0.4
                indicators["eps_growth_rating"] = "excellent"
            elif eps_growth > 0.15:  # >15%
                growth_score += 0.3
                indicators["eps_growth_rating"] = "good"
            elif eps_growth > 0.05:  # >5%
                growth_score += 0.2
                indicators["eps_growth_rating"] = "fair"
            else:
                indicators["eps_growth_rating"] = "poor"
            
            # 营收增长率
            revenue_growth = data.get("revenue_growth", 0)
            if revenue_growth > 0.20:  # >20%
                growth_score += 0.3
                indicators["revenue_growth_rating"] = "excellent"
            elif revenue_growth > 0.10:  # >10%
                growth_score += 0.25
                indicators["revenue_growth_rating"] = "good"
            elif revenue_growth > 0.05:  # >5%
                growth_score += 0.15
                indicators["revenue_growth_rating"] = "fair"
            else:
                indicators["revenue_growth_rating"] = "poor"
            
            # 整体盈利增长
            earnings_growth = data.get("earnings_growth", 0)
            if earnings_growth > 0.20:  # >20%
                growth_score += 0.3
                indicators["earnings_growth_rating"] = "excellent"
            elif earnings_growth > 0.10:  # >10%
                growth_score += 0.25
                indicators["earnings_growth_rating"] = "good"
            elif earnings_growth > 0:
                growth_score += 0.15
                indicators["earnings_growth_rating"] = "fair"
            else:
                indicators["earnings_growth_rating"] = "poor"
            
            return {
                "score": growth_score,
                "indicators": indicators,
                "eps_growth": eps_growth,
                "revenue_growth": revenue_growth,
                "earnings_growth": earnings_growth
            }
            
        except Exception as e:
            logger.warning(f"成长性指标计算失败: {str(e)}")
            return {"score": 0.0, "indicators": {}}
    
    def _calculate_valuation_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """计算估值合理性"""
        try:
            valuation_score = 0.0
            indicators = {}
            
            # PE比率
            pe_ratio = data.get("pe_ratio", 0)
            if 0 < pe_ratio < 15:  # 低估值
                valuation_score += 0.4
                indicators["pe_rating"] = "undervalued"
            elif 15 <= pe_ratio < 25:  # 合理估值
                valuation_score += 0.3
                indicators["pe_rating"] = "fair"
            elif 25 <= pe_ratio < 40:  # 略高估
                valuation_score += 0.2
                indicators["pe_rating"] = "high"
            else:
                indicators["pe_rating"] = "overvalued"
            
            # PEG比率
            peg_ratio = data.get("peg_ratio", 0)
            if 0 < peg_ratio < 1.0:  # PEG < 1 通常被认为低估
                valuation_score += 0.3
                indicators["peg_rating"] = "undervalued"
            elif 1.0 <= peg_ratio < 1.5:
                valuation_score += 0.25
                indicators["peg_rating"] = "fair"
            elif 1.5 <= peg_ratio < 2.0:
                valuation_score += 0.15
                indicators["peg_rating"] = "high"
            else:
                indicators["peg_rating"] = "overvalued"
            
            # 市净率
            price_to_book = data.get("price_to_book", 0)
            if 0 < price_to_book < 1.5:
                valuation_score += 0.3
                indicators["pb_rating"] = "undervalued"
            elif 1.5 <= price_to_book < 3.0:
                valuation_score += 0.25
                indicators["pb_rating"] = "fair"
            elif 3.0 <= price_to_book < 5.0:
                valuation_score += 0.15
                indicators["pb_rating"] = "high"
            else:
                indicators["pb_rating"] = "overvalued"
            
            return {
                "score": valuation_score,
                "indicators": indicators,
                "pe_ratio": pe_ratio,
                "peg_ratio": peg_ratio,
                "price_to_book": price_to_book
            }
            
        except Exception as e:
            logger.warning(f"估值指标计算失败: {str(e)}")
            return {"score": 0.0, "indicators": {}}
    
    def _calculate_composite_score(
        self, 
        financial_health: Dict[str, Any],
        growth_metrics: Dict[str, Any],
        valuation_metrics: Dict[str, Any]
    ) -> float:
        """计算综合得分"""
        try:
            # 权重分配
            weights = {
                "financial_health": 0.4,  # 财务健康度40%
                "growth": 0.35,           # 成长性35%
                "valuation": 0.25         # 估值25%
            }
            
            health_score = financial_health.get("score", 0)
            growth_score = growth_metrics.get("score", 0)
            valuation_score = valuation_metrics.get("score", 0)
            
            composite_score = (
                health_score * weights["financial_health"] +
                growth_score * weights["growth"] +
                valuation_score * weights["valuation"]
            )
            
            return min(max(composite_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"综合得分计算失败: {str(e)}")
            return 0.0 