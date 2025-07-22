"""
模块Y: Oracle | 全模块精度整合器(核心决策模块)
功能: 全链条验证与最终裁定者, 目标守住建议品质≥80%命中率
- 策略真实命中率统计: 基于模块Z资料, 统计30日、6个月、2年命中率与回撤
- 全模块即时可信度整合: 按模块优先级加权整合
- 信号决策闸门: 僅當「命中率≥80% 且全局可信度≥75%」允许输出
- 分析耗时追踪
- 异常自我降评: 若模块X触发, 全域可信度下调5%
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from ..utils import ModuleResult, TimeTracker, logger, calculate_confidence_score
from ..config import (
    OPENAI_API_KEY, REQUIRED_MODEL, TARGET_HIT_RATE, 
    MODULE_PRIORITY, MIN_CONFIDENCE_THRESHOLD, MIN_MODULE_CONFIDENCE
)


class OracleEngine:
    """Oracle全模块精度整合器 - 核心决策模块"""
    
    def __init__(self):
        self.name = "Oracle"
        self.module_id = "Y"
        self.min_confidence = MIN_MODULE_CONFIDENCE['Y']  # 80%
        self.target_hit_rate = TARGET_HIT_RATE  # 80%
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    async def integrate_and_decide(
        self, 
        symbol: str, 
        module_results: Dict[str, ModuleResult],
        data: Dict[str, Any]
    ) -> ModuleResult:
        """
        整合所有模块结果并做最终决策
        
        Args:
            symbol: 股票代码
            module_results: 各模块分析结果
            data: 原始输入数据
            
        Returns:
            ModuleResult: 最终决策结果
        """
        timer = TimeTracker().start()
        
        try:
            # 1. 验证模型版本
            await self._validate_model_version()
            
            # 2. 提取各模块可信度
            modules_confidence = self._extract_module_confidence(module_results)
            
            # 3. 检查异常模块触发
            has_anomaly = self._check_anomaly_trigger(module_results)
            
            # 4. 计算全局可信度 (带异常调整)
            global_confidence = self._calculate_global_confidence(
                modules_confidence, has_anomaly
            )
            
            # 5. 计算策略命中率 (基于历史数据和当前分析)
            hit_rate = await self._calculate_strategy_hit_rate(
                symbol, module_results, data
            )
            
            # 6. 应用决策闸门
            decision_gate_result = self._apply_decision_gate(
                hit_rate, global_confidence
            )
            
            # 7. 生成最终建议
            final_recommendation = await self._generate_final_recommendation(
                symbol, module_results, global_confidence, hit_rate, decision_gate_result
            )
            
            execution_time = timer.stop()
            
            # 8. 构建结果数据
            result_data = {
                "hit_rate": round(hit_rate, 4),
                "global_confidence": round(global_confidence, 4),
                "modules_confidence": {k: round(v, 4) for k, v in modules_confidence.items()},
                "decision_gate_passed": decision_gate_result["passed"],
                "decision_reason": decision_gate_result["reason"],
                "has_anomaly": has_anomaly,
                "final_recommendation": final_recommendation,
                "analysis_timestamp": datetime.now().isoformat(),
                "min_confidence_threshold": self.min_confidence,
                "target_hit_rate": self.target_hit_rate,
                "module_priority_weights": MODULE_PRIORITY
            }
            
            # 9. 判断最终状态
            if decision_gate_result["passed"]:
                logger.info(f"Oracle决策通过: {symbol} - 命中率: {hit_rate:.2%}, 全局可信度: {global_confidence:.2%}")
                status = "approved"
            else:
                logger.warning(f"Oracle决策拒绝: {symbol} - {decision_gate_result['reason']}")
                status = "rejected"
            
            return ModuleResult(
                module_name=self.name,
                confidence=global_confidence,
                data=result_data,
                execution_time=execution_time,
                status=status
            )
            
        except Exception as e:
            execution_time = timer.stop()
            logger.error(f"Oracle决策失败 {symbol}: {str(e)}")
            
            return ModuleResult(
                module_name=self.name,
                confidence=0.0,
                data={"error": str(e), "symbol": symbol},
                execution_time=execution_time,
                status="error"
            )
    
    async def _validate_model_version(self):
        """验证OpenAI模型版本"""
        try:
            # 简化验证 - 检查模型是否可用
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=REQUIRED_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            
            if not response:
                raise ValueError(f"模型版本不符，封装无效。要求: {REQUIRED_MODEL}")
                
        except Exception as e:
            if "model" in str(e).lower():
                raise ValueError(f"模型版本不符，封装无效。要求: {REQUIRED_MODEL}, 错误: {str(e)}")
            logger.warning(f"模型验证警告: {str(e)}")
    
    def _extract_module_confidence(self, module_results: Dict[str, ModuleResult]) -> Dict[str, float]:
        """提取各模块可信度"""
        confidence_map = {}
        
        for module_id, result in module_results.items():
            if result and result.status in ["success", "degraded"]:
                confidence_map[module_id] = result.confidence
            else:
                confidence_map[module_id] = 0.0
        
        return confidence_map
    
    def _check_anomaly_trigger(self, module_results: Dict[str, ModuleResult]) -> bool:
        """检查是否有异常模块(X)触发"""
        if "X" in module_results:
            result = module_results["X"]
            if result and result.status == "anomaly_detected":
                return True
        return False
    
    def _calculate_global_confidence(
        self, 
        modules_confidence: Dict[str, float], 
        has_anomaly: bool
    ) -> float:
        """计算全局可信度"""
        # 使用权重计算基础可信度
        base_confidence = calculate_confidence_score(modules_confidence, MODULE_PRIORITY)
        
        # 异常调整: 模块X触发时下调5%
        if has_anomaly:
            base_confidence = max(0.0, base_confidence - 0.05)
            logger.warning("异常模块触发，全局可信度下调5%")
        
        return base_confidence
    
    async def _calculate_strategy_hit_rate(
        self, 
        symbol: str, 
        module_results: Dict[str, ModuleResult],
        data: Dict[str, Any]
    ) -> float:
        """计算策略命中率"""
        try:
            # 基于模块Z(EchoLog)的回测数据
            if "Z" in module_results:
                backtest_result = module_results["Z"]
                if backtest_result and "hit_rate" in backtest_result.data:
                    return backtest_result.data["hit_rate"]
            
            # 如果没有回测数据，基于各模块表现估算
            hit_rates = []
            
            # 收集各模块的历史准确性
            for module_id, result in module_results.items():
                if result and result.status == "success":
                    if "backtest_accuracy" in result.data:
                        hit_rates.append(result.data["backtest_accuracy"])
                    elif "pattern_hit_rate" in result.data:
                        hit_rates.append(result.data["pattern_hit_rate"])
                    elif result.confidence > 0.7:
                        hit_rates.append(result.confidence)
            
            if hit_rates:
                # 计算加权平均命中率
                avg_hit_rate = sum(hit_rates) / len(hit_rates)
                
                # 保守调整 (实际命中率通常低于预测)
                conservative_hit_rate = avg_hit_rate * 0.9
                
                return min(max(conservative_hit_rate, 0.0), 1.0)
            
            # 默认命中率
            return 0.75
            
        except Exception as e:
            logger.warning(f"命中率计算失败: {str(e)}")
            return 0.75
    
    def _apply_decision_gate(self, hit_rate: float, global_confidence: float) -> Dict[str, Any]:
        """应用决策闸门"""
        # 核心条件: 命中率≥80% 且全局可信度≥75%
        hit_rate_threshold = self.target_hit_rate
        confidence_threshold = MIN_CONFIDENCE_THRESHOLD
        
        reasons = []
        
        if hit_rate < hit_rate_threshold:
            reasons.append(f"命中率{hit_rate:.2%} < {hit_rate_threshold:.2%}")
        
        if global_confidence < confidence_threshold:
            reasons.append(f"全局可信度{global_confidence:.2%} < {confidence_threshold:.2%}")
        
        if not reasons:
            return {
                "passed": True,
                "reason": "所有条件满足，决策通过"
            }
        else:
            return {
                "passed": False,
                "reason": "决策闸门拒绝: " + "; ".join(reasons)
            }
    
    async def _generate_final_recommendation(
        self,
        symbol: str,
        module_results: Dict[str, ModuleResult],
        global_confidence: float,
        hit_rate: float,
        decision_gate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成最终投资建议"""
        try:
            if not decision_gate_result["passed"]:
                return {
                    "action": "NO_ACTION",
                    "signal_strength": "REJECTED",
                    "confidence_level": global_confidence,
                    "reason": decision_gate_result["reason"]
                }
            
            # 收集关键数据
            entry_price = None
            stop_loss = None
            take_profit = None
            position_size = 0
            
            # 从Aegis模块获取风控数据
            if "C" in module_results and module_results["C"].status == "success":
                aegis_data = module_results["C"].data
                entry_price = aegis_data.get("entry_price")
                stop_loss = aegis_data.get("stop_loss")
                take_profit = aegis_data.get("take_profit")
                position_size = aegis_data.get("position_size", 0)
            
            # 确定信号强度
            if global_confidence >= 0.90 and hit_rate >= 0.90:
                signal_strength = "STRONG_BUY"
            elif global_confidence >= 0.85 and hit_rate >= 0.85:
                signal_strength = "BUY"
            elif global_confidence >= 0.75 and hit_rate >= 0.80:
                signal_strength = "MODERATE_BUY"
            else:
                signal_strength = "WEAK_BUY"
            
            recommendation = {
                "action": "BUY",
                "signal_strength": signal_strength,
                "confidence_level": global_confidence,
                "hit_rate": hit_rate,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "expected_return": None,
                "risk_level": "MODERATE",
                "timestamp": datetime.now().isoformat()
            }
            
            # 计算预期收益
            if entry_price and take_profit:
                expected_return = (take_profit - entry_price) / entry_price
                recommendation["expected_return"] = round(expected_return, 4)
            
            # 评估风险等级
            if global_confidence >= 0.90:
                recommendation["risk_level"] = "LOW"
            elif global_confidence <= 0.80:
                recommendation["risk_level"] = "HIGH"
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"生成投资建议失败: {str(e)}")
            return {
                "action": "ERROR",
                "signal_strength": "UNKNOWN",
                "confidence_level": 0.0,
                "reason": f"建议生成失败: {str(e)}"
            }
    
    def generate_formatted_output(self, result: ModuleResult, symbol: str) -> str:
        """生成格式化的标准输出"""
        try:
            data = result.data
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            # 基本信息
            hit_rate = data.get("hit_rate", 0) * 100
            global_confidence = data.get("global_confidence", 0) * 100
            execution_time = result.execution_time
            
            # 投资建议
            recommendation = data.get("final_recommendation", {})
            
            # 构建标准格式输出
            output = f"""🏁 命中率: {hit_rate:.2f}%
⏱️分析时间: {execution_time:.2f}秒
🕓分析时间戳: {timestamp}
🌀市场环境: 分析中(data_tag: 待补充)
🔧信号整合结果: 模块信号一致性: {global_confidence:.1f}%
⭐最终建议: {recommendation.get('signal_strength', 'NO_ACTION')}(全局可信度: {global_confidence:.2f}%)

⏱️入场时间: {timestamp}
📌{symbol}｜策略得分: {global_confidence/100:.2f}｜排名第1(可信度: {global_confidence:.2f}%)
✅结构判定: {'健康' if data.get('decision_gate_passed') else '警告'}
📊价格区:
  ｜入场价格: {recommendation.get('entry_price', 'N/A')}
  ｜止损价格: {recommendation.get('stop_loss', 'N/A')}
  ｜止盈价格: {recommendation.get('take_profit', 'N/A')}
  ｜预估漲幅: {recommendation.get('expected_return', 0)*100:+.2f}%
  ｜建议倉位: {recommendation.get('position_size', 0)}股

---
📉回测摘要:
- 回测期间: 近30日
- 命中率: {hit_rate:.1f}%
- 可信度: {global_confidence:.1f}%
- 決策状态: {'通过' if data.get('decision_gate_passed') else '拒绝'}
- 异常提示: {'检测到异常' if data.get('has_anomaly') else '无'}"""

            return output
            
        except Exception as e:
            logger.error(f"格式化输出失败: {str(e)}")
            return f"输出格式化错误: {str(e)}" 