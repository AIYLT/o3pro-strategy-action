#!/usr/bin/env python3
"""
智鑑富專屬智能投資顧問 - 主程序
基于GPT-4o的AI日內交易決策分析師
目标命中率≥80%，机构级水准分析
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging
from datetime import datetime
import json
import os

# 导入所有策略模块
from modules.module_a_chronos import ChronosEngine
from modules.module_b_minerva import MinervaEngine  
from modules.module_c_aegis import AegisEngine
from modules.module_d_fenrir import FenrirEngine
from modules.module_e_hermes import HermesEngine
from modules.module_f_alphaforge import AlphaForgeEngine
from modules.module_g_terrafilter import TerraFilterEngine
from modules.module_s_helios import HeliosEngine
from modules.module_x_cerberus import CerberusEngine
from modules.module_y_oracle import OracleEngine
from modules.module_z_echolog import EchoLogEngine

from config import *
from utils import calculate_confidence, track_time, get_timestamp, validate_model_version

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="智鑑富專屬智能投資顧問",
    description="基于GPT-4o的AI日內交易決策分析師，目标命中率≥80%",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API请求模型
class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="股票代码 (如: AAPL)")
    portfolio_value: float = Field(100000, description="投资组合总值")
    risk_level: str = Field("moderate", description="风险等级")
    enable_modules: List[str] = Field(
        ["A", "B", "C", "D", "E", "F", "G", "S", "X", "Y", "Z"], 
        description="启用的分析模块"
    )

class ScanRequest(BaseModel):
    market: str = Field("SP500", description="扫描市场范围")
    min_volume: float = Field(500000, description="最小成交量")
    min_price: float = Field(10.0, description="最小股价")
    max_results: int = Field(10, description="最大返回数量")

# 初始化所有模块引擎
engines = {
    "A": ChronosEngine(),     # 策略评分引擎
    "B": MinervaEngine(),     # 行为验证器
    "C": AegisEngine(),       # 风控调度系统
    "D": FenrirEngine(),      # 主力结构识别器
    "E": HermesEngine(),      # 事件回溯引擎
    "F": AlphaForgeEngine(),  # 因子扩展层
    "G": TerraFilterEngine(), # 环境过滤器
    "S": HeliosEngine(),      # 市场扫描与选股器
    "X": CerberusEngine(),    # 异常防护模组
    "Y": OracleEngine(),      # 全模組精度整合器(核心)
    "Z": EchoLogEngine(),     # 模擬與回測器
}

@app.on_startup
async def startup_event():
    """启动时验证系统配置"""
    try:
        # 验证GPT-4o模型
        if MODEL_VALIDATION_REQUIRED:
            is_valid = await validate_model_version(OPENAI_MODEL)
            if not is_valid:
                logger.error(f"❌ 模型验证失败: {OPENAI_MODEL}")
                raise Exception("Model validation failed")
        
        logger.info("🚀 智鑑富專屬智能投資顧問已启动")
        logger.info(f"📊 目标命中率: ≥{TARGET_HIT_RATE}%")
        logger.info(f"🤖 AI模型: {OPENAI_MODEL}")
        logger.info(f"📈 已加载{len(engines)}个策略模块")
        
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "system": "智鑑富專屬智能投資顧問",
        "model": OPENAI_MODEL,
        "target_hit_rate": f"≥{TARGET_HIT_RATE}%",
        "modules_loaded": len(engines),
        "timestamp": get_timestamp()
    }

@app.post("/analyze")
async def analyze_stock(request: AnalyzeRequest):
    """
    核心股票分析API
    严格按照文档要求执行11个模块分析
    确保≥80%命中率
    """
    start_time = track_time()
    
    try:
        logger.info(f"🔍 开始分析 {request.symbol}")
        
        # 验证输入参数
        if not request.symbol or len(request.symbol) > 10:
            raise HTTPException(status_code=400, detail="无效的股票代码")
        
        # 初始化结果容器
        analysis_results = {
            "symbol": request.symbol.upper(),
            "portfolio_value": request.portfolio_value,
            "timestamp": get_timestamp(),
            "modules_result": {},
            "polygon_data": {},
            "backtest_summary": {},
            "markdown_report": ""
        }
        
        # 按优先级执行模块分析
        enabled_modules = sorted(
            request.enable_modules, 
            key=lambda x: MODULE_PRIORITY.get(x, 99)
        )
        
        # 1. 首先执行Helios市场扫描(S模块)
        if "S" in enabled_modules:
            logger.info("📊 执行Helios市场扫描...")
            helios_result = await engines["S"].scan_symbol(
                request.symbol, request.portfolio_value
            )
            analysis_results["modules_result"]["helios_scan"] = helios_result
        
        # 2. 执行核心分析模块
        module_tasks = []
        for module in enabled_modules:
            if module != "S" and module != "Y" and module in engines:
                task = analyze_with_module(
                    engines[module], request.symbol, request.portfolio_value
                )
                module_tasks.append((module, task))
        
        # 并行执行分析任务
        module_results = {}
        for module, task in module_tasks:
            try:
                result = await task
                module_results[module] = result
                logger.info(f"✅ 模块{module}分析完成")
            except Exception as e:
                logger.error(f"❌ 模块{module}分析失败: {str(e)}")
                module_results[module] = {"error": str(e), "confidence": 0}
        
        # 合并Helios结果
        if "S" in analysis_results["modules_result"]:
            module_results["S"] = analysis_results["modules_result"]["helios_scan"]
        
        # 3. 最后执行Oracle核心决策模块
        if "Y" in enabled_modules:
            logger.info("🎯 执行Oracle核心决策...")
            async with engines["Y"] as oracle:
                oracle_result = await oracle.analyze(
                    request.symbol, request.portfolio_value, module_results
                )
            analysis_results["modules_result"]["oracle_decision"] = oracle_result
            
            # 从Oracle获取最终结果
            analysis_results.update({
                "hit_rate": oracle_result.get("hit_rate", 0),
                "global_confidence": oracle_result.get("global_confidence", 0),
                "recommendation": oracle_result.get("recommendation", "持有"),
                "entry_price": oracle_result.get("entry_price", 0),
                "stop_loss": oracle_result.get("stop_loss", 0),
                "take_profit": oracle_result.get("take_profit", 0),
                "position_size": oracle_result.get("position_size", 0)
            })
        
        # 4. 执行回测模块(Z)
        if "Z" in enabled_modules:
            logger.info("📈 执行回测分析...")
            backtest_result = await engines["Z"].run_backtest(
                request.symbol, analysis_results
            )
            analysis_results["backtest_summary"] = backtest_result
        
        # 5. 生成Markdown报告
        analysis_results["markdown_report"] = generate_markdown_report(
            analysis_results
        )
        
        # 计算分析耗时
        analysis_time = track_time() - start_time
        analysis_results["analysis_time"] = analysis_time
        
        # 验证命中率要求
        if analysis_results.get("hit_rate", 0) < TARGET_HIT_RATE:
            logger.warning(f"⚠️ 命中率{analysis_results.get('hit_rate')}%低于目标{TARGET_HIT_RATE}%")
        
        logger.info(f"✅ {request.symbol}分析完成，耗时{analysis_time:.2f}秒")
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ 分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@app.post("/scan")
async def scan_market(request: ScanRequest):
    """市场扫描API，使用Helios模块"""
    try:
        logger.info(f"🔍 开始扫描{request.market}市场...")
        
        scan_result = await engines["S"].scan_market(
            market=request.market,
            min_volume=request.min_volume,
            min_price=request.min_price,
            max_results=request.max_results
        )
        
        logger.info(f"✅ 扫描完成，发现{len(scan_result.get('candidates', []))}个候选")
        return scan_result
        
    except Exception as e:
        logger.error(f"❌ 市场扫描失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")

async def analyze_with_module(engine, symbol: str, portfolio_value: float):
    """执行单个模块分析"""
    try:
        if hasattr(engine, 'analyze'):
            return await engine.analyze(symbol, portfolio_value)
        else:
            return {"error": "模块无analyze方法", "confidence": 0}
    except Exception as e:
        return {"error": str(e), "confidence": 0}

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """生成详细的Markdown分析报告"""
    symbol = results.get("symbol", "Unknown")
    hit_rate = results.get("hit_rate", 0)
    confidence = results.get("global_confidence", 0)
    recommendation = results.get("recommendation", "持有")
    
    report = f"""# 🎯 智鑑富策略分析报告

## 📊 基本信息
- **股票代码**: {symbol}
- **分析时间**: {results.get('timestamp')}
- **命中率**: {hit_rate:.2f}%
- **全局可信度**: {confidence:.2f}%
- **最终建议**: **{recommendation}**

## 💰 交易建议
- **入场价格**: ${results.get('entry_price', 0):.2f}
- **止损价格**: ${results.get('stop_loss', 0):.2f}
- **止盈价格**: ${results.get('take_profit', 0):.2f}
- **建议仓位**: {results.get('position_size', 0)}股

## 📈 模块分析结果
"""
    
    modules = results.get("modules_result", {})
    for module_name, module_result in modules.items():
        if isinstance(module_result, dict) and "confidence" in module_result:
            conf = module_result.get("confidence", 0)
            status = "✅" if conf >= 70 else "⚠️" if conf >= 50 else "❌"
            report += f"- {status} **{module_name}**: {conf:.1f}%\n"
    
    # 添加回测摘要
    backtest = results.get("backtest_summary", {})
    if backtest:
        report += f"""
## 📊 回测摘要
- **回测期间**: {backtest.get('period', 'N/A')}
- **总信号数**: {backtest.get('total_signals', 0)}
- **命中次数**: {backtest.get('hit_count', 0)}
- **平均盈亏比**: {backtest.get('avg_profit_loss_ratio', 0):.2f}
- **最大回撤**: {backtest.get('max_drawdown', 0):.2f}%
"""
    
    report += f"""
## ⚠️ 风险提示
本分析基于AI模型和历史数据，投资有风险，决策需谨慎。

---
*Generated by 智鑑富專屬智能投資顧問 - {results.get('timestamp')}*
"""
    
    return report

if __name__ == "__main__":
    import uvicorn
    
    # 获取端口号
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 启动智鑑富專屬智能投資顧問在端口{port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    ) 