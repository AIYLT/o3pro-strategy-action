"""
智鉴富策略分析系统 - 主应用
提供RESTful API接口用于股票分析和策略决策
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime

from modules.module_a_chronos import ChronosEngine
from modules.module_b_minerva import MinervaEngine
from modules.module_c_aegis import AegisEngine
from modules.module_y_oracle import OracleEngine
from modules.module_s_helios import HeliosEngine
from utils import setup_logging, get_timestamp
from config import REQUIRED_MODEL, TARGET_HIT_RATE

# 设置日志
logger = setup_logging()

# 创建FastAPI应用
app = FastAPI(
    title="智鉴富策略分析系统",
    description="基于GPT-4o的智能投资顾问API",
    version="1.0.0"
)

# 跨域设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模块引擎
chronos_engine = ChronosEngine()
minerva_engine = MinervaEngine()
aegis_engine = AegisEngine()
oracle_engine = OracleEngine()
helios_engine = HeliosEngine()


# 请求模型
class AnalyzeRequest(BaseModel):
    symbol: str
    price: Optional[float] = None
    volume: Optional[int] = None
    account_value: Optional[float] = 100000

class ScanRequest(BaseModel):
    stock_pool: Optional[str] = "tech_leaders"
    custom_symbols: Optional[List[str]] = None


# 响应模型
class AnalysisResponse(BaseModel):
    symbol: str
    timestamp: str
    decision: str
    confidence: float
    hit_rate: float
    recommendation: Dict[str, Any]
    modules_analysis: Dict[str, Any]
    formatted_output: str
    execution_time: float


@app.get("/")
async def root():
    """根路径 - API信息"""
    return {
        "service": "智鉴富策略分析系统",
        "version": "1.0.0",
        "model": REQUIRED_MODEL,
        "target_hit_rate": f"{TARGET_HIT_RATE:.1%}",
        "status": "运行中",
        "timestamp": get_timestamp()
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "modules": {
            "chronos": "active",
            "minerva": "active", 
            "aegis": "active",
            "oracle": "active",
            "helios": "active"
        }
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalyzeRequest):
    """
    分析单个股票
    
    主要流程:
    1. 运行所有分析模块
    2. Oracle模块整合决策
    3. 返回标准格式结果
    """
    try:
        logger.info(f"开始分析股票: {request.symbol}")
        
        # 准备输入数据
        input_data = {
            "price": request.price,
            "volume": request.volume,
            "account_value": request.account_value
        }
        
        # 运行各个分析模块
        module_results = {}
        
        # 并行运行模块分析
        tasks = [
            ("A", chronos_engine.analyze(request.symbol, input_data)),
            ("B", minerva_engine.analyze(request.symbol, input_data)),
            ("C", aegis_engine.analyze(request.symbol, input_data))
        ]
        
        # 执行并行任务
        for module_id, task in tasks:
            try:
                result = await task
                module_results[module_id] = result
                logger.info(f"模块 {module_id} 完成: 可信度 {result.confidence:.2%}")
            except Exception as e:
                logger.error(f"模块 {module_id} 分析失败: {str(e)}")
                module_results[module_id] = None
        
        # Oracle模块整合决策
        oracle_result = await oracle_engine.integrate_and_decide(
            request.symbol, module_results, input_data
        )
        
        # 生成标准格式输出
        formatted_output = oracle_engine.generate_formatted_output(
            oracle_result, request.symbol
        )
        
        # 构建响应
        oracle_data = oracle_result.data
        recommendation = oracle_data.get("final_recommendation", {})
        
        response = AnalysisResponse(
            symbol=request.symbol,
            timestamp=get_timestamp(),
            decision=recommendation.get("action", "NO_ACTION"),
            confidence=oracle_result.confidence,
            hit_rate=oracle_data.get("hit_rate", 0),
            recommendation=recommendation,
            modules_analysis={
                k: v.to_dict() if v else None 
                for k, v in module_results.items()
            },
            formatted_output=formatted_output,
            execution_time=oracle_result.execution_time
        )
        
        logger.info(f"分析完成: {request.symbol} - 决策: {response.decision}")
        return response
        
    except Exception as e:
        logger.error(f"分析失败 {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/scan")
async def scan_market(request: ScanRequest):
    """
    批量扫描股票池
    
    功能:
    1. 扫描指定股票池
    2. 按S-Score排序
    3. 返回潜力候选股
    """
    try:
        logger.info(f"开始扫描股票池: {request.stock_pool}")
        
        # 执行市场扫描
        scan_result = await helios_engine.scan_market(
            stock_pool=request.stock_pool,
            custom_symbols=request.custom_symbols
        )
        
        # 构建响应
        scan_data = scan_result.data
        
        response = {
            "scan_timestamp": get_timestamp(),
            "stock_pool": request.stock_pool,
            "scan_confidence": scan_result.confidence,
            "total_scanned": scan_data.get("total_scanned", 0),
            "qualified_candidates": scan_data.get("qualified_candidates", 0),
            "strong_candidates": scan_data.get("strong_candidates", 0),
            "strong_pool": scan_data.get("strong_pool", []),
            "moderate_pool": scan_data.get("moderate_pool", []),
            "is_high_quality": scan_data.get("is_high_quality", False),
            "execution_time": scan_result.execution_time,
            "status": scan_result.status
        }
        
        logger.info(f"扫描完成: 发现 {response['qualified_candidates']} 个候选股")
        return response
        
    except Exception as e:
        logger.error(f"扫描失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@app.get("/pools")
async def get_available_pools():
    """获取可用的股票池"""
    return {
        "available_pools": {
            "tech_leaders": "科技龙头股 (8只)",
            "sp500": "标普500主要成分股 (36只)", 
            "nasdaq100": "纳斯达克100主要成分股 (32只)"
        },
        "custom_symbols_supported": True,
        "max_custom_symbols": 50
    }


@app.get("/status")
async def system_status():
    """系统状态检查"""
    return {
        "system": "智鉴富策略分析系统",
        "model": REQUIRED_MODEL,
        "target_hit_rate": TARGET_HIT_RATE,
        "modules": {
            "A_Chronos": "策略评分引擎",
            "B_Minerva": "行为验证器",
            "C_Aegis": "风控调度系统", 
            "S_Helios": "市场扫描器",
            "Y_Oracle": "核心决策模块"
        },
        "api_endpoints": {
            "/analyze": "单股分析",
            "/scan": "批量扫描",
            "/health": "健康检查"
        },
        "timestamp": get_timestamp()
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动智鉴富策略分析系统...")
    logger.info(f"模型: {REQUIRED_MODEL}")
    logger.info(f"目标命中率: {TARGET_HIT_RATE:.1%}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 