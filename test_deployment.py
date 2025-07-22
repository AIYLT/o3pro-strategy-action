#!/usr/bin/env python3
"""
快速测试Railway部署状态
"""
import requests
import time

URL = "https://web-production-2170f.up.railway.app"

def test_deployment():
    print("🚀 测试Railway部署状态...")
    print(f"🔗 URL: {URL}")
    print("-" * 50)
    
    try:
        # 测试健康检查
        response = requests.get(f"{URL}/health", timeout=10)
        print(f"📊 状态码: {response.status_code}")
        print(f"📝 响应: {response.text}")
        
        if response.status_code == 200:
            print("✅ 部署成功！")
            return True
        elif response.status_code == 404:
            print("❌ 404错误 - 应用未启动或环境变量未设置")
        else:
            print(f"⚠️ 其他错误: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 连接错误: {e}")
    
    return False

if __name__ == "__main__":
    test_deployment()
    
    print("\n" + "="*60)
    print("🔧 如果还是404，请立即检查：")
    print("1. 访问 https://railway.app")
    print("2. 进入您的项目")
    print("3. 点击 Variables 标签")
    print("4. 确保设置了这4个环境变量：")
    print("   - OPENAI_API_KEY")
    print("   - POLYGON_API_KEY")
    print("   - ENVIRONMENT=production")
    print("   - LOG_LEVEL=INFO")
    print("5. 查看 Logs 标签的错误信息")
    print("6. 确认 Deployments 标签显示构建成功")
    print("="*60) 