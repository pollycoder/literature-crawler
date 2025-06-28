#!/usr/bin/env python3
"""
快速配置代理脚本
用于快速设置第三方代理配置
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """创建.env配置文件"""
    print("🔧 AI文献爬取助手 - 代理配置向导")
    print("=" * 50)
    
    # 检查是否已存在.env文件
    env_file = Path(".env")
    if env_file.exists():
        response = input(".env文件已存在，是否覆盖？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("配置取消")
            return
    
    print("\n📋 请输入您的代理配置信息：")
    print("💡 提示：如果使用统一代理（如OneAPI），所有base_url可以相同")
    
    # OpenAI代理配置（必须）
    print("\n🔑 OpenAI代理配置（必须）：")
    openai_api_key = input("OpenAI代理API密钥: ").strip()
    openai_base_url = input("OpenAI代理Base URL (如: https://api.example.com/v1): ").strip()
    
    if not openai_api_key or not openai_base_url:
        print("OpenAI代理配置是必须的！")
        return
    
    # 其他代理配置（可选）
    configs = []
    configs.append(f"# OpenAI代理配置（必须）")
    configs.append(f"OPENAI_PROXY_API_KEY={openai_api_key}")
    configs.append(f"OPENAI_PROXY_BASE_URL={openai_base_url}")
    configs.append("")
    
    # Claude代理
    print("\n🤖 Claude代理配置（可选，回车跳过）：")
    claude_api_key = input("Claude代理API密钥: ").strip()
    claude_base_url = input(f"Claude代理Base URL (默认: {openai_base_url}): ").strip()
    
    if claude_api_key:
        if not claude_base_url:
            claude_base_url = openai_base_url
        configs.append(f"# Claude代理配置")
        configs.append(f"CLAUDE_PROXY_API_KEY={claude_api_key}")
        configs.append(f"CLAUDE_PROXY_BASE_URL={claude_base_url}")
        configs.append("")
    
    # Gemini代理
    print("\n🌟 Gemini代理配置（可选，回车跳过）：")
    gemini_api_key = input("Gemini代理API密钥: ").strip()
    gemini_base_url = input(f"Gemini代理Base URL (默认: {openai_base_url}): ").strip()
    
    if gemini_api_key:
        if not gemini_base_url:
            gemini_base_url = openai_base_url
        configs.append(f"# Gemini代理配置")
        configs.append(f"GEMINI_PROXY_API_KEY={gemini_api_key}")
        configs.append(f"GEMINI_PROXY_BASE_URL={gemini_base_url}")
        configs.append("")
    
    # DeepSeek代理
    print("\n🚀 DeepSeek代理配置（可选，回车跳过）：")
    deepseek_api_key = input("DeepSeek代理API密钥: ").strip()
    deepseek_base_url = input(f"DeepSeek代理Base URL (默认: {openai_base_url}): ").strip()
    
    if deepseek_api_key:
        if not deepseek_base_url:
            deepseek_base_url = openai_base_url
        configs.append(f"# DeepSeek代理配置")
        configs.append(f"DEEPSEEK_PROXY_API_KEY={deepseek_api_key}")
        configs.append(f"DEEPSEEK_PROXY_BASE_URL={deepseek_base_url}")
        configs.append("")
    
    # 写入.env文件
    try:
        with open(".env", "w", encoding="utf-8") as f:
            f.write("# AI文献爬取助手 - 代理配置\n")
            f.write("# 由setup_proxy.py自动生成\n")
            f.write("\n")
            f.write("\n".join(configs))
        
        print("\n配置文件已创建：.env")
        
        # 测试配置
        print("\n🔧 测试配置...")
        test_result = os.system("python main.py test-models")
        
        if test_result == 0:
            print("\n🎉 配置完成！现在可以开始使用文献爬取工具了！")
            print("\n💡 使用示例：")
            print("   python main.py search \"深度学习在图像识别中的应用\" --max-papers 10")
            print("   python main.py list-models")
        else:
            print("\n配置测试失败，请检查API密钥和Base URL是否正确")
            
    except Exception as e:
        print(f"创建配置文件失败：{e}")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("AI文献爬取助手 - 代理配置向导")
        print("\n用法：")
        print("  python setup_proxy.py    # 交互式配置")
        print("  python setup_proxy.py -h # 显示帮助")
        return
    
    # 检查是否在正确的目录
    if not Path("main.py").exists():
        print("请在项目根目录运行此脚本")
        return
    
    create_env_file()

if __name__ == "__main__":
    main()