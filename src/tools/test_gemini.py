import os
import json
import openai
from dotenv import load_dotenv

# 加载环境变量
env_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)

# 获取配置
api_key = os.getenv("API_KEY")
base_url = os.getenv(
    "BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)


def test_simple_prompt():
    """测试简单的提示词生成"""
    print(f"Using model: {model_name}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Write a story about a magic backpack."}]
    )

    print("\nSimple prompt response:")
    print("Response type:", type(response))
    print("\nResponse text:", response.choices[0].message.content)

    # 打印完整的响应对象结构
    print("\nFull response structure:")
    print(json.dumps(response.model_dump(), indent=2))


def test_chat_format():
    """测试聊天格式的提示词"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 1+1?"}
        ],
        temperature=0.3
    )

    print("\nChat format response:")
    print("Response type:", type(response))
    print("\nResponse text:", response.choices[0].message.content)


if __name__ == "__main__":
    print("Testing OpenAI Compatible API...")
    test_simple_prompt()
    print("\n" + "="*50 + "\n")
    test_chat_format()
