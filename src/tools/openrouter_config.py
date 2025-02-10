import os
import time
import logging
import openai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff

# 设置日志记录
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置文件处理器
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
file_handler.setLevel(logging.DEBUG)

# 设置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.debug("Logger initialization completed")
logger.info("API logging system started")

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录并加载环境变量
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 读取 OpenAI 兼容 API 配置
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
model = os.getenv("MODEL", "gemini-1.5-flash")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 API_KEY 环境变量")
    raise ValueError("API_KEY not found in environment variables")

logger.info(f"{SUCCESS_ICON} OpenAI 兼容 API 初始化成功")

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key=api_key, base_url=base_url)


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300
)
def generate_content_with_retry(messages, model=model):
    """带重试机制的 OpenAI 兼容 API 请求"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 OpenAI 兼容 API...")

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        if model is None:
            model = os.getenv("MODEL", "gpt-4")

        logger.info(f"{WAIT_ICON} 使用模型: {model}")

        for attempt in range(max_retries):
            try:
                response = generate_content_with_retry(messages, model)
                return response
            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
