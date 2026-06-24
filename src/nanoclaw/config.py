import os
from pathlib import Path
from dotenv import load_dotenv

from dataclasses import dataclass, field

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
STORE_DIR = BASE_DIR / "store"
DATA_DIR = BASE_DIR / "data"
## 数据库文件
DB_PATH = STORE_DIR / "nanoclaw.db"
## 状态
STATE_FILE = DATA_DIR / "state.json"
## 对话历史持久化目录
CONV_HISTORY_DIR = BASE_DIR / "conversations"
## RAG知识库：放txt或者md文档
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
## chromaDB 向量数据库存储目录
CHROMA_DB_PATH = STORE_DIR / "chroma"

## 如果api链接超时，尝试三次
_MAX_TRY_COUNT = 3

@dataclass
class Config:
    """ 项目配置，从环境变量中读取并校验 """
    ## 待会还需要回去设置一下环境变量 TODO

    ## 必须的字段：
    qq_bot_app_id: str = field(
        default_factory=lambda: os.getenv("QQ_BOT_APP_ID")
    )
    qq_bot_token: str = field(
        default_factory= lambda: os.getenv("QQ_BOT_TOKEN")
    )
    owner_openid: str = field(
        default_factory= lambda: os.getenv("OWNER_OPENID")
    )
    deepseek_api_key: str = field(
        default_factory= lambda: os.getenv("DEEPSEEK_API_KEY")
    )

    ## 可选字段
    deepseek_base_url: str = field(
        default_factory= lambda: os.getenv("DEEPSEEK_BASE_URL")
    )
    deepseek_model: str = field(
        default_factory= lambda: os.getenv("DEEPSEEK_MODEL")
    )
    assistant_name: str = field(
        default_factory=lambda: os.getenv("ASSISTANT_NAME", "Ape")
    )
    scheduler_interval: int = field(
        default_factory=lambda: int(os.getenv("SCHEDULER_INTERVAL", "60"))
    )

    # Embedding 模型名称。支持 HuggingFace 上的 sentence-transformers 模型。
    rag_embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )

config = Config()


## 模块级别的别名
## 可以直接通过from nanoclaw.config import NAME 这样跨模块调用
QQ_BOT_APP_ID = config.qq_bot_app_id
QQ_BOT_TOKEN = config.qq_bot_token
OWNER_OPENID = config.owner_openid
DEEPSEEK_API_KEY = config.deepseek_api_key
ASSISTANT_NAME = config.assistant_name
SCHEDULER_INTERVAL = config.scheduler_interval

## 单用户模式
SINGLE_USER = True

def get_chat_workspace(chat_id: str) -> Path:
    """
    获取当前chatid下的workspace

    当前的是single-user模式，共享一个workspace

    Example future structure:
        workspace/
        └── chats/
            ├── 123456/       # user chat
            │   ├── CLAUDE.md
            │   └── conversations/
            └── -987654/      # group chat (negative ID)
                ├── CLAUDE.md
                └── conversations/
    """
    if SINGLE_USER:
        return WORKSPACE_DIR
    else:
        #TODO
        pass
