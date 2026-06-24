import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm.contrib import discord

from nanoclaw.config import (
    WORKSPACE_DIR
)

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path(WORKSPACE_DIR) / "conversations"

def ensure_conversations_dir() ->None:
    """
    创建对话记录，如果不存在的话就创建
    :return:
    """
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

def _get_today_file() -> Path:
    """
    获取今天对应的对话文件路径
    文件名示例：conversations/2026-05-05.md
    :return:
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return CONVERSATIONS_DIR / f"{today}.md"

async def archive_exchange(         ## async函数，方便加入异步IO操作
        user_message: str,          ## 用户发来的原始消息
        assistant_response: str,    ## AI回复消息
        chat_id,                    ## 会话id
) -> None:
    """
    将一轮用户 ai对话追加到今天的对话文件中

    文件格式：
    ## HH:MM:SS UTC

    **User**: <用户消息>

    **Ape**: <AI 回复>

    为什么需要写成md格式呢？
    可读性强，可以直接打开文档阅读
    AI的read工具能完整读取
    AI的grep工具能够进行关键词检索
    :param user_message:
    :param assistant_response:
    :param chat_id:
    :return:
    """
    ensure_conversations_dir()

    filepath = _get_today_file()
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    ## 拼接单条对话记录
    entry = f"""
    ## {timestamp}
    **User**: {user_message}`
    **Ape**: {assistant_response}
    
------



"""
    try:
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
        else:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            content = f"# Conversation - {chat_id} - {date_str}\n\n"

        content += entry
        filepath.write_text(content, encoding="utf-8")
        logger.debug(f"Archived exchange to {filepath}")
    except Exception as e:
        ## 归档失败只记录日志
        logger.exception(f"Failed to archive exchange to {filepath}")
