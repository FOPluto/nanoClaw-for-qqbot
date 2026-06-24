import asyncio
import json
import logging
import re
import subprocess
import time

from datetime import datetime, timezone, timedelta
from typing import Any
from pathlib import Path

import httpx
import openai
from numpy import dtype
import logger
from future.backports.urllib import response
from openai import AsyncOpenAI

## duckduckgo 搜索，DDGS是同步上下文管理器
from duckduckgo_search import DDGS

## 表达式解析，计算下次触发的时间
from croniter import croniter
from pyexpat import model
from pymongo.common import MAX_MESSAGE_SIZE
from sqlalchemy.ext.asyncio import result
from torch._inductor.lookup_table import choices

from nanoclaw import db  ## 数据库相关 TODO
from nanoclaw.config import (  ## 配置文件相关
    config,
    CONV_HISTORY_DIR,  ## 对话历史存储的目录
    DATA_DIR,  ## 数据目录
    WORKSPACE_DIR, _MAX_TRY_COUNT,  ## 工作空间目录
)

## 日志文件
logging = logging.getLogger(__name__)


## 只允许一条消息调用Agent，避免互相抢占API导致混乱
_agent_lock = asyncio.Lock()

## Agent loop最大轮数, 防止进入死循环
_MAX_TOOL_ROUND = 20

## DEEPSEEK API客户端（模块级单例）
_client: AsyncOpenAI | None = None

## 通过api key和base url获得模型
def _get_client() -> AsyncOpenAI | None:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url + "/v1",
        )
    return _client

### 系统提示
## 这里的作用就是获得系统的提示词，根据config.name之类的参数定义模型的名字之类的
def _load_system_prompt() -> str:
    claude_md = WORKSPACE_DIR / "CLAUDE.md"
    if claude_md.exists():
        return claude_md.read_text(encoding="utf-8")
    ## 假如用户没有给出相应的系统提示词，就用默认的提示词
    return f"""You are {config.assistant_name}, a personal AI assistant running on QQ.

## Your Capabilities
- Read, write, and edit files in your workspace
- Execute bash commands
- Search the web and fetch web content
- Send messages to the user on QQ via the send_message tool
- Schedule recurring or one-time tasks
- Manage scheduled tasks (list, pause, resume, cancel)

## Task Scheduling
- schedule_type "cron": e.g. "0 9 * * 1" = every Monday 9am
- schedule_type "interval": value in milliseconds
- schedule_type "once": ISO 8601 timestamp

## Memory
- CLAUDE.md is your long-term memory — update it to remember important facts
- conversations/ folder has past chat history — use Grep to search it
- The user's messages and your replies are automatically archived

## Conversation History
Your conversation history is stored in conversations/ folder by date (e.g., 2026-06-07.md).
Use Glob and Grep to search past conversations.
"""


### 工具定义，使用function CALL的形式

## 这里分为系统工具和MCP工具，系统工具只需要传入工作目录就可以work，但是MCP工具需要根据Client本身

# 每个工具是一个dict 需要包含：
## type: 固定是"function"
## function.name：工具的名称（AI生成调用的时候需要使用）
## function.description：功能描述（AI阅读后决定何时调用）
## function.parameters：json Schema格式的参数定义

#### 全局变量
_BUILDIN_TOOLS = [
    # BASH: 执行 shell命令
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash shell command in the AI workspace directory and return stdout+stderr. "
                           "Use this to run code, install packages, manage files, git operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (e.g. 'ls -la', 'python script.py')",
                    },
                },
                "required": ["command"],
            }
        }
    },
    # ---- read：读取文件内容 ----
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from the workspace with line numbers (like cat -n). "
                           "Use this to check file contents before editing.",
            "parameters": {
                "type": "object",
                "properties": {  # 输入的参数
                    "file_path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    # ---- write：写入文件 ----
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file in the workspace. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    # ---- edit：精确字符串替换 ----
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Make a precise string replacement in an existing file. "
                           "Replaces the FIRST occurrence of old_string with new_string. "
                           "The old_string must match exactly, including whitespace and indentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to replace (must match literally)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    # ---- glob：文件名匹配搜索 ----
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g. '*.py', '**/*.md', 'conversations/2026-*.md').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern relative to workspace root",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    # ---- grep：文件内容搜索 ----
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files. Returns matching lines with file names and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (relative to workspace). Default: '.'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    # ---- web_search：网页搜索 ----
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return top results (title + URL + snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # ---- web_fetch：抓取网页内容 ----
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the content of a URL and return the text. Use this to read documentation or articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (https://...)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


#### MCP工具：send_message和任务管理，他们在loop中动态生成
## 因为需要持有bot实例和chat_id等运行时上下文

### 这里使用的是Function Calling的形式，后续可以实现成MCP的格式
#TODO

_MCP_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to the user on QQ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The message text to send to the user",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a task. schedule_type: 'cron' (e.g. '0 9 * * 1'), "
                           "'interval' (milliseconds), or 'once' (ISO 8601 timestamp).",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Task description/prompt for the AI to execute"},
                    "schedule_type": {"type": "string", "description": "cron, interval, or once"},
                    "schedule_value": {"type": "string", "description": "Cron expression, milliseconds, or ISO timestamp"},
                },
                "required": ["prompt", "schedule_type", "schedule_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all scheduled tasks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_task",
            "description": "Pause a scheduled task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "Task ID to pause"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_task",
            "description": "Resume a paused scheduled task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "Task ID to resume"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "Cancel and delete a scheduled task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "Task ID to cancel"}},
                "required": ["task_id"],
            },
        },
    },
    # ---- search_knowledge：RAG 知识库检索 ----
    # 不加在 _BUILTIN_TOOLS 而加在 _MCP_TOOL_DEFS 的原因是：
    # 它需要运行时上下文（RAG 引擎的延迟初始化在工具实现内部完成），
    # 与 send_message 等工具同属"有外部依赖"的工具组。
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search your personal knowledge base for documents relevant to the user's question. "
                           "Use this when the user asks about something that might be in their documents. "
                           "Returns the top matching text chunks with source file and relevance scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. 'Python decorator usage')",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

## 全部工具的合并列表（buildin + mcp）
ALL_TOOL_DEFS = _BUILDIN_TOOLS + _MCP_TOOL_DEFS

## 实际上两个是合在一起用的，只是这样写比较有可读性
## deepseek和其他比如说OPENAI的模型，能够兼容function CALL的形式，但是CLAUDE模型只能接受MCP格式

## 工具实现
## 上面定义的BUILDIN工具

## 路径安全检查
### 所有文件操作的工作都必须通过这个检查，防止AI通过路径遍历
### 防止模型访问到workspace之外的文件，造成不必要的损失
def _resolve_safe(file_path: str) -> Path:
    """
    将用户输入的路径解析到workspace内，并拒绝越权访问，
    原理：
    1, 如果路径是相对路径 -> 拼接到workspace下
    2, 如果路径是绝对路径 -> 直接resove
    3, 检查最终目录是否在workspace内，不在的话则报错
    :param file_path: 用户输入的文件或者目录路径
    :return: 返回resolve之后或者拼接之后的路径，函数将用户输入的路径返回成对应的绝对路径
    :exception: 如果用户输入的绝对路径不在workspace下，则会抛出ValueError异常
    """
    p = Path(file_path)
    if not p.is_absolute(): ## 如果不是绝对路径
        p = WORKSPACE_DIR / p
    resolved = p.resolve()
    workspace_root = WORKSPACE_DIR.resolve() ## 解析成绝对路径
    if not str(resolved).startswith(str(workspace_root)):
        raise ValueError(f"Access denied: {file_path} is outside the workspace")  ## 抛出异常
    return resolved

## bash工具
async def _bash(command: str) -> str:
    """
    执行shell命令并返回stdout + stderr

    用asyncio.to_thread把subprocess.run 包装到线程池中执行
    subprocess.run是同步阻塞调用，如果在asyncio主线程中调用会
    直接阻塞整个时间的循环，导致bot在此期间无法相应其他消息。这就需要
/；    timeout = 120 限制命令最长执行时间是两分钟，防止AI执行死循环操作
    :param command: 输入的命令
    :return: 返回stdout和stderr
    """
    try:
        process = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,                 # 支持管道、重定向等shell语法
            capture_output=True,        # 捕获 stdout 和 stderr
            text=True,                  # 返回自负床而非bytes
            cwd=str(WORKSPACE_DIR),     # 在workspace目录下执行
            timeout=120,                # 超时两分钟
        )
        output = process.stdout
        if process.stderr:
            output += "\n[stderr]\n" + process.stderr
        if not output.strip():
            output += f"Exit Code:  {process.returncode}"
    except subprocess.TimeoutExpired:
        return "Error: Commend time out after 120 seconds"

## read工具
async def _read(file_path: str) -> str:
    """读取文件并添加行号（类似 cat -n）。

    行号方便 AI 在 Edit 工具中精确指出要替换的位置。
    返回的每一行都带有行号，和行内的内容
    """
    full_path = _resolve_safe(file_path)
    if not full_path.exists():
        return f"Error: File not found: {file_path}"
    if full_path.is_dir():
        return f"Error: {file_path} is a directory, not a file"
    contents = full_path.read_text(encoding="utf-8", errors="replace")
    ## 添加行号
    lines = contents.splitlines()
    return "\n".join(f"{i+1}\t{line}" for i, line in enumerate(lines))

# ---- write 工具 ----

async def _write(file_path: str, content: str) -> str:
    """创建或覆盖文件。自动创建父目录。"""
    full_path = _resolve_safe(file_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return f"File written: {full_path.relative_to(WORKSPACE_DIR)}"


# ---- edit 工具 ----

async def _edit(file_path: str, old_string: str, new_string: str) -> str:
    """精确字符串替换：将文件中 old_string 的第一次出现替换为 new_string。

    这是 Claude Code 的标志性功能——相比 Write（重写整个文件），
    Edit 只需传要改的那几行，大幅度减少 token 消耗。
    替换规则：
    - old_string 必须逐字符精确匹配（包括空格、缩进、换行）
    - 只替换第一次出现
    - 如果 old_string 出现多次会提醒 AI
    """
    full_path = _resolve_safe(file_path)
    if not full_path.exists():
        return f"Error: File not found: {file_path}"
    content = full_path.read_text(encoding="utf-8")

    count = content.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {file_path}. Check whitespace, indentation, and line endings."
    if count > 1:
        # 提醒 AI 有重复，它可能需要更精确的上下文来唯一确定位置
        note = f"Warning: old_string appears {count} times. Replacing only the first occurrence.\n"
    else:
        note = ""

    new_content = content.replace(old_string, new_string, 1)
    full_path.write_text(new_content, encoding="utf-8")
    return f"{note}File edited: {full_path.relative_to(WORKSPACE_DIR)}"


# ---- glob 工具 ----

async def _glob(pattern: str) -> str:
    """查找匹配 glob 模式的文件列表。

    示例：
      pattern="*.py"      → 所有 Python 文件
      pattern="**/*.md"   → 递归查找所有 Markdown 文件
      pattern="conversations/2026-*" → 2026 年的对话文件
    """
    try:
        matches = sorted(WORKSPACE_DIR.glob(pattern))
    except Exception as e:
        return f"Error: Invalid glob pattern: {e}"

    if not matches:
        return "No matches found."

    lines = []
    for m in matches[:100]:  # 最多 100 个结果，防止输出过长
        rel = m.relative_to(WORKSPACE_DIR)
        suffix = "/" if m.is_dir() else ""
        lines.append(str(rel) + suffix)
    result = "\n".join(lines)
    if len(matches) > 100:
        result += f"\n... and {len(matches) - 100} more (truncated)"
    return result


# ---- grep 工具 ----

async def _grep(pattern: str, path: str = ".") -> str:
    """在文件中搜索正则表达式，返回匹配行及文件名和行号。

    如果 path 是文件 → 搜索该文件
    如果 path 是目录 → 递归搜索所有文本文件
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    search_path = _resolve_safe(path)
    results: list[str] = []

    if search_path.is_file():
        files = [search_path]
    else:
        # 递归搜索所有文件，跳过二进制文件、隐藏目录和 .git
        files = [
            p for p in search_path.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.parts if part != ".")
            and ".git" not in p.parts
        ]

    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue  # 跳过无法读取的文件

        for line_no, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                rel_path = fp.relative_to(WORKSPACE_DIR)
                results.append(f"{rel_path}:{line_no}: {line}")

    if not results:
        return "No matches found."
    if len(results) > 200:
        results = results[:200]
        results.append("... (truncated at 200 matches)")
    return "\n".join(results)


# ---- web_search 工具 ----

async def _web_search(query: str) -> str:
    """在 DuckDuckGo 上搜索，返回前 8 条结果。

    用 asyncio.to_thread 包装同步的 DDGS 调用，避免阻塞事件循环。
    """
    try:
        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=8))

        results = await asyncio.to_thread(_sync_search)

        if not results:
            return "No search results found."

        lines = []
        for r in results:
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"- **{title}**\n  URL: {href}\n  {body}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error during web search: {e}"


# ---- web_fetch 工具 ----

async def _web_fetch(url: str) -> str:
    """抓取网页内容并返回文本。

    用 httpx 发起异步 HTTP GET 请求，支持自动跟随重定向。
    结果截断到 8000 字符，防止网页过长撑爆上下文窗口。
    """
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; nanoclaw-py bot)"
                },
            )
            response.raise_for_status()
            # 取前 8000 字符，网页可能非常大
            text = response.text[:8000]
            return text if text.strip() else "(empty page)"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ---- search_knowledge 工具 ----

async def _search_knowledge(query: str) -> str:
    """在个人知识库中执行语义检索。

    流程：query → embedding → ChromaDB 检索 top-K → 格式化返回。

    为什么用 asyncio.to_thread？
    SentenceTransformer.encode() 是同步 PyTorch 调用，必须放到线程池，
    否则会阻塞事件循环导致 Bot 无响应。

    为什么 lazy import？
    如果用户没装 chromadb / sentence-transformers，import 会在运行时报错，
    而不是阻塞整个 agent 模块的加载。
    """
    try:
        from nanoclaw.rag import get_rag
        rag = get_rag()
    except ImportError as e:
        return f"Error: RAG module unavailable — {e}. Install with: pip install chromadb sentence-transformers"
    except Exception as e:
        return f"Error: Failed to load RAG engine — {e}"

    try:
        chunks = await asyncio.to_thread(rag.query, query)
    except Exception as e:
        return f"Error: RAG query failed — {e}"

    if not chunks:
        return "No relevant information found in the knowledge base. Add documents to the knowledge/ directory and run /ingest."

    # 格式化检索结果，标准 RAG prompt 注入格式
    result = f"Found {len(chunks)} relevant passage(s) from your knowledge base:\n\n"
    result += "\n\n".join(chunks)
    result += "\n\n---\nUse this information to answer the user's question. "
    result += "If the retrieved passages don't fully answer the question, "
    result += "say what you found and note what's missing."
    return result

## 对话历史管理
### 用服务端的sessionid管理会话
### deepseek不能存储相关的历史记录，必须自己把完整的message列表存在本地
## 存储格式："data/conversations/{chat_id}.json"
# [{openai-format message}, {openai-format message}, ...]
# 每次对话只保留最近的30条消息，超过的自动丢弃
# deepseek v3对话窗口64k tokens，30条消息在安全范围内

_MAX_MESSAGE = 30  ## 单次对话最多保留的历史消息数量

## 该函数能够获得指定chatid下的对话历史记录文件路径
def _history_file(chat_id: str) -> Path:
    """获取指定 chat_id 的对话历史文件路径。"""
    return CONV_HISTORY_DIR / f"{chat_id}.json"



def _load_history(chat_id: str) -> list[dict]:
    """
    加载当前chatid下的对话历史记录

    返回openai格式的messages列表（这个不包含system prompt）

    :param chat_id:
    :param str:
    :return:
    """
    f = _history_file(chat_id)
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                ## 加入读出来的data是一个list类型，代表除了上一次还有更早的历史
                ## 只取最近的消息
                return data[:]
        except (json.decoder.JSONDecodeError, KeyError):
            logging.warning("Corrupted history file for %s, resetting", chat_id)
    return []  ## 没有相关历史就返回空


def _save_history(chat_id: str, message: list[dict]) -> None:
    CONV_HISTORY_DIR.mkdir(parents=True, exist_ok=True) ## 确保历史记录根目录存在
    trimmed = message[-MAX_MESSAGE_SIZE:]  ## 再次截断，只取最近的

    temp_index = message.index(trimmed[0])

    while trimmed and temp_index >= 0 and trimmed[0]["role"] == "tool":
        temp = message[temp_index]
        trimmed = temp + trimmed
        temp_index -= 1

    _history_file(chat_id).write_text(
        json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def clear_session_id() -> None:
    """清除所有对话历史（对应 /clear 命令）。"""
    if CONV_HISTORY_DIR.exists():
        for f in CONV_HISTORY_DIR.glob("*.json"):
            f.unlink()  # 删除所有 md 文件
        logging.info("All conversation histories cleared")


## 工具调度器

## 当AI返回tool calls的时候，工具调度器需要负责
# 1. 根据function。name找到对应的函数实现
# 2. 解析json arguments -> 传给实现函数
# 3. 捕获异常，返回错误消息，让AI看到错误并重试


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    bot: Any,
    chat_id: str,
    db_path: str,
    notify_state: dict[str, bool] | None,
) -> str:
    """执行单个工具调用并返回结果字符串。"""
    try:
        # ---- 内置工具 ----
        if name == "bash":
            return await _bash(arguments["command"])

        elif name == "read":
            return await _read(arguments["file_path"])

        elif name == "write":
            return await _write(arguments["file_path"], arguments["content"])

        elif name == "edit":
            return await _edit(arguments["file_path"], arguments["old_string"], arguments["new_string"])

        elif name == "glob":
            return await _glob(arguments["pattern"])

        elif name == "grep":
            return await _grep(arguments["pattern"], arguments.get("path", "."))

        elif name == "web_search":
            return await _web_search(arguments["query"])

        elif name == "web_fetch":
            return await _web_fetch(arguments["url"])

        # ---- MCP 工具（需要 bot 上下文） ----
        elif name == "send_message":
            await bot.send_message_raw(openid=chat_id, text=arguments["text"])
            if notify_state is not None:
                notify_state["sent"] = True
            return "Message sent."

        elif name == "schedule_task":
            stype = arguments["schedule_type"]
            svalue = arguments["schedule_value"]
            now = datetime.now(timezone.utc)

            if stype == "cron":
                next_run = croniter(svalue, now).get_next(datetime).isoformat()
            elif stype == "interval":
                next_run = (now + timedelta(milliseconds=int(svalue))).isoformat()
            elif stype == "once":
                next_run = svalue
            else:
                return f"Error: Unknown schedule_type: {stype}"

            task_id = await db.create_task(db_path, chat_id, arguments["prompt"], stype, svalue, next_run)
            return f"Task {task_id} scheduled. Next run: {next_run}"

        elif name == "list_tasks":
            tasks = await db.get_all_tasks(db_path)
            if not tasks:
                return "No scheduled tasks."
            lines = [
                f"- [{t['id']}] {t['status']} | {t['schedule_type']}({t['schedule_value']}) | {t['prompt'][:60]}"
                for t in tasks
            ]
            return "\n".join(lines)

        elif name == "pause_task":
            ok = await db.update_task_status(db_path, arguments["task_id"], "paused")
            return f"Task {arguments['task_id']} paused." if ok else f"Task {arguments['task_id']} not found."

        elif name == "resume_task":
            ok = await db.update_task_status(db_path, arguments["task_id"], "active")
            return f"Task {arguments['task_id']} resumed." if ok else f"Task {arguments['task_id']} not found."

        elif name == "cancel_task":
            ok = await db.delete_task(db_path, arguments["task_id"])
            return f"Task {arguments['task_id']} cancelled." if ok else f"Task {arguments['task_id']} not found."

        elif name == "search_knowledge":
            return await _search_knowledge(arguments["query"])

        else:
            return f"Error: Unknown tool: {name}"

    except Exception as e:
        logging.exception("Tool %s failed", name)
        return f"Error executing {name}: {e}"


async def _call_api_try_function(
        model: str,
        message: str,
        tools: list[dict[str, Any]],
        temp: float, ## 间隔多少秒重试一次
        client: Any
):
    for i in range(_MAX_TRY_COUNT):  ## 这里是由于网络波动，可能尝试连接失败
        try:
            return await client.chat.completions.create(
                model=model,
                messages=message,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=4096,
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            if i == _MAX_TRY_COUNT - 1:
                raise f"Try Time Out: {e}"
            await asyncio.sleep(temp * (1 + i))




## 最后就是Agent loop
## 核心编排逻辑

async def _agent_loop(
        message: list[dict],
        bot: Any,
        chat_id: str,
        db_path: str,
        notify_state: dict[str, bool] | None,
) -> str:
    """
    Agent Loop: 反复调用deepseek api，直到ai不再要求调用工具
    整体流程：
    while 轮数 < 上限:
        1. 调用DEEPSEEK api（带tool定义的）
        2. 检查响应：
            a. 如果有tool_call -> 逐个执行 -> 结果追加到message -> continue
            b. 如果没有tool_call -> 返回文本内容 -> break
        3. 如果有content + tool_calls的情况，就先保存content，然后再执行tools
    :param message:
    :param bot:
    :param chat_id:
    :param db_path:
    :param notify_state:
    :return:
    """
    client = _get_client()
    accumulated_text: list[str] = []  ## 收集所有中途的文本片段

    for turn in range(_MAX_TOOL_ROUND):
        logging.info(f"[model]: {config.deepseek_model}")
        response = await _call_api_try_function(
            model=config.deepseek_model,
            message=message,
            tools=ALL_TOOL_DEFS,
            temp=1.0,
            client=client,
        )
        choice = response.choices[0]
        msg = choice.message

        # 收集文本内容（可能和 tool_calls 同时存在）
        if msg.content:
            accumulated_text.append(msg.content)

        # 检查是否有工具调用
        if msg.tool_calls:
            # 1. 把 AI 的回复（含 tool_calls）添加到 messages
            #    model_dump 把 Pydantic 对象转为 dict，exclude_none 去掉空字段
            assistant_msg = msg.model_dump(exclude_none=True)
            # 确保没有 weight 字段（某些 API 会抛 validation error）
            assistant_msg.pop("weight", None)
            message.append(assistant_msg)

            # 2. 执行每个工具调用，结果追加到 messages
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                logging.info("Tool call: %s(%s)", func_name, str(func_args)[:100])
                result = await _execute_tool(func_name, func_args, bot, chat_id, db_path, notify_state)

                # 把工具执行结果作为 tool role 消息追加
                message.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result if result is not None else "",
                })

            # 3. 继续循环，让 AI 看到工具结果后决定下一步
            continue

        # 没有 tool_calls → AI 认为对话可以结束了
        break

    else:
        # while 循环自然结束（没用 break）→ 超出最大轮数
        accumulated_text.append("\n\n[Maximum tool call rounds reached. Stopping.]")

    return "".join(accumulated_text) or "Done."



async def run_agent(prompt: str, bot: Any, chat_id: str, db_path: str) -> str:
    async with asyncio.Lock():
        return await _run_agent_inner(prompt, bot, chat_id, db_path)


async def _run_agent_inner(prompt: str, bot: Any, chat_id: str, db_path: str) -> str:
    """Agent 内部实现（调用方已经持锁）。"""

    # 1. 加载系统提示
    system_prompt = _load_system_prompt()

    # 2. 加载历史对话
    history = _load_history(chat_id)

    # 3. 构建 messages 列表
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": prompt},
    ]

    # 4. 执行 Agent Loop
    result = await _agent_loop(messages, bot, chat_id, db_path, None)

    # 5. 保存对话历史（system prompt 不存，只存 user + assistant + tool 消息）
    #    把 system 消息去掉后保存
    _save_history(chat_id, messages[1:])

    return result




async def run_task_agent(
    prompt: str,
    bot: Any,
    chat_id: str,
    db_path: str,
    notify_state: dict[str, bool] | None = None,
) -> str:
    """执行定时任务 Agent（无历史、无锁、未持久化会话）。

    与 run_agent 的区别：
    - 不加锁（scheduler 自然地串行调用）
    - 不加载/保存对话历史（每次任务独立上下文）
    - 不透传 notify_state（用于检测 AI 是否调了 send_message）
    """
    system_prompt = _load_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    return await _agent_loop(messages, bot, chat_id, db_path, notify_state)
