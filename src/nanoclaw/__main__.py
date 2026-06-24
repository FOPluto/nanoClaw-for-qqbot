# __main__.py
# ============
# 程序入口文件。当执行 `python -m nanoclaw` 时，Python 会自动运行此文件。
#
# 启动流程：
#   1. 配置日志格式和级别
#   2. async 初始化：创建目录、初始化数据库、写入默认 CLAUDE.md
#   3. 创建 QQ Bot 客户端
#   4. 连接 QQ WebSocket，开始接收消息

import asyncio     # 异步 IO（用于 _prepare_runtime 中的 async 初始化）
import logging     # 日志（在 main 之前配置日志格式）

# ---- 导入各模块 ----
from nanoclaw.bot import setup_bot                                  # QQ Bot 工厂
from nanoclaw.config import (                                       # 配置
    ASSISTANT_NAME,     # AI 助手名字
    DATA_DIR,           # 数据目录
    DB_PATH,            # SQLite 数据库路径
    KNOWLEDGE_DIR,      # RAG 知识库目录
    QQ_BOT_APP_ID,      # QQ 机器人 AppID
    QQ_BOT_TOKEN,       # QQ 机器人 Token
    STORE_DIR,          # 持久化目录
    WORKSPACE_DIR,      # AI 工作目录
)
from nanoclaw.db import init_db                                     # 数据库初始化
from nanoclaw.memory import ensure_workspace                        # 工作区初始化

# ---- 日志配置 ----
# 在 main() 调用前配置，确保所有模块的日志输出统一格式。
# 格式说明：
#   %(asctime)s  → 2026-06-07 12:30:45,123
#   %(name)s     → nanoclaw.bot / nanoclaw.agent 等，方便定位日志来源
#   %(levelname)s → INFO / WARNING / ERROR / DEBUG
#   %(message)s  → 日志正文
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # 生产环境用 INFO，调试时可改为 DEBUG
)
logger = logging.getLogger(__name__)


# ---- 运行时初始化 ----
# async 初始化：需要等待的操作用 async/await 处理。
# 不放在 main() 的同步部分，因为 init_db 需要异步 SQLite 连接。

async def _prepare_runtime() -> None:
    """在启动 Bot 前准备运行时环境。

    1. 创建必要的目录（不存在则创建）
    2. 初始化 SQLite 数据库（建表）
    3. 初始化 AI 工作区（CLAUDE.md + conversations/）

    所有操作都是幂等的：重复执行不会出错或覆盖数据。
    """
    # 创建目录
    for d in (WORKSPACE_DIR, STORE_DIR, DATA_DIR, KNOWLEDGE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 初始化数据库（建表）
    await init_db(str(DB_PATH))
    logger.info("Database initialized at %s", DB_PATH)

    # 初始化工作区（CLAUDE.md + conversations/）
    ensure_workspace()
    logger.info("Workspace ready at %s", WORKSPACE_DIR)


# ---- 主函数 ----

def main() -> None:
    """程序主入口。

    为什么 async 和 sync 混合？
    - _prepare_runtime() 是 async → 用 asyncio.run() 在同步代码中执行
    - client.run() 是 sync 阻塞调用 → 它内部管理自己的 asyncio 事件循环

    qq-botpy 的 Client.run() 内部：
    1. 创建 asyncio 事件循环
    2. 建立 WebSocket 连接
    3. 接收事件并调用 on_xxx 回调
    4. 阻塞直到退出（Ctrl+C 或连接断开）
    """
    # 阶段 1：异步初始化（创建目录、建表、写文件）
    asyncio.run(_prepare_runtime())

    # 阶段 2：创建 Bot 客户端
    client = setup_bot()

    # 阶段 3：启动 WebSocket 连接（阻塞）
    logger.info("%s QQ Bot is starting...", ASSISTANT_NAME)
    client.run(
        appid=QQ_BOT_APP_ID,   # QQ 开放平台的 AppID（非 QQ 号）
        secret=QQ_BOT_TOKEN,     # 机器人 Token（非密码，是开放平台生成的密钥）
        # client.run() 内部：
        #   → 用 appid+token 换取 access_token
        #   → 建立 WebSocket 连接到 wss://api.sgroup.qq.com/websocket
        #   → 循环接收事件推送
    )

    # 正常退出（Ctrl+C 触发 KeyboardInterrupt，被下面的 except 捕获）


# ---- 入口 ----

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 用户按 Ctrl+C：优雅退出，不打堆栈
        logger.info("Shutting down...")
    # 其他未捕获异常会正常打印 traceback，方便排查问题
