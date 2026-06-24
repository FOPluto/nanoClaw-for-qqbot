# bot.py
# =======
# QQ Bot 消息处理模块：连接 QQ 开放平台，接收消息，调度 AI Agent 回复。
#
# QQ 开放平台的消息类型：
#   C2C 消息（Consumer to Consumer）→ 用户私聊机器人
#   群聊 @ 消息                    → 群内 @ 机器人
#
# QQ 的身份体系：openid
#   不同于 QQ 号，openid 是平台分配给每个(用户×机器人)组合的唯一标识符，
#   目的是保护用户隐私——同一个用户在不同机器人那里看到的是不同的 openid。
#
# 通信方式：WebSocket（长连接）
#   qq-botpy 库封装了 WebSocket 连接管理，我们只需继承 botpy.Client
#   并覆写 on_xxx 事件回调即可。底层自动处理重连、心跳、序列化等。
import asyncio
import logging    # 日志
import re         # 正则表达式（提取命令、去除 @ 前缀）

# botpy：QQ 官方 Python SDK。PyPI 包名是 qq-botpy，import 名是 botpy。
import botpy

# C2CMessage: 私聊消息对象（.author.id=openid, .content=文本, .id=消息ID）
# GroupMessage: 群聊消息对象（.group_openid=群ID, .author.id=发送者openid）
from botpy.message import C2CMessage, GroupMessage

# 项目内部模块
from nanoclaw.agent import run_agent, clear_session_id       # AI Agent
from nanoclaw.conversations import archive_exchange           # 对话归档
from nanoclaw.config import ASSISTANT_NAME, DB_PATH, OWNER_OPENID, KNOWLEDGE_DIR  # 配置
from nanoclaw.scheduler import setup_scheduler                # 定时任务调度器

logger = logging.getLogger(__name__)

# QQ 单条文本消息最大长度（字符数）。超过需拆分为多条消息发送。
_QQ_MAX_LENGTH = 2000

# 全局变量：持有 QQBotClient 实例，供 scheduler 等外部模块访问。
_bot_client: "QQBotClient | None" = None
_scheduler_started: bool = False  # 防止重复启动调度器


def _is_owner(openid: str) -> bool:
    """判断消息发送者是否为 Bot Owner（你自己）。

    只有 Owner 能使用这个 Bot，其他人发消息会被静默忽略。
    这保证了 Bot 只为你一个人服务，避免被其他人滥用消耗 API 额度。
    """


    return openid == OWNER_OPENID


def _extract_command(text: str) -> str | None:
    """从消息文本中提取命令。

    输入示例：
      "/clear"           → "/clear"
      "/clear@bot_xxx"   → "/clear"    （QQ 群聊中命令可能带 @ 后缀）
      "hello world"      → None        （不是命令）

    正则 r"(/\w+)" 解释：
      /     → 匹配斜杠
      \w+   → 匹配一个或多个字母/数字/下划线
    """
    m = re.match(r"(/\w+)", text)
    return m.group(1) if m else None


class QQBotClient(botpy.Client):
    """QQ Bot 客户端，继承 botpy.Client，覆写事件回调方法。"""

    # ---- 生命周期回调 ----

    async def on_ready(self):
        """WebSocket 连接成功后触发（只触发一次）。

        在这里启动定时任务调度器，确保 bot 就绪后才开始检查到期任务。
        """
        global _bot_client, _scheduler_started
        _bot_client = self

        # self.robot 是 botpy 提供的机器人信息对象，包含 name、id 等字段
        logger.info(f"{ASSISTANT_NAME} QQ Bot is online as {self.robot.name}")

        if not _scheduler_started:
            _scheduler_started = True
            # 创建并启动定时任务调度器
            scheduler = setup_scheduler(self, str(DB_PATH))
            scheduler.start()
            logger.info("Scheduler started")

    # ---- 消息回调 ----

    async def on_c2c_message_create(self, message: C2CMessage):
        """收到 C2C 私聊消息时触发。

        C2CMessage 重要字段：
          message.author.id    → 发送者 openid（如 "A1B2C3D4E5F6..."）
          message.content      → 消息文本
          message.id           → 消息 ID（用于回复引用）
        """

        print(message.author)
        await self._handle_message(
            chat_id=message.author.user_openid,                     # 私聊：对话 ID = 发送者的 openid
            sender_openid=message.author.user_openid,               # 发送者 = 用户
            text=message.content.strip(),
            message_id=message.id,
            is_group=False,
        )

    async def on_group_at_message_create(self, message: GroupMessage):
        """收到群聊 @ 机器人消息时触发。

        GroupMessage 重要字段：
          message.group_openid  → 群 openid
          message.author.id     → 发送者 openid
          message.content       → 消息文本（以 <@!bot_id> 开头）

        注意：群聊消息一定有 @ 前缀（QQ 只在有人 @ 机器人时才推送消息）。
        需要先把 <@!数字> 前缀去除，才能得到用户的真实输入。
        """
        # 群内先做 owner 校验，非 owner 直接丢弃，避免群友滥用
        if not _is_owner(message.author.id):
            return

        # 去除消息开头的 <@!123456789> 格式的 @ 提醒标记
        text = re.sub(r"<@!\d+>\s*", "", message.content.strip())
        if not text:
            return

        await self._handle_message(
            chat_id=message.group_openid,       # 群聊：对话 ID = 群的 openid
            sender_openid=message.author.id,    # 发送者（用于权限校验）
            text=text,
            message_id=message.id,
            is_group=True,
        )

    # ---- 核心消息处理 ----

    async def _handle_message(
        self,
        chat_id: str,          # 消息来源 ID（私聊=用户openid，群聊=群openid）
        sender_openid: str,    # 发送者 openid（用于权限校验）
        text: str,             # 用户输入的文本
        message_id: str,       # 消息 ID（用于回复引用）
        is_group: bool,        # 是否群聊
    ):
        """统一的消息处理入口。"""
        # 权限校验：只处理 Owner 的消息
        if not _is_owner(sender_openid):
            return

        # 检查是否为内置命令
        cmd = _extract_command(text)

        if cmd == "/start":
            await self._send_message(
                chat_id,
                f"Hi! I'm {ASSISTANT_NAME}, your personal AI assistant. "
                "Send me a message to get started.\n\n"
                "Commands:\n"
                "/clear - Reset conversation session\n"
                "/ingest - Index documents from knowledge/ folder (RAG)",

                is_group,
                message_id,
            )
            return

        if cmd == "/clear":
            clear_session_id()
            await self._send_message(
                chat_id,
                "Session cleared. Starting fresh!",
                is_group,
                message_id,
            )
            return

        if cmd == "/ingest":  #TODO
            await self._send_message(chat_id, "Ingesting documents from knowledge/ ...", is_group, message_id)
            try:
                from nanoclaw.rag import get_rag
                rag = get_rag()
                stats = rag.ingest_directory()
                if not stats:
                    await self._send_message(
                        chat_id,
                        f"No supported files found in {KNOWLEDGE_DIR}.\n"
                        "Supported: .txt .md. Put documents there and /ingest again.",
                        is_group,
                        None,
                    )
                else:
                    total = sum(stats.values())
                    summary = "\n".join(f"  {name}: {count} chunks" for name, count in stats.items())
                    await self._send_message(
                        chat_id,
                        f"Ingestion complete. {total} chunks from {len(stats)} file(s):\n{summary}\n\n"
                        "AI can now search this knowledge with the search_knowledge tool.",
                        is_group,
                        None,
                    )
            except ImportError as e:
                await self._send_message(
                    chat_id,
                    f"RAG dependencies not installed: {e}\nRun: pip install chromadb sentence-transformers",
                    is_group,
                    None,
                )
            except Exception as e:
                logger.exception("/ingest failed")
                await self._send_message(chat_id, f"Ingestion failed: {e}", is_group, None)
            return

        # 非命令消息 → 调用 AI Agent 处理
        logger.info("[Call Agent]\n")
        response = await run_agent(text, self, chat_id, str(DB_PATH))

        # 归档本轮对话到 workspace/conversations/  这里是给用户可视化用的
        await archive_exchange(text, response, chat_id)

        # QQ 消息有长度限制，超长内容拆分发送
        for i in range(0, len(response), _QQ_MAX_LENGTH):
            chunk = response[i : i + _QQ_MAX_LENGTH]
            # 只有第一条消息引用原始消息，后续分片不引用
            await self._send_message(chat_id, chunk, is_group, message_id if i == 0 else None)

    # ---- 消息发送 ----

    async def _send_message(
        self,
        openid: str,                      # 目标 openid
        text: str,                        # 消息文本
        is_group: bool,                   # 是否发送到群
        reply_msg_id: str | None = None,  # 引用的消息 ID（None 表示不引用）
    ):
        """发送消息的封装，根据 is_group 调用不同的 API。

        QQ Bot API 的消息类型（msg_type）：
          0 → 纯文本
          1 → Markdown（仅群聊支持）
          2 → Ark 消息（卡片）
          3 → Embed 消息
          4 → 富媒体
        """
        try:
            if is_group:
                await self.api.post_group_message(
                    group_openid=openid,
                    msg_type=0,                      # 纯文本
                    content=text,
                    msg_id=reply_msg_id or "",       # 引用回复的消息 ID
                )
            else:
                await self.api.post_c2c_message(
                    openid=openid,
                    msg_type=0,
                    content=text,
                    msg_id=reply_msg_id or "",
                )
        except Exception:
            logger.exception("Failed to send message to %s", openid)

    async def send_message_raw(self, openid: str, text: str) -> None:
        """发送 C2C 私聊消息（无引用），供调度器和 AI 工具调用。

        这是 send_message MCP 工具的底层实现。
        与 _send_message 的区别：不引用消息、总是发送 C2C、不吃异常。
        """
        try:
            await self.api.post_c2c_message(
                openid=openid,
                msg_type=0,
                content=text,
            )
        except Exception:
            logger.exception("Failed to send raw message to %s", openid)


# ---- 工厂函数 ----

def setup_bot() -> QQBotClient:
    """创建 QQBotClient 实例并配置 Intents。

    Intents（意图）控制了 Bot 接收哪些类型的事件：
      public_messages=True → 接收群聊 @ 消息
      direct_message=True  → 接收 C2C 私聊消息

    不需要 guilds（频道）相关事件，因为我们只用私聊和群聊。
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global _bot_client
    intents = botpy.Intents(public_messages=True, direct_message=True)
    _bot_client = QQBotClient(intents=intents)
    return _bot_client
