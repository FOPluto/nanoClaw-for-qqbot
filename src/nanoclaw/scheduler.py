# scheduler.py
# =============
# 定时任务调度模块：定期扫描数据库中到期的任务，调用 AI Agent 执行。
#
# 工作流程：
#   每隔 SCHEDULER_INTERVAL 秒
#     → 查询到期任务（db.get_due_tasks）
#       → 对每个到期任务
#         → 包装 prompt（告知 AI 这是定时任务，必须发消息通知用户）
#         → 调用 run_task_agent（agent.py）
#         → 计算下次执行时间
#         → 更新数据库状态
#         → 记录执行日志
#
# 使用的调度类型：
#   "cron"     → cron 表达式（如 "0 9 * * 1" = 每周一上午 9 点）
#   "interval" → 固定间隔（毫秒）
#   "once"     → 一次性任务（ISO 时间戳）

import logging                              # 日志
import time                                 # 高精度计时（计算执行耗时）
from datetime import datetime, timedelta, timezone  # 时间计算

# apscheduler：AsyncIO 定时任务框架。
# AsyncIOScheduler 是它的异步调度器，和我们的 asyncio 事件循环无缝兼容。
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# croniter：解析 cron 表达式计算下一次触发时间。
# 虽然 apscheduler 内置了 cron 支持，但我们的任务数据来自数据库而非代码配置，
# 所以用 croniter 单独计算下次执行时间。
from croniter import croniter

from nanoclaw import db                                # 数据库操作
from nanoclaw.agent import run_task_agent              # AI Agent（定时任务专用）
from nanoclaw.config import SCHEDULER_INTERVAL         # 扫描间隔（默认 60 秒）

logger = logging.getLogger(__name__)

# 调度器实例（setup_scheduler 中创建）
_scheduler: AsyncIOScheduler | None = None


def setup_scheduler(bot, db_path: str) -> AsyncIOScheduler:
    """创建并配置定时任务调度器。

    参数：
      bot: QQBotClient 实例，给 Agent 发消息用
      db_path: SQLite 数据库路径

    返回：已配置但未启动的调度器（调用方负责 .start()）。
    """
    global _scheduler

    _scheduler = AsyncIOScheduler()

    # add_job：注册一个定时执行的函数
    # "interval" 表示固定间隔执行
    # seconds=SCHEDULER_INTERVAL 是间隔秒数
    _scheduler.add_job(
        _check_tasks,                   # 要执行的函数
        "interval",              # 调度类型
        seconds=SCHEDULER_INTERVAL,     # 间隔（默认 60 秒）
        args=[bot, db_path],            # 传给 _check_tasks 的参数
        id="check_tasks",               # Job ID（用于后续管理）
        replace_existing=True,          # 如果已有同名 job，替换而不是报错
    )

    return _scheduler


async def _check_tasks(bot, db_path: str) -> None:
    """扫描数据库中所有到期任务，逐个执行。

    这是调度器的"心跳函数"，每 SCHEDULER_INTERVAL 秒调用一次。
    """
    try:
        tasks = await db.get_due_tasks(db_path)
    except Exception:
        logger.exception("Failed to query due tasks")
        return

    # 逐个执行——没有用 asyncio.gather 并发，是因为每个任务
    # 都要调 AI API，并发多个可能导致限流。
    for task in tasks:
        try:
            await _execute_task(task, bot, db_path)
        except Exception:
            logger.exception("Failed to execute task %s", task["id"])


async def _execute_task(task: dict, bot, db_path: str) -> None:
    """执行单条定时任务。

    流程：
    1. 包装 prompt → 告知 AI 它的身份（定时任务执行者）
    2. 调 Agent → AI 处理任务并（应该）发消息通知用户
    3. Fallback → 如果 AI 忘记发消息，这里兜底发一条
    4. 计算下次执行时间并更新数据库
    5. 记录执行日志
    """
    task_id = task["id"]
    task_chat_id = task["chat_id"]    # 任务创建时的 openid
    prompt = task["prompt"]           # AI 看到的任务描述

    logger.info("Executing task %s for chat %s: %s", task_id, task_chat_id, prompt[:80])

    # 包装 prompt：告诉 AI 这是定时任务，必须发消息通知用户。
    # 不加这句话 AI 可能只执行操作但不通知，导致用户对定时任务无感知。
    wrapped_prompt = (
        "You are executing a scheduled task. "
        "You MUST use the send_message tool to notify the user in QQ. "
        f"Task: {prompt}"
    )

    # notify_state 用 mutable dict 在线程间传递状态。
    # send_message 工具被调用时会把 sent 设为 True。
    notify_state: dict[str, bool] = {"sent": False}

    start = time.monotonic()  # 高精度计时器开始（不受系统时间调整影响）
    try:
        result = await run_task_agent(wrapped_prompt, bot, task_chat_id, db_path, notify_state)

        # Fallback：如果 AI 执行了任务但忘记调 send_message，我们兜底发一条。
        # 这是一个防御性设计，实际中 Claude 可能会忘记调用工具。
        if not notify_state["sent"]:
            await bot.send_message_raw(openid=task_chat_id, text=f"定时提醒：{prompt}")

        duration_ms = int((time.monotonic() - start) * 1000)
        await db.log_task_run(db_path, task_id, duration_ms, "success", result=result)

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        await db.log_task_run(db_path, task_id, duration_ms, "error", error=str(e))
        result = f"Error: {e}"

    # 计算下次执行时间
    stype = task["scheduled_type"]
    svalue = task["schedule_value"]
    now = datetime.now(timezone.utc)

    if stype == "cron":
        # croniter 计算 cron 表达式的下一次触发
        next_run = croniter(svalue, now).get_next(datetime).isoformat()
        await db.update_task_after_run(db_path, task_id, result, next_run, "active")

    elif stype == "interval":
        # 当前时间 + 间隔 = 下次执行时间
        next_run = (now + timedelta(milliseconds=int(svalue))).isoformat()
        await db.update_task_after_run(db_path, task_id, result, next_run, "active")

    elif stype == "once":
        # 一次性任务：标记为 completed，next_run 设为 None
        await db.update_task_after_run(db_path, task_id, result, None, "completed")

    else:
        logger.warning("Unknown schedule_type %s for task %s", stype, task_id)
