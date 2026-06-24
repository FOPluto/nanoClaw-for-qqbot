## 数据库模块db.py
## 数据库层，负责定时任务的增删改查
##
## 为什么使用SQLite？
# - 零配置 不需要单独安装数据库服务
# - 轻量级，适合单用户的个人项目
# - aiosqlite 提供异步接口，不阻塞事件循环
#
# 为什么不用 ORM
# - 项目只有两张表手写sql更加直观，减少依赖，后续再进行扩展

## uuid 标准库：生成全局唯一id
import uuid

## datetime 时间标准库，timezone.utc 确保所有时间是 UTC ，避免时区混乱
from datetime import datetime, timedelta, timezone

import aiosqlite
from nltk import data

## 建表sql
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,            -- 任务唯一id，八位十六进制
    chat_id TEXT NOT NULL,          -- 消息来源的openid（私聊是用户openid，群聊是群聊的openid）
    prompt TEXT NOT NULL,           -- 任务的提示词，用户要求ai做什么
    scheduled_type TEXT NOT NULL,   -- 调用类型'cron'（cron 表达式）、'interval'（间隔毫秒）、'once'（一次性 ISO 时间戳）
    schedule_value TEXT NOT NULL,   -- 调用参数，应该是函数的相关参数
    next_run TEXT,
    last_run TEXT,
    last_result TEXT,
    status TEXT DEFAULT 'active',
    create_at TEXT NOT NULL         -- 创建时间
);

-- 索引：加速按照 next_run 查找到期任务
CREATE INDEX IF NOT EXISTS scheduled_next_run_tasks_id ON scheduled_tasks (next_run);

-- 索引：加速按照 state 过滤任务（列出活跃的任务）
CREATE INDEX IF NOT EXISTS scheduled_tasks_states ON scheduled_tasks (status);

-- 任务执行日志表：记录每次task执行的详情，用于追溯
CREATE TABLE IF NOT EXISTS task_run_logs (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT,
    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_task_run_logs_task_id ON task_run_logs(task_id);
    
"""

async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_CREATE_TABLE)
        await db.commit()


async def create_task(
        db_path: str,
        chat_id: str,
        prompt: str,
        scheduled_type: str,
        scheduled_value: str,
        next_run: str,
):
    """
    创建一个新的定时任务
    :param db_path:
    :param chat_id: 消息来源的openid，回复消息的时候需要用到
    :param prompt: AI执行接受的提示词
    :param scheduled_type: 任务的类型
    :param schedule_value: cron表达式，时间戳
    :param next_run: 下一次执行的时间
    :return:
    """
    task_id = uuid.uuid4().hex[:8] ## uuid生成随机的uuid，取1前八位作为短ID
    # 单人使用不会发生碰撞
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO scheduled_tasks (id, chat_id, prompt, scheduled_type, schedule_value, next_run, create_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ## 这里一般使用?传递参数，占位符传参，防止SQL注入
            (task_id, chat_id, prompt, scheduled_type, scheduled_value, next_run, datetime.now(timezone.utc))
        )

        await db.commit()

    return task_id

async def get_all_tasks(db_path: str) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        ## 设置查询返回结果返回dict，而不是tuple
        ## 可读性更强，更好调用
        db.row_factory = aiosqlite.Row
        await db.execute(
            """
            SELECT * FROM scheduled_tasks
            """
        )
        await db.commit()
        rows = await db.fetchall()
        return [dict(r) for r in rows]

async def get_due_tasks(db_path: str) -> list[dict]:
    """
    获取所有到期未执行的任务
    用scheduler每秒调用一次，检查是否有需要执行的任务
    :param db_path:
    :return:
    """

    ## 生成当前时间的iso字符串，和数据库中的next_run进行比较
    ## ISO字符串可以直接按照字典序直接比较大小
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM scheduled_tasks WHERE status = 'active' AND next_run < ?",
            (now, ),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

async def update_task_status(db_path, task_id: str, status: str):
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            UPDATE scheduled_tasks SET status = ? WHERE id = ?
            """,
            (status, task_id, ),
        )
        await db.commit()


async def delete_task(db_path: str, task_id: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            DELETE FROM scheduled_tasks WHERE id = ?
            """,
            (task_id, ),
        )
        await db.commit()


async def update_task_after_run(
        db_path: str,
        task_id: str,
        last_result: str,
        next_run: str | None = None,
        status: str = 'active',
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE scheduled_tasks SET status = ?, last_run = ?, last_result = ?, next_run = ? WHERE id = ?
            """,
            (status, now, last_result, next_run, task_id, ),
        )
        await db.commit()


async def log_task_run(
        db_path: str,
        task_id: str,
        duration_ms: int,
        status: str,
        result: str | None = None,
        error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result, error) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, now, duration_ms, status, result, error, ),
        )
        await db.commit()


async def get_task_logs(db_path: str) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(
            """
            SELECT * FROM task_run_logs
            """
        )
        await db.commit()
        rows = await db.fetchall()
        return [dict(r) for r in rows]

