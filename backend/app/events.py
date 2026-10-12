import asyncio


class Notifier:
    """进程内发布/订阅：WS 连接订阅，调度器发布任务事件。"""

    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    async def publish(self, event: dict) -> None:
        for q in list(self._subs):
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)
