import asyncio

async def say_after(delay: int, message: str):
    await asyncio.sleep(delay)
    return f"{delay} : {message}"

async def main():
    task1 = asyncio.create_task(
        say_after(1, message="hello")
    )
    task2 = asyncio.create_task(
        say_after(2, message="demo")
    )

    res = await asyncio.gather(task1, task2)

    print(res)



async def main2():
    res = await asyncio.gather(
        say_after(1, message="hello"),
        say_after(2, message="demo")
    )

    print(res)



asyncio.run(main2())