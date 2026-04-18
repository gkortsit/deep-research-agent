import asyncio
from dotenv import load_dotenv
from auto_mode.main import input_with_fallback
from manager import ResearchManager

load_dotenv()


async def main() -> None:
    query = input_with_fallback(
        "What would you like to research? ",
        "Impact of electric vehicles on the grid.",
    )
    await ResearchManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())
