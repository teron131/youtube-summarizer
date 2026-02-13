"""Helper functions for async task execution and response formatting."""

import asyncio
from datetime import UTC, datetime

from fastapi import HTTPException

from youtube_summarizer.settings import get_settings

DEFAULT_TASK_TIMEOUT_SECONDS = get_settings().task_timeout_seconds


async def run_async_task(func, *args, timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS):
    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, func, *args),
            timeout=timeout,
        )
    except TimeoutError as err:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {timeout} seconds",
        ) from err
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:100]}") from e


def get_processing_time(start_time: datetime) -> str:
    return f"{(datetime.now(UTC) - start_time).total_seconds():.1f}s"
