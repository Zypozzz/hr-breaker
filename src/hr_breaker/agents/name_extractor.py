from pydantic import BaseModel
from pydantic_ai import Agent

from hr_breaker.config import get_flash_model, get_model_settings, get_settings
from hr_breaker.utils.retry import run_with_retry


class ExtractedName(BaseModel):
    first_name: str | None
    last_name: str | None


SYSTEM_PROMPT = """Extract the person's name from this resume/CV content.

Return:
- first_name: The person's first/given name
- last_name: The person's last/family name (may include middle names)

If you cannot find a name, return null for both fields.
Handle any format: LaTeX, plain text, markdown, HTML, etc.
Ignore formatting commands - extract the actual name text only.
"""


async def extract_name(content: str) -> tuple[str | None, str | None]:
    """Extract first and last name from resume content using LLM."""
    settings = get_settings()
    agent = Agent(
        get_flash_model(),
        output_type=ExtractedName,
        system_prompt=SYSTEM_PROMPT,
        model_settings=get_model_settings(),
    )
    # Only send first N chars - name should be at the top
    snippet = content[:settings.agent_name_extractor_chars]
    result = await run_with_retry(agent.run, f"Extract the name from this resume:\n\n{snippet}")
    return result.output.first_name, result.output.last_name
