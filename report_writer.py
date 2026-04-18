from __future__ import annotations

from datetime import datetime
from pathlib import Path

from subagents.writer_agent import ReportData

REPORTS_DIR = Path("reports")


def save_report(report: ReportData) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = REPORTS_DIR / f"report-{timestamp}.md"

    follow_ups = "\n".join(f"- {q}" for q in report.follow_up_questions)
    contents = (
        "# Research Report\n\n"
        "## Summary\n\n"
        f"{report.short_summary}\n\n"
        "## Report\n\n"
        f"{report.markdown_report}\n\n"
        "## Follow-up Questions\n\n"
        f"{follow_ups}\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path
