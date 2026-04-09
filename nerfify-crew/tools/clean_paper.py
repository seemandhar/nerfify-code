"""
LLM-based paper cleaning tool for CrewAI agents.

Reads raw markdown from disk, sends it to an LLM for cleaning,
and writes the cleaned result to disk. This avoids passing large
content through CrewAI tool arguments.
"""
from __future__ import annotations

import os
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


CLEANER_SYSTEM = (
    "You are a meticulous Markdown cleaner for research papers. "
    "Apply the following strictly. Output ONLY valid GitHub-flavored Markdown—no commentary, no code fences, no extra prose.\n\n"
    "Rules:\n"
    "1) Text hygiene: de-hyphenate wrapped words, fix OCR ligatures, normalize quotes/dashes/units; remove duplicate lines/sections; fix spacing/formatting.\n"
    "2) Equations: NEVER delete or alter equations. Preserve inline $…$ and display $$…$$ math exactly as authored, including numbering/labels if present.\n"
    "3) Scope pruning: remove generic narrative sections (Introduction/background, Related Work surveys, qualitative/marketing-style Results). "
    "Keep ONLY content needed to implement and reproduce experiments: problem setup, assumptions, notation, model architecture, objectives/losses, algorithms/pseudocode, training schedule, datasets, preprocessing, hyperparameters, ablations that affect implementation, evaluation protocol/metrics definitions.\n"
    "4) Tables: delete benchmark/comparison tables. If a single clear takeaway is obvious, replace with a one-line textual takeaway. "
    "Retain implementation-critical tables (hyperparams, layer configs, dataset splits) but convert them into concise bullet lists; no raw HTML or Markdown tables.\n"
    "5) Figures: remove images/captions/links/placeholders. If nearby text contains implementation-relevant details, keep that text as plain prose. Do NOT invent content.\n"
    "6) Strip raw HTML artifacts entirely (<td>, <tr>, <table>, inline styles). Keep only pure Markdown and math.\n"
    "7) Citations: remove unnecessary/dangling citations and citation dumps. Keep citations only when required to identify datasets/codebases/definitions essential for reproduction.\n"
    "8) Summaries: when paragraphs are verbose or narrative, compress to 3–6 bullets emphasizing actionable implementation details and experimental setup (no fixed word threshold).\n"
    "9) Final check: output must parse as clean, minimal Markdown with intact math, no images, no HTML table tags, no benchmark tables, and no broken anchors.\n"
    "10) Brevity: make the output as short as possible while preserving all information required for implementation and experiments.\n"
)


class CleanPaperSchema(BaseModel):
    """Input schema for clean_paper tool."""
    input_path: str = Field(
        default="",
        description="Path to the raw markdown file to clean (e.g. workspace/raw_paper.md)",
    )
    output_path: str = Field(
        default="",
        description="Path to write the cleaned markdown to (e.g. workspace/cleaned_paper.md)",
    )


class CleanPaperTool(BaseTool):
    """Clean a raw research paper markdown using an LLM.

    Reads the raw markdown from disk, sends it to an LLM for cleaning,
    and writes the cleaned result to disk. Use this instead of trying to
    pass large paper content through other tools.
    """

    name: str = "clean_paper"
    description: str = (
        "Clean a raw research paper markdown using an LLM. "
        "Provide the input_path (raw markdown file) and output_path (where to save cleaned markdown). "
        "The tool reads the file, sends it to an LLM for intelligent cleaning "
        "(removing narrative, keeping implementation details, preserving math), "
        "and writes the result to disk. Returns a summary of what was done."
    )
    args_schema: type = CleanPaperSchema

    # Set by the pipeline before use
    model: str = "anthropic/claude-sonnet-4-20250514"

    def _run(self, input_path: str = "", output_path: str = "") -> str:
        if not input_path or not output_path:
            return "Error: Both input_path and output_path are required."

        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            return f"Error: Input file not found: {input_path}"

        raw_text = input_file.read_text(encoding="utf-8")
        raw_size = len(raw_text)

        if raw_size == 0:
            return "Error: Input file is empty."

        try:
            import litellm

            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLEANER_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "Clean the following Markdown. "
                            "Output ONLY the cleaned Markdown.\n\n" + raw_text
                        ),
                    },
                ],
                max_tokens=32000,
            )

            cleaned = (response.choices[0].message.content or "").strip()

            if not cleaned:
                return "Error: LLM returned empty response."

            # Write to disk
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(cleaned + "\n", encoding="utf-8")

            clean_size = len(cleaned)
            ratio = clean_size / raw_size * 100 if raw_size > 0 else 0

            # Token usage
            usage = getattr(response, "usage", None)
            tok_info = ""
            if usage:
                tok_info = (
                    f" | Tokens: {getattr(usage, 'prompt_tokens', '?')} in, "
                    f"{getattr(usage, 'completion_tokens', '?')} out"
                )

            return (
                f"Cleaned paper saved to {output_path}\n"
                f"Raw: {raw_size:,} chars -> Cleaned: {clean_size:,} chars "
                f"({ratio:.0f}% of original){tok_info}"
            )

        except Exception as e:
            return f"Error during LLM cleaning: {e}"
