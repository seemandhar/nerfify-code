"""
CrewAI Crew definition for the Nerfify pipeline.

Assembles agents and tasks into a Crew with the appropriate process type
and configuration for converting research papers to NeRFStudio code.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process

from agents.definitions import build_agents
from config import PipelineConfig
from tasks import build_tasks


class NerfifyCrew:
    """Orchestrates the Nerfify multi-agent pipeline using CrewAI."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.config.ensure_dirs()

    def run(
        self,
        *,
        arxiv: str | None = None,
        pdf_url: str | None = None,
        pdf_path: str | None = None,
        method_name: str | None = None,
        data_path: str | None = None,
    ) -> dict:
        """Run the full Nerfify pipeline."""

        # Determine paper input
        if arxiv:
            paper_input = f"arXiv paper: {arxiv}"
        elif pdf_url:
            paper_input = f"PDF URL (download first): {pdf_url}"
        elif pdf_path:
            paper_input = f"Local PDF: {pdf_path}"
        else:
            raise ValueError("Must provide arxiv, pdf_url, or pdf_path")

        # Create unique run directories
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        workspace = self.config.workspace_dir / run_id
        output_dir = self.config.generated_dir / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Nerfify-Crew Pipeline")
        print(f"{'='*60}")
        print(f"  Run ID:    {run_id}")
        print(f"  Workspace: {workspace}")
        print(f"  Output:    {output_dir}")
        print(f"  Paper:     {paper_input}")
        print(f"{'='*60}\n")

        # Build agents
        agents = build_agents(self.config)
        print(f"Loaded {len(agents)} agents: {', '.join(agents.keys())}")

        # Build tasks
        tasks = build_tasks(
            agents,
            paper_input=paper_input,
            workspace=workspace,
            output_dir=output_dir,
            method_name=method_name,
            data_path=data_path or self.config.default_dataset,
            config=self.config,
        )
        print(f"Created {len(tasks)} pipeline tasks\n")

        # Enable retry for rate limits
        import litellm
        litellm.num_retries = 5
        litellm.retry_after = 60  # wait 60s before retry on 429 (rate limit is per-minute)

        # Assemble the crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            full_output=True,
        )

        # Run
        start_time = time.time()
        result = crew.kickoff()
        elapsed = time.time() - start_time

        # Collect generated files
        gen_files = sorted(
            f.relative_to(output_dir)
            for f in output_dir.rglob("*")
            if f.is_file()
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Run ID:     {run_id}")
        print(f"  Duration:   {elapsed:.1f}s")
        print(f"  Output:     {output_dir}")
        if gen_files:
            print(f"  Files ({len(gen_files)}):")
            for f in gen_files:
                print(f"    {f}")
        else:
            print("  WARNING: No files generated!")
        print(f"{'='*60}\n")

        # Save run result
        run_result = {
            "run_id": run_id,
            "workspace": str(workspace),
            "output_dir": str(output_dir),
            "duration_s": round(elapsed, 1),
            "generated_files": [str(f) for f in gen_files],
            "result": str(result),
        }
        (workspace / "result.json").write_text(
            json.dumps(run_result, indent=2, default=str)
        )

        # Extract and save token usage if available
        if hasattr(result, "token_usage"):
            run_result["token_usage"] = {
                "total_tokens": getattr(result.token_usage, "total_tokens", None),
                "prompt_tokens": getattr(result.token_usage, "prompt_tokens", None),
                "completion_tokens": getattr(result.token_usage, "completion_tokens", None),
            }
            (workspace / "result.json").write_text(
                json.dumps(run_result, indent=2, default=str)
            )

        return run_result
