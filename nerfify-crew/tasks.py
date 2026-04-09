"""
CrewAI task definitions for the Nerfify pipeline.

Maps the pipeline stages to CrewAI Task objects with proper dependencies,
context passing, and output expectations.
"""
from __future__ import annotations

from pathlib import Path

from crewai import Task

from config import PipelineConfig


def build_tasks(
    agents: dict,
    *,
    paper_input: str,
    workspace: Path,
    output_dir: Path,
    method_name: str | None = None,
    data_path: str | None = None,
    config: PipelineConfig,
) -> list[Task]:
    """Build the ordered list of CrewAI tasks for the pipeline.

    Returns tasks in execution order. CrewAI handles dependencies via
    the `context` parameter (tasks that must complete before this one).
    """

    method_override = ""
    if method_name:
        method_override = f"\nIMPORTANT: Use this exact method name: {method_name}"

    # ── Task 1: Parse the paper ──────────────────────────────────
    parse_task = Task(
        description=f"""Parse the research paper and produce clean markdown.

Input: {paper_input}
Workspace: {workspace}

Steps:
1. If this is an arXiv ID/URL, download the PDF to {workspace}/paper.pdf
   Use shell_command: `wget -O {workspace}/paper.pdf <url>`
2. Run mineru to extract markdown using shell_command:
   `export HF_HOME=$HOME/.cache/huggingface && mineru -p <pdf_path> -o {workspace}/mineru_output`
3. Find the largest .md file in the mineru output using shell_command:
   `find {workspace}/mineru_output -name "*.md" -type f -exec ls -lh {{}} \\; | sort -k5 -hr | head -5`
4. Copy the raw mineru markdown to workspace using shell_command:
   `cp <mineru_md_file> {workspace}/raw_paper.md`
5. Use the clean_paper tool to clean the markdown:
   clean_paper(input_path="{workspace}/raw_paper.md", output_path="{workspace}/cleaned_paper.md")
   This sends the raw paper to an LLM for intelligent cleaning (removes narrative, keeps implementation details, preserves math).
6. Use web_search to look up referenced papers, GitHub repos, and datasets mentioned in the paper

IMPORTANT: Use the clean_paper tool for step 5. Do NOT try to read the paper content and write it back through file_write or shell_command — the content is too large for tool arguments.

Report the paper title and approximate size of cleaned output.""",
        expected_output=(
            "Cleaned paper markdown saved to workspace. Report includes paper title, "
            "size of cleaned output, and any references found."
        ),
        agent=agents["parser"],
    )

    # ── Task 2: Citation Recovery (skipped in fast mode) ─────────
    citation_task = None
    if not config.fast_mode:
        citation_task = Task(
            description=f"""Recover implementation details from cited papers.

Read the cleaned paper from: {workspace}/cleaned_paper.md

Identify implementation gaps where the paper references cited works for critical
details (architecture, loss functions, training protocols) without fully describing them.

For each gap:
1. Use web_search to find the cited paper
2. Use web_fetch to read the relevant sections
3. Extract the specific missing details

Write results to:
- {workspace}/citation_details.json
- {workspace}/citation_recovery.md

If the paper is self-contained, write a note saying no recovery needed.""",
            expected_output=(
                "Citation details JSON and recovery markdown saved to workspace. "
                "Report includes number of gaps found and key details recovered."
            ),
            agent=agents["citation_recovery"],
            async_execution=True,  # Runs in parallel with planner
        )

    # ── Task 3: Plan the architecture (parallel with citation if enabled) ──
    plan_task = Task(
        description=f"""Create an architecture plan for the NeRFStudio implementation.

Read these files:
- {workspace}/cleaned_paper.md (the cleaned research paper)
- {workspace}/citation_details.json (recovered citation details, if it exists)
- Template files at: {config.template_root}

Design a dependency DAG and file generation plan.{method_override}

Use web_search to look up NeRFStudio documentation and similar implementations.

Write the plan to: {workspace}/dag_plan.json

The plan must include: method_name, nodes, edges, files, base_architecture, summary.""",
        expected_output=(
            "DAG plan JSON saved to workspace with method name, component graph, "
            "file generation order, and architecture choice."
        ),
        agent=agents["planner"],
        async_execution=bool(citation_task),  # Only async if citation runs in parallel
    )

    # ── Sync barrier: code generation waits for parallel tasks ───

    # ── Task 4: Generate code ────────────────────────────────────
    code_task = Task(
        description=f"""Generate the complete NeRFStudio method implementation.

Read these files for context:
- {workspace}/cleaned_paper.md
- {workspace}/dag_plan.json
- {workspace}/citation_recovery.md

Output directory: {output_dir}
{method_override}

CRITICAL: You MUST use the file_write tool to save each file to disk.
DO NOT just return code in your response — the tester needs files on disk.

Use file_write to create ALL 8 files:
1. file_write(file_path="{output_dir}/method_template/__init__.py", content=...)
2. file_write(file_path="{output_dir}/method_template/template_config.py", content=...)
3. file_write(file_path="{output_dir}/method_template/template_datamanager.py", content=...)
4. file_write(file_path="{output_dir}/method_template/template_field.py", content=...)
5. file_write(file_path="{output_dir}/method_template/template_model.py", content=...)
6. file_write(file_path="{output_dir}/method_template/template_pipeline.py", content=...)
7. file_write(file_path="{output_dir}/README.md", content=...)
8. file_write(file_path="{output_dir}/pyproject.toml", content=...)

FIRST: Use file_read to read the template files listed in your backstory.
At minimum read template_config.py, template_model.py, template_field.py, and pyproject.toml.
Also read VanillaNerfOriginal.py from the examples for the base NeRF pattern.
Use web_search to verify NeRFStudio APIs and PyTorch patterns.

After writing all files:
- Verify files exist: use file_glob with pattern="**/*" on {output_dir}
- Run the mandatory self-check:
  1. Import chain validation
  2. METHOD_NAME consistency
  3. Config wiring
  4. Forward pass return types
  5. Loss function completeness
  6. pyproject.toml dependencies

Fix any issues found during self-check immediately by rewriting with file_write.""",
        expected_output=(
            "All 8 NeRFStudio method files written to disk using file_write in the output directory. "
            "Self-check passed with all imports resolving, consistent method name, "
            "correct config wiring, and complete loss functions."
        ),
        agent=agents["coder"],
    )

    # ── Task 5: Review (optional) ────────────────────────────────
    tasks = [t for t in [parse_task, citation_task, plan_task, code_task] if t]

    if config.enable_review:
        review_task = Task(
            description=f"""Review the generated NeRFStudio code for correctness.

Read ALL generated files from: {output_dir}
Compare against templates at: {config.template_root}

Check for:
1. Import errors and circular imports
2. API mismatches with NeRFStudio templates
3. METHOD_NAME consistency across all files
4. Missing implementations or placeholder code
5. Dependencies not declared in pyproject.toml
6. Loss functions matching the paper

Use web_search to verify NeRFStudio APIs if needed.

Write review to: {workspace}/review_result.json
Format: {{"approved": true/false, "issues": [...], "summary": "..."}}

If critical issues found, list precise fixes needed.""",
            expected_output=(
                "Review JSON saved with approval status, list of issues, and summary."
            ),
            agent=agents["reviewer"],
        )
        tasks.append(review_task)

        # Fix task if review finds issues — the coder fixes based on review
        fix_task = Task(
            description=f"""Fix any issues found during code review.

Read the review from: {workspace}/review_result.json
If approved=true, report that no fixes are needed.

If approved=false, read the issues list and fix each one:
- Read the affected files from {output_dir}
- Apply the fixes
- Rewrite the complete files (not patches)
- Ensure METHOD_NAME remains consistent
- Update pyproject.toml if needed

Re-run the self-check after fixing.""",
            expected_output=(
                "All review issues fixed. Files rewritten with corrections applied. "
                "Self-check passed after fixes."
            ),
            agent=agents["coder"],
        )
        tasks.append(fix_task)

    # ── Task 6: Smoke Test ───────────────────────────────────────
    if config.enable_smoke_test:
        test_task = Task(
            description=f"""Run smoke tests on the generated NeRFStudio implementation.

Output directory: {output_dir}
Conda environment: {config.conda_env}
Data path: {data_path or config.default_dataset}

Run these steps in order using shell_command. Run each step SEPARATELY (one shell_command per step). Stop on first failure.

Step 1 — Install package:
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && pip install -e . 2>&1 | tail -5

Step 2 — Register CLI:
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && ns-install-cli

Step 3 — Import check:
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && python -c "from method_template.template_config import method_template; print('All imports OK')"

Step 4 — Read pyproject.toml to find the method CLI name (under [project.entry-points."nerfstudio.method_configs"]).

Step 5 — Short training run (10 iterations). This MUST complete all 10 iterations:
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && ns-train <METHOD_NAME> --data {data_path or config.default_dataset} --max-num-iterations 10 --viewer.quit-on-train-completion True --vis none {config.dataparser}

IMPORTANT: The ns-train command may take a few minutes. Let it run to completion — do NOT cancel it.
The command will exit on its own after 10 iterations. Check the output for "Training Finished" or similar.

Write results to: {workspace}/test_result.json
Include: passed (true/false), steps_completed (list), error_log (if failed).""",
            expected_output=(
                "Test results JSON saved with pass/fail status, completed steps, "
                "and error logs if any step failed."
            ),
            agent=agents["tester"],
        )
        tasks.append(test_task)

        # Debug loop (if test fails)
        for debug_iter in range(config.max_debug_iterations):
            debug_task = Task(
                description=f"""Debug iteration {debug_iter + 1}: Fix errors from smoke test.

FIRST: Read {workspace}/test_result.json using file_read.
If "passed" is true, IMMEDIATELY respond with "No debugging needed — tests pass." and stop. Do NOT use any other tools.

ONLY if "passed" is false, then debug:
1. Read the error_log from the test results
2. Read the failing code files from {output_dir}
3. Diagnose the root cause
4. Fix the code by rewriting affected files in {output_dir}
5. Keep METHOD_NAME consistent

After fixing, re-run smoke test as separate commands:
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && pip install -e . 2>&1 | tail -5
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && ns-install-cli
  eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && python -c "from method_template.template_config import method_template; print('OK')"

Update {workspace}/test_result.json with new results.""",
                expected_output=(
                    "If tests pass: 'No debugging needed'. "
                    "If tests fail: diagnosis saved, code fixed, smoke test re-run."
                ),
                agent=agents["debugger"],
            )
            tasks.append(debug_task)

    # ── Task 7: Full Training + PSNR Feedback Loop ─────────────
    if config.enable_training:
        scene = config.scenes[0] if config.scenes else "garden"
        psnr_target = config.get_psnr_target(scene)
        training_output_dir = output_dir / "training_outputs"
        read_tb_script = config.read_tb_script

        train_task = Task(
            description=f"""Run full training of the generated NeRFStudio method.

Output directory: {output_dir}
Training output directory: {training_output_dir}
Conda environment: {config.conda_env}
Data path: {data_path or config.default_dataset}
Scene: {scene}
Max iterations: {config.default_max_iters}
GPU: {config.training_gpu}

Steps:
1. Install the method:
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; cd {output_dir} && pip install -e . && ns-install-cli

2. Read pyproject.toml to find the method CLI name (under [project.entry-points."nerfstudio.method_configs"])

3. Run full training with TensorBoard and live viewer:
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
   CUDA_VISIBLE_DEVICES={config.training_gpu} ns-train <METHOD_NAME_CLI> \\
     --data {data_path or config.default_dataset} \\
     --vis viewer+tensorboard \\
     --max-num-iterations {config.default_max_iters} \\
     --output-dir {training_output_dir}/{scene} \\
     --viewer.quit-on-train-completion True \\
     --viewer.websocket-port {config.viewer_port} \\
     {config.dataparser}

4. After training completes, find the latest run directory:
   ls -t {training_output_dir}/{scene}/<METHOD_NAME_CLI>/ | head -1

5. Read the PSNR curve using TensorBoard:
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
   python {read_tb_script} {training_output_dir}/{scene}/<METHOD_NAME_CLI>/<LATEST_TIMESTAMP>/ --json

6. Parse the JSON output and analyze:
   - final_psnr: PSNR at end of training
   - max_psnr: Peak PSNR during training
   - issues: Detected problems (NaN, drops, still rising)
   - samples: The full PSNR curve

Write training results to: {workspace}/training_result.json
Include: final_psnr, max_psnr, psnr_target={psnr_target}, issues, pass/fail status""",
            expected_output=(
                "Training completed. PSNR curve extracted from TensorBoard. "
                "Training results JSON saved with PSNR metrics and pass/fail status."
            ),
            agent=agents["tester"],
        )
        tasks.append(train_task)

        # PSNR debug loop — if PSNR too low, diagnose and fix
        if config.enable_psnr_feedback:
            for psnr_iter in range(config.max_psnr_fix_iterations):
                psnr_fix_task = Task(
                    description=f"""PSNR fix iteration {psnr_iter + 1}: Analyze training results and fix code if PSNR is below target.

Read the training results from: {workspace}/training_result.json
PSNR target: {psnr_target} dB (dataset: {config.dataset}, scene: {scene})

If final_psnr >= {psnr_target} AND no critical issues → report PASS and stop.

If final_psnr < {psnr_target} OR critical issues found:

1. DIAGNOSE the cause by analyzing the PSNR curve shape:
   - Flat/low curve → loss function not converging, check loss weights and gradient flow
   - Rising then sudden drop → training instability, learning rate too high, exploding gradients
   - Slowly rising but not reaching target → architecture too small, missing components
   - NaN values → numerical issues, missing clamps, bad initialization
   - Curve plateaus early → learning rate too low, or model capacity issue

2. Read the paper again ({workspace}/cleaned_paper.md) and compare:
   - Are all loss terms implemented with correct weights?
   - Are all architectural components present?
   - Are hyperparameters matching the paper?

3. Read the current code from {output_dir} and identify specific fixes

4. Fix the code:
   - Rewrite affected files in {output_dir}
   - Keep METHOD_NAME consistent
   - Focus on the diagnosed issue

5. Re-install and re-train:
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
   cd {output_dir} && pip install -e . && ns-install-cli && \\
   CUDA_VISIBLE_DEVICES={config.training_gpu} ns-train <METHOD_NAME_CLI> \\
     --data {data_path or config.default_dataset} \\
     --vis viewer+tensorboard \\
     --max-num-iterations {config.default_max_iters} \\
     --output-dir {training_output_dir}/{scene} \\
     --viewer.quit-on-train-completion True \\
     --viewer.websocket-port {config.viewer_port} \\
     {config.dataparser}

6. Read new PSNR:
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
   python {read_tb_script} {training_output_dir}/{scene}/<METHOD_NAME_CLI>/<LATEST_TIMESTAMP>/ --json

Update {workspace}/training_result.json with new results.
Write diagnosis to: {workspace}/psnr_diagnosis_{psnr_iter + 1}.json""",
                    expected_output=(
                        "PSNR diagnosis saved. If below target: code fixed, retrained, "
                        "and new PSNR measured. Updated training results saved."
                    ),
                    agent=agents["debugger"],
                )
                tasks.append(psnr_fix_task)

    return tasks
