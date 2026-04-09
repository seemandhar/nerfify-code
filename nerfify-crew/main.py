#!/usr/bin/env python3
"""
Nerfify-Crew — CrewAI Multi-Agent Pipeline

Converts research papers about NeRF methods into complete NeRFStudio implementations
using specialized CrewAI agents.

Usage:
    # From arXiv URL
    python main.py --arxiv 2308.12345

    # From local PDF
    python main.py --pdf /path/to/paper.pdf

    # With options
    python main.py --arxiv 2308.12345 --method-name my_nerf --no-review --train
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from config import PipelineConfig
from crew import NerfifyCrew


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Nerfify-Crew: Convert NeRF papers to NeRFStudio code using CrewAI agents"
    )
    parser.add_argument("--arxiv", type=str, help="arXiv URL or ID (e.g., 2308.12345)")
    parser.add_argument("--url", type=str, help="Direct PDF URL")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    parser.add_argument("--method-name", type=str, help="Override method name (snake_case)")
    parser.add_argument("--data", type=str, help="Path to training dataset")
    parser.add_argument("--max-iters", type=int, default=3000, help="Max training iterations")

    # Model configuration
    parser.add_argument(
        "--model", type=str, default=None,
        help="Default LLM model (e.g., anthropic/claude-sonnet-4-20250514, openai/gpt-4o)"
    )
    parser.add_argument(
        "--coder-model", type=str, default=None,
        help="LLM model for the coder agent (defaults to opus)"
    )

    # Pipeline toggles
    parser.add_argument("--no-review", action="store_true", help="Skip code review step")
    parser.add_argument("--no-test", action="store_true", help="Skip smoke testing")
    parser.add_argument("--train", action="store_true", help="Enable full training + PSNR feedback")
    parser.add_argument("--no-psnr-feedback", action="store_true", help="Disable PSNR feedback loop")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device for training")
    parser.add_argument(
        "--dataset", type=str, default="mipnerf360",
        choices=["mipnerf360", "blender", "llff"],
    )
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--expected-psnr", type=float, default=None)
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip citation recovery if self-contained")
    parser.add_argument("--tiered", action="store_true", help="Use tiered model routing per agent")

    args = parser.parse_args()

    if not args.arxiv and not args.url and not args.pdf:
        parser.error("Must provide --arxiv, --url, or --pdf")

    # Build config
    config = PipelineConfig()
    config.enable_review = not args.no_review
    config.enable_smoke_test = not args.no_test
    config.enable_training = args.train
    config.enable_psnr_feedback = args.train and not args.no_psnr_feedback
    config.training_gpu = args.gpu
    config.dataset = args.dataset
    if args.scenes:
        config.scenes = args.scenes
    config.expected_psnr = args.expected_psnr
    config.fast_mode = args.fast
    config.tiered_models = args.tiered
    config.default_max_iters = args.max_iters

    if args.model:
        config.default_model = args.model
    if args.coder_model:
        config.coder_model = args.coder_model
        config._coder_model_explicit = True
    config.sync_models_to_provider()

    # Run the crew
    nerfify = NerfifyCrew(config=config)
    result = nerfify.run(
        arxiv=args.arxiv,
        pdf_url=args.url,
        pdf_path=args.pdf,
        method_name=args.method_name,
        data_path=args.data,
    )

    print(f"\nOutput directory: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
