"""Entry point that builds the Diagnostic Copilot's knowledge base in Chroma."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.logger import logging
from src_agent.config import load_agent_service_config
from src_agent.rag.ingestion_pipeline import IngestionPipeline

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent" / "default.yaml"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the knowledge-base ingest entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk, caption, embed, and upsert the knowledge base into "
        "the Chroma server."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the knowledge-base ingestion."""
    load_dotenv()
    arguments = parse_arguments()
    configuration = load_agent_service_config(arguments.config)
    pipeline = IngestionPipeline(configuration.rag, configuration.backends)
    document_count = pipeline.run(reset=arguments.reset)
    logging.info(f"Knowledge base ready ({document_count} documents)")


if __name__ == "__main__":
    main()
