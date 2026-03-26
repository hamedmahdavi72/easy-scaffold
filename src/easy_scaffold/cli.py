# src/easy_scaffold/cli.py
import asyncio
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from easy_scaffold.app import App
from easy_scaffold.common.custom_exceptions import ConfigException
from easy_scaffold.common.env_bootstrap import load_local_dotenv

logger = logging.getLogger(__name__)

# Load `.env` before Hydra (override=False: platform-injected env wins). Disable on AWS with
# EASY_SCAFFOLD_LOAD_DOTENV=0 if you do not want the file read at all.
project_root = Path(__file__).parent.parent.parent
load_local_dotenv(project_root)


@hydra.main(config_path="../../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for the Easy Scaffold CLI."""

    # Set up logging from the configuration
    log_level = cfg.logging.level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        app = App(cfg)
        asyncio.run(app.run())

    except ConfigException as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


