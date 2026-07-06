"""
Setup and configuration for UCI ML Repository datasets.

Ensures ucimlrepo is installed and provides utility functions.
"""

import subprocess
import sys
import logging

logger = logging.getLogger(__name__)


def check_and_install_ucimlrepo():
    """
    Check if ucimlrepo is installed, install if not.
    """
    try:
        import ucimlrepo
        logger.info(f"ucimlrepo is installed: {ucimlrepo.__version__}")
        return True
    except ImportError:
        logger.warning("ucimlrepo not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
            logger.info("ucimlrepo installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ucimlrepo: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_and_install_ucimlrepo()
