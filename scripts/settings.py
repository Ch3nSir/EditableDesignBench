"""Portable defaults shared by command-line entry points."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env', override=False)
DATASET_ROOT = Path(os.environ.get('EDITABLEDESIGNBENCH_DATASET_ROOT', PROJECT_ROOT / 'data/EditableDesignBench-v3')).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get('EDITABLEDESIGNBENCH_MODEL_ROOT', PROJECT_ROOT / 'models')).expanduser().resolve()
