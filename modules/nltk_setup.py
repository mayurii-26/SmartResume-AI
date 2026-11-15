"""
SmartResume AI - Centralized NLTK Environment and Resource Setup
Safely configures NLTK data directory and ensures required NLP resources
are available across both local environments and cloud deployments (e.g. Streamlit Community Cloud).
"""

import os
import sys
import logging
import shutil
import stat
from pathlib import Path
from typing import List, Tuple
import nltk

logger = logging.getLogger(__name__)

# List of required NLTK resources: (internal_search_path, package_name)
REQUIRED_RESOURCES: List[Tuple[str, str]] = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
]

_SETUP_DONE = False
_ACTIVE_NLTK_DIR: Path = None


def get_secure_nltk_dir() -> Path:
    """
    Determines the safest writable NLTK data directory.
    Prioritizes user home directory (~/nltk_data) to avoid cloud mount permission
    conflicts, falling back to local workspace data directory if home is inaccessible.
    """
    global _ACTIVE_NLTK_DIR
    if _ACTIVE_NLTK_DIR and _ACTIVE_NLTK_DIR.exists():
        return _ACTIVE_NLTK_DIR

    # 1. Check custom environment variable
    env_dir = os.environ.get("NLTK_DATA")
    if env_dir:
        candidate = Path(env_dir)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            _ACTIVE_NLTK_DIR = candidate
            return _ACTIVE_NLTK_DIR
        except Exception:
            pass

    # 2. Try user home directory (~/nltk_data), which is private on Streamlit Cloud Linux (/home/appuser)
    try:
        candidate = Path.home() / "nltk_data"
        candidate.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = candidate / ".perm_check"
        test_file.touch(exist_ok=True)
        test_file.unlink(missing_ok=True)
        _ACTIVE_NLTK_DIR = candidate
        return _ACTIVE_NLTK_DIR
    except Exception:
        pass

    # 3. Fallback to local workspace data/nltk_data if home is restricted
    from config.settings import DATA_DIR
    candidate = DATA_DIR / "nltk_data"
    candidate.mkdir(parents=True, exist_ok=True)
    _ACTIVE_NLTK_DIR = candidate
    return _ACTIVE_NLTK_DIR


def setup_secure_nltk_directory() -> Path:
    """
    Configures NLTK search paths and security settings.
    Disables pathsec enforcement to prevent false-positive refusals on container
    mount points (Streamlit Cloud) and registers the private data directory.
    """
    # Disable NLTK path security enforcement if present (NLTK >= 3.9)
    try:
        import nltk.pathsec
        nltk.pathsec.ENFORCE = False
    except Exception:
        pass

    target_dir = get_secure_nltk_dir()
    target_str = str(target_dir)

    # On POSIX (Linux/macOS, e.g. Streamlit Cloud), set 0700 private permissions
    if os.name != "nt" and target_dir.exists():
        try:
            os.chmod(target_dir, stat.S_IRWXU)  # 0o700
        except Exception as e:
            logger.debug(f"Could not chmod {target_dir}: {e}")

    # Clean potentially problematic paths from nltk.data.path
    # Particularly paths inside /mount/src on Streamlit Cloud
    cleaned_paths = []
    for p in nltk.data.path:
        p_str = str(p)
        p_norm = os.path.normpath(p_str).replace("\\", "/").lower()
        if "/mount/src" in p_norm and "nltk_data" in p_norm and p_str != target_str:
            continue
        if p_str not in cleaned_paths:
            cleaned_paths.append(p_str)

    # Ensure target directory is at the very top of nltk.data.path
    if target_str in cleaned_paths:
        cleaned_paths.remove(target_str)
    cleaned_paths.insert(0, target_str)

    nltk.data.path = cleaned_paths
    os.environ["NLTK_DATA"] = target_str

    return target_dir


def _seed_from_local_cache(target_dir: Path) -> None:
    """
    Copies pre-bundled NLTK resources from repo data/nltk_data if available,
    avoiding redundant network downloads when offline.
    """
    try:
        from config.settings import DATA_DIR
        local_repo_data = DATA_DIR / "nltk_data"
        if not local_repo_data.exists() or local_repo_data == target_dir:
            return

        for item in local_repo_data.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
    except Exception as e:
        logger.debug(f"Could not seed local NLTK data: {e}")


def ensure_nltk_resources(force: bool = False) -> bool:
    """
    Ensures all required NLTK resources (punkt_tab, punkt, stopwords, wordnet, vader_lexicon)
    are present and loaded without crashing the application.
    """
    global _SETUP_DONE
    if _SETUP_DONE and not force:
        return True

    target_dir = setup_secure_nltk_directory()
    target_str = str(target_dir)

    # Attempt to seed from local repository cache if target is currently empty
    _seed_from_local_cache(target_dir)

    all_available = True
    for resource_path, package_name in REQUIRED_RESOURCES:
        # Check if findable
        resource_found = False
        try:
            nltk.data.find(resource_path)
            resource_found = True
        except (LookupError, Exception):
            resource_found = False

        if not resource_found:
            logger.info(f"Downloading required NLTK resource '{package_name}' into {target_str}...")
            try:
                nltk.download(package_name, download_dir=target_str, quiet=True, raise_on_error=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource '{package_name}': {e}")
                # Try fallback download without download_dir specified
                try:
                    nltk.download(package_name, quiet=True)
                except Exception:
                    all_available = False

    # Fix permissions recursively on POSIX systems if any files were written
    if os.name != "nt" and target_dir.exists():
        try:
            for root, dirs, files in os.walk(target_dir):
                for d in dirs:
                    try:
                        os.chmod(os.path.join(root, d), stat.S_IRWXU)
                    except Exception:
                        pass
                for f in files:
                    try:
                        os.chmod(os.path.join(root, f), stat.S_IRUSR | stat.S_IWUSR)
                    except Exception:
                        pass
        except Exception:
            pass

    _SETUP_DONE = True
    return all_available


# Automatically configure environment upon import
setup_secure_nltk_directory()

