import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "perturbation: синтетическое возмущение ракурса (perspective/mirror/rotation)")
    config.addinivalue_line("markers", "slow: долгий тест")
