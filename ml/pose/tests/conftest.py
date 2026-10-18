"""Маркеры для тестов ml/pose (дублирует корневую регистрацию A01 — безопасно)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: тяжёлый тест (реальные библиотеки/модели), пропускается без них")
    config.addinivalue_line("markers", "gpu: нужен GPU")
