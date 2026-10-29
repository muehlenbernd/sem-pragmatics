import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="run slow tests (e.g. .02 grid sweeps)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip = pytest.mark.skip(reason="use --slow to run")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip)
