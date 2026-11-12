import pytest

from dsmodel import demo


@pytest.fixture(scope="session")
def demo_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    return root, demo.make_demo_dataset(root)
