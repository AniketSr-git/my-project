import sys, pathlib
# Make sure Python can see the src/ folder
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from example import add

def test_add():
    assert add(2, 3) == 5
