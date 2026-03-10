"""Test dependency version constraints."""

import ast
import os


def test_peft_version_constraint():
    """Test that peft has proper version constraint in setup.py."""
    setup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "setup.py")

    with open(setup_path, "r") as f:
        content = f.read()

    # Check that peft has version constraints
    assert "peft>=" in content, "peft should have a minimum version constraint"
    assert "peft" in content and "<0.18" in content, "peft should have an upper bound < 0.18"


if __name__ == "__main__":
    test_peft_version_constraint()
    print("Dependency version constraint test passed!")
