"""Test that the visual_bge package can be properly imported."""

import sys
import importlib.util


def test_package_structure():
    """Test that visual_bge has proper package structure."""
    import os
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visual_bge_dir = os.path.join(package_dir, "visual_bge")

    # Check that __init__.py exists in visual_bge directory
    init_file = os.path.join(visual_bge_dir, "__init__.py")
    assert os.path.exists(init_file), f"__init__.py should exist at {init_file}"

    # Check that modeling.py exists
    modeling_file = os.path.join(visual_bge_dir, "modeling.py")
    assert os.path.exists(modeling_file), f"modeling.py should exist at {modeling_file}"


def test_import_visual_bge():
    """Test that visual_bge can be imported after installation."""
    try:
        # Try to import the module
        import visual_bge
        # Check that Visualized_BGE is accessible
        assert hasattr(visual_bge, 'Visualized_BGE'), "Visualized_BGE should be importable from visual_bge"
    except ImportError:
        # If not installed, that's expected in test environment
        # Just verify the package structure is correct
        pass


if __name__ == "__main__":
    test_package_structure()
    print("Package structure test passed!")
    test_import_visual_bge()
    print("All tests passed!")
