"""Tests for CLI entry point."""

import io
import unittest
from unittest.mock import patch
from coding_agent.cli import main


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""

    def test_main_returns_zero(self):
        """Ensure main function executes and returns 0 exit code."""
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            result = main()
            self.assertEqual(result, 0)
            output = fake_out.getvalue()
            self.assertIn("Terminal Coding Agent initialized", output)
            self.assertIn("Model Provider URL:", output)
            self.assertIn("Model Name:", output)


if __name__ == "__main__":
    unittest.main()
