import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PluginContractTests(unittest.TestCase):
    def test_codex_manifest_uses_standard_skill_directory(self):
        manifest = json.loads(
            (ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["skills"], "./skills/")
        canonical = ROOT / "skills" / "ghost-decode"
        compatibility = ROOT / "codex-skills" / "ghost-decode"
        self.assertTrue((canonical / "SKILL.md").is_file())
        self.assertTrue((compatibility / "SKILL.md").is_file())
        self.assertEqual(
            (canonical / "SKILL.md").read_text(encoding="utf-8"),
            (compatibility / "SKILL.md").read_text(encoding="utf-8"),
        )
        canonical_agent = (canonical / "agents" / "openai.yaml").read_text(
            encoding="utf-8"
        )
        compatibility_agent = (
            compatibility / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")
        self.assertEqual(canonical_agent, compatibility_agent)

    def test_release_versions_are_aligned(self):
        codex = json.loads(
            (ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        claude = json.loads(
            (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        marketplace = json.loads(
            (ROOT / ".claude-plugin" / "marketplace.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(codex["version"], "1.4.1")
        self.assertEqual(claude["version"], codex["version"])
        self.assertEqual(marketplace["plugins"][0]["version"], codex["version"])

    def test_runtime_dependencies_use_headless_opencv(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("opencv-python-headless>=4.8", requirements)
        self.assertIsNone(
            re.search(r"(?m)^opencv-python(?:[<>=!~].*)?$", requirements)
        )

    def test_codex_skill_has_valid_minimal_frontmatter_and_consistent_rules(self):
        skill = (ROOT / "skills" / "ghost-decode" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        frontmatter = re.match(r"\A---\n(.*?)\n---\n", skill, re.DOTALL)
        self.assertIsNotNone(frontmatter)
        keys = {
            line.split(":", 1)[0]
            for line in frontmatter.group(1).splitlines()
            if ":" in line
        }
        self.assertEqual(keys, {"name", "description"})
        self.assertIn("one primary decode", skill)
        self.assertIn("at most one retry", skill)
        self.assertNotIn("exactly once", skill.lower())
        self.assertNotIn("## Decoder", skill)
        self.assertIn("<python> -m pip install -r", skill)
        self.assertIn("--method farneback --stride 2", skill)

    def test_skill_metadata_matches_the_skill_name(self):
        agent_yaml = (
            ROOT / "skills" / "ghost-decode" / "agents" / "openai.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn('display_name: "Ghost Font Decoder"', agent_yaml)
        self.assertIn("$ghost-decode", agent_yaml)


if __name__ == "__main__":
    unittest.main()
