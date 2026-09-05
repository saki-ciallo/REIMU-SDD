from __future__ import annotations

import unittest

from add_system.configuration.blocks import (
    BlockSpec,
    FeedForwardType,
    MixerType,
    expand_block_specs,
    normalize_block_specs,
)


class BlockSpecTests(unittest.TestCase):
    def test_block_spec_normalizes_mapping(self) -> None:
        spec = BlockSpec.from_raw(
            {"attn": "attention", "mlp": "moe", "num_layers": 3},
            location="blocks[0]",
        )

        self.assertEqual(
            spec,
            BlockSpec(
                mixer=MixerType.ATTENTION,
                feed_forward=FeedForwardType.MOE,
                num_layers=3,
            ),
        )
        self.assertEqual(
            spec.to_legacy_dict(),
            {
                "attn": "attention",
                "mlp": "moe",
                "num_layers": 3,
            },
        )

    def test_block_mapping_uses_numeric_key_order(self) -> None:
        specs = normalize_block_specs(
            {
                "10": {"attn": "mamba3", "mlp": "mlp"},
                "2": {"attn": "gdn2", "mlp": "moe"},
            }
        )

        self.assertEqual(
            [spec.mixer for spec in specs],
            [MixerType.GDN2, MixerType.MAMBA3],
        )

    def test_expand_block_specs_returns_one_spec_per_layer(self) -> None:
        grouped = normalize_block_specs([{"attn": "raven", "mlp": "mlp", "num_layers": 2}])

        expanded = expand_block_specs(grouped)

        self.assertEqual(len(expanded), 2)
        self.assertTrue(all(spec.num_layers == 1 for spec in expanded))

    def test_invalid_block_specs_fail_early(self) -> None:
        cases: list[tuple[dict[str, object], type[Exception]]] = [
            ({"attn": "unknown", "mlp": "mlp"}, ValueError),
            ({"attn": "attention", "mlp": "unknown"}, ValueError),
            ({"attn": "attention"}, ValueError),
            ({"attn": "attention", "mlp": "mlp", "num_layers": 0}, ValueError),
            ({"attn": "attention", "mlp": "mlp", "extra": True}, ValueError),
        ]
        for raw, error_type in cases:
            with self.subTest(raw=raw), self.assertRaises(error_type):
                BlockSpec.from_raw(raw, location="blocks[0]")


if __name__ == "__main__":
    unittest.main()
