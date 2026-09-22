import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dafr.adaptation import adaptation_objective, configure_adaptation, prototype_loss
from dafr.checkpoint import load_checkpoint, save_checkpoint
from dafr.data import PairedManifestDataset
from dafr.forensic_views import ForensicTransform
from dafr.model import DAFR
from dafr.objectives import base_objective, decorrelation_loss, supervised_contrastive_loss
from dafr.registry import SourceRegistry


class CoreTest(unittest.TestCase):
    def test_source_metric_losses(self):
        embeddings = torch.randn(4, 256, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = prototype_loss(embeddings, labels) + supervised_contrastive_loss(
            embeddings, labels
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_pipeline(self):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = ForensicTransform(size=32)
        images = [Image.fromarray(np.random.default_rng(i).integers(
            0, 256, size=(40, 40, 3), dtype=np.uint8
        )) for i in range(2)]
        views = [transform(image) for image in images]
        batch = {name: torch.stack([item[name] for item in views]).to(device)
                 for name in views[0]}
        model = DAFR(source_names=("generator_a", "generator_b")).to(device).eval()
        with torch.inference_mode():
            output = model(batch, include_auxiliary=True)
            losses = base_objective(
                output,
                torch.tensor([0, 1], device=device),
                torch.tensor([-1, 0], device=device),
            )
        self.assertEqual(tuple(output["source_embedding"].shape), (2, 256))
        self.assertTrue(torch.allclose(output["expert_weights"].sum(1),
                                       torch.ones(2, device=device), atol=1e-5))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(decorrelation_loss(
            output["authenticity_embedding"], output["source_embedding"]
        )))

        configure_adaptation(model)
        output = model(batch)
        objective = adaptation_objective(
            output,
            torch.tensor([0, 1], device=device),
            torch.tensor([-1, 0], device=device),
        )
        objective["total"].backward()
        self.assertIsNotNone(model.source_residual_adapter.up.weight.grad)
        self.assertTrue(all(not p.requires_grad for p in model.expert_encoders.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_checkpoint(model, checkpoint)
            restored = load_checkpoint(checkpoint, device)
            self.assertEqual(restored.source_names, model.source_names)
            reference = Path(directory) / "reference.png"
            images[1].save(reference)
            pair = PairedManifestDataset(
                [{"path": str(reference), "binary_label": "1", "source": "generator_a"}],
                ("generator_a",), degradation_probability=1.0,
            )[0]
            self.assertEqual(set(pair[0]), set(pair[1]))
            self.assertEqual(tuple(pair[0]["rgb"].shape), (3, 256, 256))
            registry = SourceRegistry()
            registry.register("generator_a", [reference])
            registry.refresh(model, device, transform)
            ranking = registry.rank(registry.embeddings[0])
            self.assertEqual(ranking[0]["source"], "generator_a")
            self.assertAlmostEqual(ranking[0]["similarity"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
