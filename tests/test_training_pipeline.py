from __future__ import annotations

import torch

from latticememory.training import (
    E8RoutingLoss,
    RoutingTrainingExample,
    build_msmarco_examples,
    evaluate_routing_examples,
    train_lattice_adapter_from_examples,
)


class TinyKeywordEncoder:
    def __init__(self):
        self.d_model = 8
        self._basis = {
            "france": 0,
            "paris": 0,
            "hamlet": 1,
            "shakespeare": 1,
            "everest": 2,
            "mountain": 2,
            "mars": 3,
            "planet": 3,
        }

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        rows = []
        for text in sentences:
            vec = torch.zeros(self.d_model, dtype=torch.float32)
            lower = str(text).lower()
            for token, dim in self._basis.items():
                if token in lower:
                    vec[dim] = 1.0
            if vec.sum() == 0:
                vec[-1] = 1.0
            rows.append(vec)
        return torch.stack(rows).numpy()


def test_build_msmarco_examples_extracts_selected_passages_and_negatives():
    rows = [
        {
            "query": "what city is the capital of france",
            "passages": {
                "is_selected": [0, 1, 0],
                "passage_text": [
                    "Berlin is the capital of Germany.",
                    "Paris is the capital of France.",
                    "Madrid is the capital of Spain.",
                ],
            },
        }
    ]

    examples = build_msmarco_examples(rows, min_negatives=1)

    assert examples == [
        RoutingTrainingExample(
            query="what city is the capital of france",
            positive="Paris is the capital of France.",
            negatives=[
                "Berlin is the capital of Germany.",
                "Madrid is the capital of Spain.",
            ],
        )
    ]


def test_build_msmarco_examples_accepts_triplet_rows():
    rows = [
        {
            "query": "who wrote hamlet",
            "positive": "William Shakespeare wrote Hamlet.",
            "negative": "Charles Dickens wrote Oliver Twist.",
        }
    ]

    examples = build_msmarco_examples(rows)

    assert examples == [
        RoutingTrainingExample(
            query="who wrote hamlet",
            positive="William Shakespeare wrote Hamlet.",
            negatives=["Charles Dickens wrote Oliver Twist."],
        )
    ]


def test_e8_routing_loss_backpropagates_to_query_embeddings():
    torch.manual_seed(7)
    query_embeddings = torch.randn(4, 16, requires_grad=True)
    positive_embeddings = query_embeddings.detach().clone() + 0.01 * torch.randn(4, 16)
    negative_embeddings = torch.randn(4, 2, 16)
    loss_fn = E8RoutingLoss(d_model=16, lambda_address=0.5, lambda_hamming=0.1)

    output = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)
    output.total.backward()

    assert output.total.item() > 0
    assert output.contrastive.item() > 0
    assert output.address.item() > 0
    assert query_embeddings.grad is not None
    assert torch.isfinite(query_embeddings.grad).all()


def test_train_and_evaluate_examples_reports_lattice_routes():
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", ["Mars is a planet."]),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", ["Paris is in France."]),
        RoutingTrainingExample("which mountain is tallest", "Mount Everest is the tallest mountain.", ["Hamlet is a play."]),
        RoutingTrainingExample("what is the red planet", "Mars is known as the red planet.", ["Mount Everest is tall."]),
    ]

    train_result = train_lattice_adapter_from_examples(
        base_encoder=TinyKeywordEncoder(),
        examples=examples,
        d_model=8,
        epochs=4,
        batch_size=2,
        lr=1e-2,
        lambda_address=1.0,
        device="cpu",
    )
    metrics = evaluate_routing_examples(train_result.dual_encoder, examples, top_k=1)

    assert metrics["total"] == 4
    assert metrics["correct_at_1"] == 4
    assert metrics["lattice_route_rate"] == 1.0
    assert metrics["path_counts"]["lattice_exact"] == 4
    assert "mean_hamming_distance" in metrics
    assert "hamming_distance_histogram" in metrics
    assert "hamming_distance_to_positive" in metrics["rows"][0]
