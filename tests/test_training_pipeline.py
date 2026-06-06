from __future__ import annotations

import torch

from latticememory.training import (
    E8RoutingLoss,
    RoutingTrainingExample,
    _main,
    _encode_trainable_texts,
    _move_features_to_device,
    build_msmarco_examples,
    evaluate_routing_examples,
    load_sts_examples,
    train_and_evaluate_msmarco,
    train_full_encoder_from_examples,
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


class CountingTinyKeywordEncoder(TinyKeywordEncoder):
    def __init__(self):
        super().__init__()
        self.encoded_texts: list[str] = []

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        inputs = [sentences] if isinstance(sentences, str) else list(sentences)
        self.encoded_texts.extend(str(text) for text in inputs)
        return super().encode(inputs, batch_size=batch_size, **kwargs)


class TinyTrainableTextEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 8):
        super().__init__()
        self.d_model = d_model
        self.ids: dict[str, int] = {}
        self.embeddings = torch.nn.Embedding(32, d_model)
        torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        self.saved_to = None

    def _id_for(self, text: str) -> int:
        if text not in self.ids:
            self.ids[text] = len(self.ids)
        return self.ids[text]

    def tokenize(self, texts, **kwargs):
        ids = [self._id_for(str(text)) for text in texts]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

    def forward(self, features, **kwargs):
        return {"sentence_embedding": self.embeddings(features["input_ids"])}

    def encode(self, sentences, batch_size: int = 64, **kwargs):
        is_single = isinstance(sentences, str)
        texts = [sentences] if is_single else list(sentences)
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                features = self.tokenize(texts[start : start + batch_size])
                rows.append(self.forward(features)["sentence_embedding"])
        encoded = torch.cat(rows, dim=0).detach().cpu().numpy()
        return encoded[0] if is_single else encoded

    def save(self, path: str):
        self.saved_to = path


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
    assert metrics["unique_document_keys"] >= 1
    assert metrics["max_documents_per_key"] >= 1
    assert metrics["documents_in_collision_keys"] >= metrics["collision_key_count"]


def test_training_records_sts_metrics_each_epoch():
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", ["Mars is a planet."]),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", ["Paris is in France."]),
        RoutingTrainingExample("which mountain is tallest", "Mount Everest is the tallest mountain.", ["Hamlet is a play."]),
        RoutingTrainingExample("what is the red planet", "Mars is known as the red planet.", ["Mount Everest is tall."]),
    ]
    sts_examples = load_sts_examples(source="builtin", limit=4)

    train_result = train_lattice_adapter_from_examples(
        base_encoder=TinyKeywordEncoder(),
        examples=examples,
        d_model=8,
        epochs=3,
        batch_size=2,
        lr=1e-2,
        lambda_address=1.0,
        device="cpu",
        sts_examples=sts_examples,
    )

    assert len(train_result.train_result.epoch_metrics) == 3
    assert [row["epoch"] for row in train_result.train_result.epoch_metrics] == [1, 2, 3]
    for row in train_result.train_result.epoch_metrics:
        assert row["metric"] == "sts"
        assert row["count"] == 4
        assert "pearson" in row
        assert "spearman" in row


def test_lambda_hard_zero_skips_negative_text_encoding():
    encoder = CountingTinyKeywordEncoder()
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", ["UNIQUE_NEGATIVE_MARKER"]),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", ["ANOTHER_NEGATIVE_MARKER"]),
    ]

    train_lattice_adapter_from_examples(
        base_encoder=encoder,
        examples=examples,
        d_model=8,
        epochs=1,
        batch_size=2,
        lambda_hard=0.0,
        device="cpu",
    )

    assert "UNIQUE_NEGATIVE_MARKER" not in encoder.encoded_texts
    assert "ANOTHER_NEGATIVE_MARKER" not in encoder.encoded_texts


def test_full_encoder_training_computes_live_target_keys_each_batch(monkeypatch):
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", []),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", []),
        RoutingTrainingExample("which mountain is tallest", "Mount Everest is the tallest mountain.", []),
        RoutingTrainingExample("what is the red planet", "Mars is known as the red planet.", []),
    ]
    target_key_calls = []
    original_target_keys = E8RoutingLoss._target_keys

    def tracking_target_keys(self, positive_embeddings):
        target_key_calls.append(positive_embeddings.detach().clone())
        return original_target_keys(self, positive_embeddings)

    monkeypatch.setattr(E8RoutingLoss, "_target_keys", tracking_target_keys)

    result = train_full_encoder_from_examples(
        encoder=TinyTrainableTextEncoder(),
        examples=examples,
        d_model=8,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        lambda_address=2.0,
        lambda_neighborhood=0.1,
        lambda_hard=0.0,
        device="cpu",
    )

    assert len(target_key_calls) == 4
    assert result.training_mode == "full_encoder"
    assert len(result.train_min_hamming_history) == 2
    assert len(result.train_lattice_route_rate_history) == 2


def test_full_encoder_training_records_sts_metrics_each_epoch(tmp_path):
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", []),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", []),
    ]
    metrics_path = tmp_path / "epoch_metrics.jsonl"

    result = train_full_encoder_from_examples(
        encoder=TinyTrainableTextEncoder(),
        examples=examples,
        d_model=8,
        epochs=2,
        batch_size=1,
        lr=1e-2,
        lambda_address=1.0,
        lambda_hard=0.0,
        sts_examples=load_sts_examples(source="builtin", limit=2),
        epoch_metrics_path=metrics_path,
        device="cpu",
    )

    assert [row["epoch"] for row in result.epoch_metrics] == [1, 2]
    assert metrics_path.read_text(encoding="utf-8").count("\n") == 2


def test_full_encoder_training_writes_progress_metrics(tmp_path):
    examples = [
        RoutingTrainingExample("what city is the capital of france", "Paris is the capital of France.", []),
        RoutingTrainingExample("name the author of hamlet", "William Shakespeare wrote Hamlet.", []),
    ]
    progress_path = tmp_path / "progress_metrics.jsonl"

    train_full_encoder_from_examples(
        encoder=TinyTrainableTextEncoder(),
        examples=examples,
        d_model=8,
        epochs=1,
        batch_size=1,
        lr=1e-2,
        lambda_address=1.0,
        lambda_hard=0.0,
        progress_metrics_path=progress_path,
        log_every_batches=1,
        device="cpu",
    )

    rows = [line for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert '"metric": "train_progress"' in rows[0]


def test_train_msmarco_cli_accepts_full_encoder_mode(monkeypatch, tmp_path):
    calls = []

    def fake_train_and_evaluate_msmarco(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "latticememory.training.train_and_evaluate_msmarco",
        fake_train_and_evaluate_msmarco,
    )

    assert _main([
        "--training-mode",
        "full_encoder",
        "--output-dir",
        str(tmp_path),
    ]) == 0

    assert calls[0]["training_mode"] == "full_encoder"


def test_move_features_to_device_handles_tokenizer_batchencoding():
    from transformers.tokenization_utils_base import BatchEncoding

    features = BatchEncoding({"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])})
    moved = _move_features_to_device(features, torch.device("meta"))

    assert isinstance(moved, BatchEncoding)
    assert moved["input_ids"].device.type == "meta"


def test_encode_trainable_texts_reads_sentence_embedding_from_batchencoding():
    from transformers.tokenization_utils_base import BatchEncoding

    class BatchEncodingForwardEncoder(TinyTrainableTextEncoder):
        def forward(self, features, **kwargs):
            return BatchEncoding({"sentence_embedding": self.embeddings(features["input_ids"])})

    encoded = _encode_trainable_texts(
        BatchEncodingForwardEncoder(),
        ["hello"],
        device=torch.device("cpu"),
        d_model=8,
    )

    assert encoded.shape == (1, 8)
