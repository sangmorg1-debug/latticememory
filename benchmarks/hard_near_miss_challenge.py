"""Hard near-miss challenge set for HammingRouter safety experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTENTS: list[dict[str, Any]] = [
    {
        "intent_id": "cancel_subscription",
        "canonical": "How do I cancel my subscription?",
        "answer": "You can cancel from billing settings. Access remains active until the end of the paid period.",
        "paraphrases": [
            "I want to cancel my plan",
            "Stop my subscription renewal",
            "How can I end my membership?",
            "Cancel my monthly service",
            "Turn off my subscription",
            "I need to stop being charged for my plan",
            "End my paid account",
            "Please cancel my recurring plan",
        ],
    },
    {
        "intent_id": "pause_subscription",
        "canonical": "How do I pause my subscription?",
        "answer": "You can pause eligible plans from billing settings and resume later without creating a new account.",
        "paraphrases": [
            "Can I temporarily stop my plan?",
            "Pause my monthly subscription",
            "Put my membership on hold",
            "Can I suspend billing for a while?",
            "I want to pause instead of cancel",
            "Freeze my subscription for a month",
            "Stop my plan temporarily",
            "Can I hold my account without deleting it?",
        ],
    },
    {
        "intent_id": "refund_request",
        "canonical": "How do I request a refund?",
        "answer": "Refund requests can be submitted from the order or billing page and are reviewed under the refund policy.",
        "paraphrases": [
            "Can I get my money back?",
            "I want a refund for my payment",
            "How do I ask for a charge refund?",
            "Refund my order",
            "Please return my payment",
            "I was charged and need a refund",
            "Where do I submit a refund request?",
            "Can you reverse this charge?",
        ],
    },
    {
        "intent_id": "return_status",
        "canonical": "How do I check my return status?",
        "answer": "Return status is available from the returns page using your order number or account order history.",
        "paraphrases": [
            "Where is my return?",
            "Track my returned item",
            "Has my return been received?",
            "Check the status of my return shipment",
            "Did you get my returned package?",
            "Return tracking update",
            "What happened to my return?",
            "Is my return processed yet?",
        ],
    },
    {
        "intent_id": "reset_password",
        "canonical": "How do I reset my password?",
        "answer": "Use the password reset link on the sign-in page and follow the email instructions.",
        "paraphrases": [
            "I forgot my password",
            "Send me a password reset",
            "I cannot remember my login password",
            "Help me reset account password",
            "Password recovery help",
            "How do I create a new password?",
            "I need a reset link",
            "Forgot password for my account",
        ],
    },
    {
        "intent_id": "change_email",
        "canonical": "How do I change my account email?",
        "answer": "Account email changes can be made from profile settings and may require verification of both addresses.",
        "paraphrases": [
            "Update my email address",
            "Change the email on my account",
            "I need to use a different login email",
            "How do I replace my account email?",
            "Switch my email address",
            "Edit my profile email",
            "Use a new email for my account",
            "Change where account emails go",
        ],
    },
    {
        "intent_id": "shipping_status",
        "canonical": "Where is my order?",
        "answer": "Order tracking is available from your order history or the shipping confirmation email.",
        "paraphrases": [
            "Track my package",
            "When will my order arrive?",
            "Shipping status for my order",
            "Where is my shipment?",
            "Delivery tracking update",
            "Has my package shipped?",
            "Check order delivery status",
            "Find my package location",
        ],
    },
    {
        "intent_id": "change_delivery_address",
        "canonical": "How do I change my delivery address?",
        "answer": "Delivery address changes are available before shipment from the order details page.",
        "paraphrases": [
            "Update the shipping address",
            "Send my order to a different address",
            "Can I change where my package goes?",
            "Fix the delivery address",
            "I entered the wrong shipping address",
            "Change address before it ships",
            "Redirect my package",
            "Edit my order address",
        ],
    },
    {
        "intent_id": "invoice_copy",
        "canonical": "How do I get a copy of my invoice?",
        "answer": "Invoices can be downloaded from billing history in your account.",
        "paraphrases": [
            "Download my receipt",
            "Where can I find my invoice?",
            "I need a copy of my bill",
            "Send me my payment receipt",
            "Get invoice for my subscription",
            "Print my billing invoice",
            "Where is my receipt?",
            "Can I download last month's invoice?",
        ],
    },
    {
        "intent_id": "update_payment_method",
        "canonical": "How do I update my payment method?",
        "answer": "You can update cards or payment details from billing settings before the next renewal.",
        "paraphrases": [
            "Change my credit card",
            "Update billing card",
            "Use a different payment method",
            "Replace my card on file",
            "Edit payment details",
            "Add a new billing card",
            "How do I change payment info?",
            "Update my subscription payment",
        ],
    },
    {
        "intent_id": "delete_account",
        "canonical": "How do I delete my account?",
        "answer": "Account deletion can be requested from privacy settings and may permanently remove account data.",
        "paraphrases": [
            "Permanently remove my account",
            "Delete all my account data",
            "Close my account forever",
            "I want my account erased",
            "Remove my profile permanently",
            "Delete my user account",
            "How do I erase my account?",
            "Submit account deletion request",
        ],
    },
    {
        "intent_id": "deactivate_account",
        "canonical": "How do I deactivate my account?",
        "answer": "Deactivation hides or disables the account without necessarily deleting all data.",
        "paraphrases": [
            "Temporarily disable my account",
            "Deactivate my profile",
            "Can I turn off my account for now?",
            "Disable account access",
            "Make my account inactive",
            "Suspend my account",
            "Hide my account without deleting it",
            "Deactivate instead of delete",
        ],
    },
]

NEAR_MISS_INTENT_PAIRS: list[tuple[str, str]] = [
    ("cancel_subscription", "pause_subscription"),
    ("refund_request", "return_status"),
    ("reset_password", "change_email"),
    ("shipping_status", "change_delivery_address"),
    ("invoice_copy", "update_payment_method"),
    ("delete_account", "deactivate_account"),
    ("cancel_subscription", "refund_request"),
    ("pause_subscription", "update_payment_method"),
    ("shipping_status", "return_status"),
    ("change_email", "delete_account"),
]


def build_challenge_dataset() -> dict[str, Any]:
    by_id = {intent["intent_id"]: intent for intent in INTENTS}
    paraphrases: list[list[str]] = []
    near_misses: list[list[str]] = []

    for intent in INTENTS:
        canonical = intent["canonical"]
        for paraphrase in intent["paraphrases"]:
            paraphrases.append([canonical, paraphrase])

    for left_id, right_id in NEAR_MISS_INTENT_PAIRS:
        left = by_id[left_id]
        right = by_id[right_id]
        left_prompts = [left["canonical"], *left["paraphrases"]]
        right_prompts = [right["canonical"], *right["paraphrases"]]
        for idx in range(5):
            near_misses.append([left_prompts[idx], right_prompts[(idx + 1) % len(right_prompts)]])

    return {
        "artifact_type": "latticememory_hard_near_miss_challenge",
        "artifact_version": 1,
        "domain": "hard_customer_support_near_misses",
        "intents": INTENTS,
        "paraphrases": paraphrases,
        "near_misses": near_misses,
    }


def _split_pairs(pairs: list[list[str]], train_count: int) -> tuple[list[list[str]], list[list[str]]]:
    grouped_train = []
    grouped_heldout = []
    for idx, pair in enumerate(pairs):
        if idx % 8 < train_count:
            grouped_train.append(pair)
        else:
            grouped_heldout.append(pair)
    return grouped_train, grouped_heldout


def write_challenge_dataset(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_challenge_dataset()
    cal_para, heldout_para = _split_pairs(dataset["paraphrases"], train_count=5)
    cal_near, heldout_near = _split_pairs(dataset["near_misses"], train_count=3)

    paths = {
        "source": output / "hard_near_miss_source.json",
        "calibration": output / "calibration_data.json",
        "heldout_paraphrases": output / "heldout_paraphrases.json",
        "heldout_near_misses": output / "heldout_near_misses.json",
    }
    paths["source"].write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    paths["calibration"].write_text(
        json.dumps(
            {
                "domain": dataset["domain"],
                "source": "built_in_hard_near_miss_challenge",
                "paraphrases": cal_para,
                "near_misses": cal_near,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["heldout_paraphrases"].write_text(json.dumps({"paraphrases": heldout_para}, indent=2), encoding="utf-8")
    paths["heldout_near_misses"].write_text(json.dumps({"near_misses": heldout_near}, indent=2), encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the built-in hard near-miss challenge dataset")
    parser.add_argument("--output-dir", default="benchmarks/demo_data/hard_near_miss_challenge")
    args = parser.parse_args()
    paths = write_challenge_dataset(args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
