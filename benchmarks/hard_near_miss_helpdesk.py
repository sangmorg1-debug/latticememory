"""Hard near-miss challenge dataset: IT/HR helpdesk domain.

A second domain for multi-domain proof hardening. Intent pairs are chosen to be
genuinely confusable (same surface vocabulary, different intent) so the safety
gate is tested against a harder population than the single customer-support domain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTENTS: list[dict[str, Any]] = [
    {
        "intent_id": "reset_vpn_access",
        "canonical": "How do I reset my VPN access?",
        "answer": "Submit a VPN reset request via the IT portal. Access is restored within one business day after identity verification.",
        "paraphrases": [
            "I can't connect to the VPN",
            "VPN credentials stopped working",
            "Renew my remote access",
            "Restore VPN login",
            "Fix my VPN connection issue",
            "I need new VPN credentials",
            "Reset my remote network access",
            "Get VPN working again",
        ],
    },
    {
        "intent_id": "request_new_laptop",
        "canonical": "How do I request a new laptop?",
        "answer": "Submit a hardware request in the IT portal. Approval from your manager is required before provisioning.",
        "paraphrases": [
            "Order a new computer for me",
            "I need a replacement laptop",
            "Request a work laptop",
            "Get a new device provisioned",
            "My laptop needs replacing",
            "Submit equipment request",
            "Can I get a new machine?",
            "Hardware request for a laptop",
        ],
    },
    {
        "intent_id": "report_laptop_stolen",
        "canonical": "My laptop was stolen, what do I do?",
        "answer": "Report the theft to IT Security immediately via the security hotline. We will remotely wipe the device and issue a replacement.",
        "paraphrases": [
            "Someone took my work computer",
            "Lost my laptop, it may have been stolen",
            "Work device was taken",
            "My laptop is missing and I think it was stolen",
            "Stolen company laptop procedure",
            "What happens if my laptop is stolen?",
            "Report a missing device",
            "Company laptop theft report",
        ],
    },
    {
        "intent_id": "request_software_license",
        "canonical": "How do I request a software license?",
        "answer": "Software license requests are submitted via the IT portal and require business justification and manager approval.",
        "paraphrases": [
            "Get access to licensed software",
            "Request a tool license for my team",
            "I need a software subscription",
            "How do I get access to a paid application?",
            "Software access request procedure",
            "License request for a new app",
            "How do I get a seat for a program?",
            "Obtain software for my project",
        ],
    },
    {
        "intent_id": "revoke_software_license",
        "canonical": "How do I revoke a software license for a departing employee?",
        "answer": "Submit a license revocation request when offboarding an employee. IT will deactivate the seat within 24 hours.",
        "paraphrases": [
            "Remove software access from a leaving employee",
            "Offboard software license",
            "Cancel a seat for someone who left",
            "Deactivate tool access for departing team member",
            "How do I remove a license when someone leaves?",
            "Software license cleanup for offboarding",
            "Revoke application access after termination",
            "Remove a user from a software subscription",
        ],
    },
    {
        "intent_id": "submit_expense_report",
        "canonical": "How do I submit an expense report?",
        "answer": "Expense reports are submitted in the HR portal under Finance > Expenses. Attach receipts and get manager sign-off within 30 days.",
        "paraphrases": [
            "How do I get reimbursed for a work expense?",
            "File my expenses from the business trip",
            "Submit receipts for reimbursement",
            "Expense claim process",
            "I have business expenses to report",
            "Reimbursement request for work costs",
            "How do I claim back travel costs?",
            "Log my expenses in the HR system",
        ],
    },
    {
        "intent_id": "request_expense_advance",
        "canonical": "How do I request a travel expense advance?",
        "answer": "Expense advances for travel must be requested at least 5 business days before the trip via the HR Finance portal.",
        "paraphrases": [
            "Get advance payment for business travel",
            "Can I get money upfront for a work trip?",
            "Pre-trip expense advance request",
            "Travel advance before the trip",
            "How do I get prepaid for travel costs?",
            "Upfront reimbursement for upcoming work trip",
            "Advance for conference travel",
            "Pre-approval advance for business expenses",
        ],
    },
    {
        "intent_id": "request_leave",
        "canonical": "How do I request time off?",
        "answer": "Submit a leave request in the HR portal under Time & Attendance. Requests require at least 2 weeks notice for planned leave.",
        "paraphrases": [
            "Book annual leave",
            "How do I take a vacation day?",
            "Request PTO",
            "Apply for time off",
            "Schedule holiday leave",
            "Put in a leave request",
            "How do I take a day off?",
            "Submit PTO request in the system",
        ],
    },
    {
        "intent_id": "check_leave_balance",
        "canonical": "How do I check my remaining leave balance?",
        "answer": "Leave balances are visible in the HR portal under My Profile > Time Off.",
        "paraphrases": [
            "How many vacation days do I have left?",
            "Check my PTO balance",
            "Remaining holiday allowance",
            "How much annual leave is left?",
            "View my time off balance",
            "PTO days remaining",
            "See how many leave days I have",
            "Check available time off",
        ],
    },
    {
        "intent_id": "update_emergency_contact",
        "canonical": "How do I update my emergency contact?",
        "answer": "Emergency contact information can be updated in HR portal under My Profile > Personal Information.",
        "paraphrases": [
            "Change my next of kin details",
            "Update who to call in an emergency",
            "Edit emergency contact info in HR",
            "Add a new emergency contact",
            "Replace my emergency contact person",
            "Change emergency contact on file",
            "Update my HR personal record emergency contact",
            "How do I edit my emergency contact details?",
        ],
    },
    {
        "intent_id": "enroll_benefits",
        "canonical": "How do I enroll in employee benefits?",
        "answer": "Benefits enrollment is available during the open enrollment window in the HR portal under Benefits. Contact HR for late enrollment exceptions.",
        "paraphrases": [
            "Sign up for health insurance",
            "Enroll in the company benefits plan",
            "How do I get company health coverage?",
            "Benefits sign-up process",
            "Add dental coverage to my benefits",
            "How do I choose my employee benefits?",
            "Select benefits package",
            "Opt into company healthcare plan",
        ],
    },
    {
        "intent_id": "change_benefits",
        "canonical": "How do I change my benefits selections?",
        "answer": "Benefits changes outside open enrollment require a qualifying life event such as marriage, birth, or loss of other coverage.",
        "paraphrases": [
            "Modify my health plan mid-year",
            "Change my insurance plan",
            "Switch benefit options outside open enrollment",
            "Update benefits for a life event",
            "Can I change healthcare coverage now?",
            "Adjust my benefits after a life change",
            "How do I update my benefits selection?",
            "Alter insurance enrollment mid-year",
        ],
    },
]

NEAR_MISS_INTENT_PAIRS: list[tuple[str, str]] = [
    ("reset_vpn_access", "request_new_laptop"),
    ("request_new_laptop", "report_laptop_stolen"),
    ("request_software_license", "revoke_software_license"),
    ("submit_expense_report", "request_expense_advance"),
    ("request_leave", "check_leave_balance"),
    ("enroll_benefits", "change_benefits"),
    ("update_emergency_contact", "enroll_benefits"),
    ("reset_vpn_access", "report_laptop_stolen"),
    ("submit_expense_report", "request_leave"),
    ("request_software_license", "request_new_laptop"),
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
        "domain": "hard_helpdesk_near_misses",
        "intents": INTENTS,
        "paraphrases": paraphrases,
        "near_misses": near_misses,
    }


def _split_pairs(
    pairs: list[list[str]], train_count: int
) -> tuple[list[list[str]], list[list[str]]]:
    grouped_train = []
    grouped_heldout = []
    for idx, pair in enumerate(pairs):
        if idx % 8 < train_count:
            grouped_train.append(pair)
        else:
            grouped_heldout.append(pair)
    return grouped_train, grouped_heldout


def build_prompt_response_stream(dataset: dict[str, Any]) -> list[dict[str, str]]:
    """Build a deterministic cache simulation stream from canonical answers."""
    rows: list[dict[str, str]] = []
    for intent in dataset["intents"]:
        prompts = [intent["canonical"], *intent["paraphrases"][:5]]
        for prompt in prompts:
            rows.append(
                {
                    "intent_id": intent["intent_id"],
                    "prompt": prompt,
                    "response": intent["answer"],
                }
            )
    return rows


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
        "prompts_responses": output / "prompts_responses.json",
    }
    paths["source"].write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    paths["calibration"].write_text(
        json.dumps(
            {
                "domain": dataset["domain"],
                "source": "built_in_hard_near_miss_helpdesk",
                "paraphrases": cal_para,
                "near_misses": cal_near,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["heldout_paraphrases"].write_text(
        json.dumps({"paraphrases": heldout_para}, indent=2), encoding="utf-8"
    )
    paths["heldout_near_misses"].write_text(
        json.dumps({"near_misses": heldout_near}, indent=2), encoding="utf-8"
    )
    paths["prompts_responses"].write_text(
        json.dumps(build_prompt_response_stream(dataset), indent=2), encoding="utf-8"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the built-in hard near-miss helpdesk challenge dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/demo_data/hard_near_miss_helpdesk",
    )
    args = parser.parse_args()
    paths = write_challenge_dataset(args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
