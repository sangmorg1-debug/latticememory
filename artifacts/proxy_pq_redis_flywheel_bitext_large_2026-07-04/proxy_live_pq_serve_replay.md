# LatticeMemory Live Proxy Replay Demo

| Metric | Value |
|---|---:|
| Total requests | 6000 |
| Hit rate | 0.9917 |
| Upstream call rate | 0.0083 |
| False-positive rate | 0.0000 |
| Adversarial false-positive rate | 0.0000 |
| Avg latency ms | 16.976 |

## Analytics

```json
{
  "cache_entries": 50,
  "estimated_savings_usd": 1.4875,
  "flywheel": {},
  "hamming_router": {
    "calibrated": false,
    "fp_rate": 0.0,
    "mode": "serve",
    "recall": 0.0
  },
  "hit_rate": 0.9917,
  "hits": 5950,
  "intent_distribution": {
    "cancel_order": 2,
    "change_order": 2,
    "change_shipping_address": 2,
    "check_cancellation_fee": 2,
    "check_invoice": 2,
    "check_payment_methods": 1,
    "check_refund_policy": 2,
    "complaint": 2,
    "contact_customer_service": 1,
    "contact_human_agent": 2,
    "create_account": 1,
    "delete_account": 2,
    "delivery_options": 2,
    "delivery_period": 2,
    "edit_account": 2,
    "get_invoice": 1,
    "get_refund": 2,
    "newsletter_subscription": 2,
    "payment_issue": 1,
    "place_order": 1,
    "recover_password": 1,
    "registration_problems": 2,
    "review": 2,
    "set_up_shipping_address": 2,
    "switch_account": 2,
    "track_order": 2,
    "track_refund": 2
  },
  "misses": 50,
  "rolling_hit_rate": 0.95,
  "total_events": 6550,
  "total_requests": 6000
}
```

This live replay validates proxy wiring and analytics. It does not prove RedisVL superiority,
general RAG superiority, or safety of raw PQ hits without validation.
