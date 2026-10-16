# Round 1 review

Development evidence only. All original responses, run records and logged categories
are preserved unchanged. These classifications are annotations after the round.

**0/12 completed; 12 requests; no Worker execution and no fitted analysis.**
Eleven failures exposed the setup contract defect: unrestricted metric strings
and alternative dictionaries were accepted by the advertised schema but rejected
by later engine checks outside bounded structured repair. One response exhausted
its 16,384-token output limit before returning a proposal. None is evidence of
predictive performance or successful orchestration.

| Dataset | Original category | Reviewed classification | Reported model |
|---|---|---|---|
| breast_cancer_diagnostic | invalid_inference | engine input-contract defect | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free |
| heart_disease_cleveland | runtime_error:ValidationError | engine input-contract defect | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free |
| abalone | invalid_inference | engine input-contract defect | nvidia/nemotron-3-super-120b-a12b:free |
| energy_efficiency | invalid_inference | engine input-contract defect | dots-studio/dots-3-note-preview:free |
| seeds | runtime_error:ValidationError | engine input-contract defect | dots-studio/dots-3-note-preview:free |
| dry_bean | invalid_inference | engine input-contract defect | nvidia/nemotron-3.5-lightning:free |
| optical_digits | runtime_error:ValidationError | engine input-contract defect | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free |
| adult | invalid_inference | engine input-contract defect | poolside/laguna-s-2.1:free |
| covertype | output_truncated | agent output limit | poolside/laguna-s-2.1:free |
| anuran_calls | invalid_inference | engine input-contract defect | dots-studio/dots-3-note-preview:free |
| bike_sharing_hourly | runtime_error:ValidationError | engine input-contract defect | cohere/north-mini-code:free |
| online_retail | runtime_error:ValidationError | engine input-contract defect | nvidia/nemotron-3-super-120b-a12b:free |

## Accounting

- Provider-reported aggregate input_tokens: 17340.
- Provider-reported aggregate output_tokens: 44055.
- Provider-reported aggregate reasoning_tokens: 39694.
- Provider-reported aggregate latency_ms: 1193541.971734999.
- Provider-reported aggregate cost_usd: 0.0.

All twelve envelopes report these totals; they are not independently reconciled
billing figures. The provider reported more reasoning than completion tokens in
one envelope; original figures are kept without arithmetic correction. Structural
inference was unavailable for all runs; no project or external prediction scorer
ran. Zero Worker attempts/repairs; setup did not reach any workflow event.

## Corrections to verify in a fresh round

B-0001: typed complete nonrecursive alternatives, exact allowed metric IDs, local
column/type validation inside bounded structured repair, and retention of known
setup failure usage/model mix in summary. Engine input-contract/accounting fix;
no weakened safeguard and no oracle targets supplied.

L-0001 / RL-0001: request a concise setup proposal and only genuinely ambiguous
alternatives. Generic instruction, not a guarantee against hidden reasoning or
provider truncation. Fresh verification pending; no retrospective success claim.

## Provider timing

A successful response took about 459 seconds with timeout_s=90. That setting is an
SDK I/O timeout, not a strict wall-clock deadline. Retain the actual measured
latencies. Reported request costs total $0. No held-out files were opened.
