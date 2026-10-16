# Batch 06 review

Fixed code51ba007/controller9447df6/rails07; low effort unchanged. Full gate
476passed1skip,experiments18passed. Two failed tables rerun; neither qualifies.
45requests,130129input+103502output (88400reportedreasoning,not added),
$0.115940568. Cumulative1127requests,7047437tokens,$4.4820545792.
Series10/12 at least once; acceptance and ablation pending.

## Optical Digits: improved validation, failing embedding implementation
Validation completes without undeclared per-row files. Test computes a metric
but both runnable attempts fail when embedding is None. Generated code uses
sklearn.neural_network._base.ACTIVATIONS[e.activation](...), assigns its return
and assumes a matrix. Those private helpers mutate their argument and return
None. It also uses only the first hidden layer, not the configured bottleneck.
This is a generic generated-code/library-contract failure. Refine representation
rail to use public activation results and explicitly return a finite 2D
bottleneck matrix; do not patch or rerun this individual script manually.
No hidden-label probe: no completed test embedding artifact exists.

## Retail: valid preparation recommendation now executes
Navigator chooses build_entities, and it completes. Prepared columns are
entity_id,recency_days,frequency,total_value,mean_value,n_invoices. Setup
still declares groupCustomerID/timeInvoiceDate. Both are missing from the
prepared table. Joint splitter correctly refuses instead of silently dropping
constraints. The preparation prompt only required an explicit entity ID,
without requiring the declared name, so the Worker renamed it. Fix this generic
output contract and validate it during collection, enabling normal Worker
repair before marking preparation completed. Keep constraints strict. Clarify
that declared time fields apply to final analysis units, not merely raw event
timestamps used to aggregate static features. Do not inject manifest answers.

## Next
Consolidate output-contract repairs and instruction refinements. Prototype
compact fingerprints outside active source: synthetic cells/full-row hashes
and privacy verdicts match; historicalv2 read/write remains unchanged.
17 existing synthetic privacy tests passed against prototype (two schema/plan
tests intentionally excluded for adaptation before integration). Fixturecache
13.15MB versus1.66MB; no claim about actual Covertype size yet. Integrate with
permanent migration/equivalence tests and full gate before more live calls,
needed because45GB Covertype cache exceeds current39GB free disk for acceptance.
No primary artifacts removed. All old scores preserved, extra-constraint
scientific question remains pending. No held-out/bio/paper/push/freeze/release.
