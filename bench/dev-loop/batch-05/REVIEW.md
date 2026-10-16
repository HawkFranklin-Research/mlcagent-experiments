# Batch 05 review: reasoning-effort contrast

Fixed source identities exactly match batch04: codeafe46c3,controller1344fc8,
rails06,DeepSeek V4.1 Flash,temp0,max16384,limits unchanged. Only requested
effort changes from provider-default to low; failed-table queue shrinks from
five to three. Request records confirm reasoning={effort:low}. This is a
development contrast, not a randomized causal estimate; provider load and the
realized concurrent workload are uncontrolled.

One of three qualifies: Bike. Series10/12 once; acceptanceK3 not started.
84requests,200809input+215833output tokens;186651reasoning not added again.
Cost$0.2442439458,all core usage known. Cumulative1082requests,6813806tokens,
$4.3661140112. Reported model remains the fixed DeepSeek identifier.

## Outcomes and causal attribution
- Bike: report completed; group=None,time[dteday,hr],all external checks pass.
  39requests,2repairs,541.34s,$0.092760514. Batch04 used40requests,2repairs,
 848.89s,$0.151602591 but failed chronology. The existing engine bug is
 avoided by this setup, not fixed by reasoning effort.
- Digits: reached validation rather than batch04's EDA truncation. Three
 validation attempts: predict on PCA Pipeline; inverse_transform on MLP
 Pipeline; final script fixes both reconstruction branches, computes the
 metrics, then saves an UNDECLARED val_embeddings.parquet with row_id only
 as index and no partition column. Runtime correctly refuses. Validation
 template unnecessarily asks embedding extraction although only metrics
 are declared. Clarify candidate interfaces and outputs, preserve row guard.
 34requests,7repairs,680.75s,$0.132198635. Getting further costs more than
 failing earlier; do not call the larger cost a regression or improvement.
- Retail: split refuses missing group IDs after fallback distribution-only
 EDA. Both original and repaired Navigator replies recommend build_entities.
 First reply advances an unknown goal (data.outputs_recorded); repair fixes
 advances but question effects omit kind and invent undeclared returns_rule
 parameters. validate_effect correctly refuses. OptionsProposed sourcefallback,
 requestnull; engine recommends describe_distributions. This is NOT evidence
 the Navigator chose to skip preparation. Actual lengths are within limits;
 no presentation-length explanation. Prompt/schema must specify valid effect
 shapes, exact declared action parameters and local subgoal identifiers.
 Never accept undeclared scientific parameters to force completion.
 11requests,377.74s,$0.0192847968; cheaper because it failed earlier.

## Next consolidated work
Fix joint group/time enforcement or clearly refuse impossible constraints.
Clarify Navigator effect/goal contracts and representation metric-only versus
embedding outputs, with regression tests. Low effort remains fixed. Engine
and instruction changes get separate commits/ledger entries. Retail's exact
metadata question remains with the author for future batches; current04/05
scores remain unchanged. Keep original attempts, curations and cache restores.
No hidden-label tuning, held-out/biomedical run,paper edits,push,freeze or release.
