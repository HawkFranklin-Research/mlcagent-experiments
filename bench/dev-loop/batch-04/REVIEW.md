# Batch 04 review

Fixed code afe46c3, controller1344fc8, rails-0.6; DeepSeek V4.1 Flash, temp0,
max16384 and provider-default reasoning unchanged.467 passed/1 skipped before
launch; experiments18 passed. No active-run source change.

## Outcomes
Four of five workflows reached the report; two qualified: Anuran Calls and
Seeds. Series9/12 qualified at least once, not K=3 acceptance. Bike and Retail
completed but failed exact setup agreement; Bike also failed time ordering.
Optical Digits stopped in EDA with output_truncated, before PCA evaluation.
184 requests;433509 input+769945 output tokens;716958 reported reasoning,
not added again. Provider-reported cost$0.74417513; no unknown core usage.
Cumulative998 requests/6397164 total tokens/$4.1218700654.

## Evidence for the fixes
- Seeds: both model steps exit0, then validation/test/report pass. The earlier
  KMeans/threadpoolctl sandbox block is resolved in this live run.
- Anuran: categorical validation/test metrics complete after repairs. Prior
  output truncations remain retained, not erased by eventual completion.
- Bike: validation records poisson_deviance, fixing the action/default mismatch.
  It reaches report without an upstream validation-recovery loop. The bounded
  scoring-only fallback is covered by offline tests, not needed in this run.
- Digits: failure happened before the PCA scoring fix could be exercised.
- Retail: selected build_entities before split and completed its customer-level
  clustering workflow. This progresses beyond earlier raw-group splitting.

## Remaining causes, kept separate
1. Bike declares time columns[dteday,hr] and additionally groups by day. Engine
   freeze gives grouping precedence and silently ignores declared chronology.
   SplitFrozen records group splitting; external time_ordered=false confirms
   the defect. Both constraints must hold, or the engine must refuse clearly.
   Extra grouping alone is not the leakage cause; random group partitioning is.
2. Exact metadata scoring: Bike's extra day group and Retail's InvoiceDate differ
   from researcher defaults. Retail passes all applicable leakage checks under
   the prospectively approved static task, but time field equality is false.
   Author has been asked about compatible additional constraints for later
   batches. No previous or current scores changed. Bike must fail until actual
   chronology is correct regardless of that decision.
3. Digits generation truncation persists. Output budgets include reasoning under
   the API contract. Preserve a provider inconsistency: one Anuran request
   reports16384 completion versus16595 reasoning tokens. Raw fields retained;
   no guessed correction or addition to completion. Aggregate provider numbers
   are reported, not independently verified token counts.

## External hidden-label measurement, not agent work
Seeds assignments:126 training/42 test rows. TestARI0.3387861089,
NMI0.4850069866; source/output hashes in the attempt's
external-evaluation-score.json. Hidden labels never entered the project.
These scores are not used to tune methods or rails.

## Next, fixed contrast
Batch05 changes ONLY requested reasoning effort to low. Supported model
metadata lists max/high/low/none, not medium. Code, rails, model, limits and
scoring criteria stay unchanged; failed tables only: Bike,Digits,Retail.
Pending engine/scorer corrections wait until after that isolated contrast.
No held-out/biomedical run, paper edit, reference analysis, push or freeze.
Original projects retained; closed derived caches byte-verified archived with
restoration metadata. Curated hashes exclude private data/models/tables.
