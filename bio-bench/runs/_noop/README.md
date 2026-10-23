# Configuration no-ops (not studies)

`eval-02-no-manifest-2026-10-23/`: the first eval-02 launch omitted `--manifest bio-bench/manifest.yaml`. The controller read the engine's benchmark manifest, found no evaluation tasks and finished with 0 jobs and 0 model requests. No agent ran and no biomedical table was opened. It was moved here so that eval-02 runs in its own folder with the preregistered command; it is not a retry of any study.
