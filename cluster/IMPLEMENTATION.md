# Cluster RL implementation

Approved plan: 10 manipulation tasks × state/rgb/map, seed 0, official state
training budgets; uniform native physics; Mac orchestration and bounded agent
recovery; dedicated image; PVC checkpoints and portable downloads. Production
training is not launched during implementation.

- [x] Inspect the PVC using a dedicated CPU login pod: writable, exact ViT-L/16 weights found.
- [x] Add RGB observations without changing task dynamics.
- [x] Add checkpoint continuation and portable map loading.
- [x] Add cluster configuration, manifests, worker, controller and agent recovery.
- [x] Add image, documentation and regression/smoke tests.
- [x] Mac Python 3.9 operational tests, manifest client dry-run, actual agent JSON decision,
  actual PVC → Mac transfer/checksum, shell syntax and Ruff.
- [ ] Finish real A6000 environment, 30-combination learning and resume checks.

The Mac has no Docker; the image has not been built/pushed. The temporary GPU pod
uses the same pinned setup script as the Dockerfile. Production submission always
reruns the GPU suite inside the supplied image before unlocking the 30 training Jobs.
