# Cluster RL implementation (validation at 4e41f28)

Approved plan: 10 manipulation tasks × state/rgb/map, seed 0, official state
training budgets; uniform native physics; Mac orchestration and bounded agent
recovery; dedicated image; PVC checkpoints and portable downloads. Production
training is not launched during implementation.

Latest user updates: preserve main's task-specific goal visibility and DINO PCA
viewer, enable native RGB textures, and merge the verified implementation into main.

- [x] Inspect the PVC using a dedicated CPU login pod: writable, exact ViT-L/16 weights found.
- [x] Add RGB observations without changing task dynamics.
- [x] Add checkpoint continuation and portable map loading.
- [x] Add cluster configuration, manifests, worker, controller and agent recovery.
- [x] Add image, documentation and regression/smoke tests.
- [x] Mac Python 3.9 operational tests, manifest client dry-run, actual agent JSON decision,
  actual PVC → Mac transfer/checksum, shell syntax and Ruff.
- [x] Pass all 30 baseline task/policy PPO smoke tests on A6000.
- [x] Verify actual textured observations and all 10 RGB PPO combinations after integrating main.
- [x] Finish affected map PPO and checkpoint continuation checks on current main: 51/51 GPU cases.
- [x] Integrate latest main into the verified branch before merging back.

Validation: 30 Mac operational tests; 41 core/checkpoint/PCA tests and 11 subtests
(one optional viewer test skipped); all 30 baseline PPO combinations; 51 focused
GPU cases after main integration. Actual eval-only recovery and PVC → Mac download
verified 42 files by SHA-256. GPU PPO tests use small environment counts and short
training budgets; sustained 128-environment RGB memory/throughput was not measured.

The Mac has no Docker; the image has not been built/pushed. The temporary GPU pod
uses the same pinned setup script as the Dockerfile. Production submission always
reruns the GPU suite inside the supplied image before unlocking the 30 training Jobs.

## Contact geometry update

Map observations now use robot-frame normals, episode-fixed isotropic bounds and
7D contact encoding; cache version 3 and `contact_robot_v1` metadata reject old map runs.
Normalized epsilon is a numerical guard only, not an MLP feature. Before the user
waived further validation, 30 Mac operational tests and 16 lightweight tests plus
11 subtests passed. The new geometry/GPU suites were not run; the pending dedicated
validation pod was deleted. Earlier GPU results above apply to 4e41f28, not this update.
