# FANTrack Code Review

## Branch Context

| Item | Value |
|------|-------|
| **Reviewed branch** | `copy` (current branch, also the remote default HEAD) |
| **Compared against** | `origin/main` |
| **Review type** | Static analysis of `copy` source files + diff analysis of `copy` vs `main` |
| **Review date** | 2026-03-08 |

> **Important**: This review covers two layers:
> 1. **Static findings** — bugs and issues found in the `copy` branch source files regardless of branch history.
> 2. **Branch-diff findings** — issues specific to `copy` that are already fixed in `main`, or vice versa. These are marked with `[COPY vs MAIN]`.

Files changed between `copy` and `main` (relevant Python files):
- `fan_track/config/config.py`
- `fan_track/data_generation/kitti_assocnet_dataset.py`
- `fan_track/data_generation/kitti_simnet_dataset.py`
- `fan_track/data_generation/simnet_batch_sampler.py`
- `fan_track/experiments/run_tracker.py`
- `fan_track/network/tracker.py`
- `fan_track/network/train.py`

---

## Code Review Summary

**Files statically reviewed**: 7 core files (~2,100 lines)
— `network/model.py`, `network/tracker.py`, `network/train.py`, `network/layers.py`,
`config/config.py`, `utils/generic_utils.py`, `experiments/train_simnet.py`

**Overall assessment**: `REQUEST_CHANGES`

---

## Findings

### P0 - Critical

**1. `train.py:397–398` — Tensors used as feed_dict keys instead of placeholders**

```python
self.model.avg_loss: avg_loss,       # avg_loss is a computed TF op, not a placeholder
self.model.avg_accuracy: avg_accuracy,
```

`validate_network()` passes the computed tensors `model.avg_loss` / `model.avg_accuracy` as
feed_dict keys instead of the correct placeholders `model.in_avg_loss_ph` /
`model.in_avg_accuracy_ph`. TensorFlow will raise a runtime error on every validation call.
The correct usage is visible in `train_assocnet()` at lines 321–322.

---

### P1 - High

**2. `simnet_batch_sampler.py:87` — `if True:` always regenerates dataset** `[COPY vs MAIN]`

```python
# copy branch:
if True:
    dataset_obj = KittiSimnetDataset()  # regenerates every run

# main branch (fixed):
if not (os.path.exists(self.dataset_path)):
    dataset_obj = KittiSimnetDataset()
```

The `copy` branch unconditionally regenerates the simnet dataset on every run, wasting
significant compute. The `main` branch already fixes this with a path-existence check.

**3. `tracker.py:359–363` — Hardcoded Colab path for AVOD** `[COPY vs MAIN]`

```python
# copy branch:
avod_root_dir = '/content/fantrack/fan_track/object_detector/'

# main branch (fixed):
os.path.join(generic_utils.get_project_root(), 'object_detector/avod/data/outputs', ...)
```

`tracker.py` in `copy` hardcodes `/content/fantrack/...` — a Google Colab absolute path.
This is already fixed in `main` to use `get_project_root()`.

**4. `kitti_assocnet_dataset.py:52` — Same hardcoded Colab path** `[COPY vs MAIN]`

```python
avod_root_dir = '/content/fantrack/fan_track/object_detector/'  # copy branch only
```

Same issue as above. The `main` branch uses the `avod_root_dir()` helper function instead.

**5. `tracker.py:1063–1079` — Debug save block with hardcoded path still present** `[COPY vs MAIN]`

```python
save_path = '/content/demo/data/%s/' % video_no  # copy branch only
```

A debug block that saves inference data to a hardcoded Colab path is still present in `copy`.
This silently creates directories and writes files during every inference run. Already removed
in `main`.

**6. `config.py:91–92` — Different AVOD checkpoint name vs `main`** `[COPY vs MAIN]`

```python
# copy branch:
AVOD_CKPT_NAME = 'avod_cars_fast'
AVOD_CKPT_NUMBER = '00093000'

# main branch:
AVOD_CKPT_NAME = 'pyramid_cars_with_aug_example'
AVOD_CKPT_NUMBER = '00221000'
```

The CLAUDE.md documentation states the cars checkpoint is `pyramid_cars_with_aug_example`
at step `00221000`. The `copy` branch uses a different checkpoint name and step, which may
produce different (lower quality) tracking results.

**7. `generic_utils.py:450` — `training_type` CLI argument always silently overridden**

```python
args = parser.parse_args()
args.training_type = TrainingType.SimnetRun  # unconditional override
```

The `--training_type` argument is parsed but then immediately replaced. Any user who passes
`--training_type 0` (Both) or `--training_type 2` (Assocnet) gets `SimnetRun` regardless.
The flag is effectively broken.

**8. `generic_utils.py:422` — `epochs` declared as `float`, used in `range()`**

```python
parser.add_argument('--epochs', type=float, default=100, ...)
...
for epoch in range(self.model.args.epochs):  # train.py:291
```

`range()` raises `TypeError: 'float' object cannot be interpreted as an integer`.
Should be `type=int`.

**9. `train.py:112` — `Saver(max_to_keep=None)` stores unlimited checkpoints**

```python
saver = tf.train.Saver(max_to_keep=None)
```

Used in both `train_simnet` and `train_assocnet`. Over 500 epochs this silently fills the disk.
`SimnetConfig.MAX_CHECKPOINTS = 0` and `--max_to_keep` arg exist but are not wired here.

**10. `model.py:308` — Wrong tensor used in histogram summary**

```python
tf.summary.histogram('conv2_1', self.conv2_0_act)  # should be conv2_1_act
```

`conv2_1` histogram logs `conv2_0_act` again — the actual `conv2_1_act` activations are
never visible in TensorBoard.

**11. `layers.py:67` — Wrong dimension in column index computation**

```python
col_idx = tf.floormod(indices, height)  # should be `width`
```

Column index in a flattened 2D array is `index % width`, not `index % height`.
Currently harmless because all maps are square, but semantically wrong.

**12. `generic_utils.py:73` — `os` used without explicit import**

`trim_txt_files` and `prepare_args` use `os.walk` / `os.path.join`, but `generic_utils.py`
has no `import os`. It works only because `from fan_track.config.config import *` incidentally
pulls in `os` as a side effect — this is fragile and order-dependent.

---

### P2 - Medium

**13. `kitti_simnet_dataset.py` — KITTI dataset workaround removed in `main`** `[COPY vs MAIN]`

```python
# copy branch retains this guard:
if video_no == '0001' and frame_id in range(177, 181):
    print('There is missing data in KITTI tracking dataset at seq 1, frame 177-180!')
    continue
```

The `copy` branch retains a workaround for a known missing-data issue in KITTI sequence 0001
frames 177–180. The `main` branch removes it silently without explanation. If the underlying
data issue is not resolved, removing this guard will cause silent crashes or corrupt training
examples on that sequence.

**14. `config.py:14` — Hardcoded Colab path as default**

```python
KITTI_ROOT = '/content/Kitti'
```

Only works on Google Colab. Local users must edit source. Should use an environment variable:
`os.environ.get('KITTI_ROOT', '/content/Kitti')`.

**15. `model.py:797–803` — Dead inner function `elemwise_cross_entropy`**

```python
def elemwise_cross_entropy(x):
    '''x[0]: element from the label map...'''
    # no body, no call
```

Defined inside `get_cross_entropy_loss()` but never implemented or called. Remove it.

**16. `model.py:2` — Unused import**

```python
from tensorflow.python import debug as tf_debug
```

The `tf_debug` wrapper is commented out (line 67). The import serves no purpose.

**17. `model.py:14–17` — Hardcoded thread count at class level**

```python
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
```

Hardcoded to 44 at class scope — not configurable without modifying source.

**18. `generic_utils.py:112–113` — Redundant isinstance check**

```python
elif isinstance(input_data, tf.Tensor):
    if isinstance(input_data, tf.Tensor):  # always True here
```

Inner check is always true given the outer `elif`. Remove the inner `if`.

**19. Wildcard imports across multiple files**

```python
from fan_track.experiments.ablation import *   # tracker.py:5
from fan_track.config.config import *           # generic_utils.py, train.py, model.py
```

Pollutes namespaces, hides what symbols are actually needed, and creates the fragile `os`
dependency in `generic_utils.py` (finding #12).

---

### P3 - Low

**20. `model.py:347–528` — 180+ lines of commented-out code**

The entire `create_sim_map` method is commented out. Should be deleted (git history
preserves it) or restored if needed.

**21. `train.py:59` — Typo in attribute name**

`self.asoocNet_train_op` (double-o). Consistent across 5+ usages, but affects readability.

**22. `train.py:61` — Typo in TensorBoard tag**

`tf.summary.scalar('averege_loss', ...)` → `'average_loss'`

**23. `tracker.py` uses tabs; rest of codebase uses spaces**

Mixing tabs and spaces will cause `IndentationError` in Python 3 if files are ever mixed
or edited carelessly. PEP 8 requires spaces only.

**24. `kitti_simnet_dataset.py` — Debug `print` statements left in** `[COPY vs MAIN]`

```python
print(self.dataset_config.dataset_dir)
print(self.dataset_config)
```

These were added in `copy` and are not present in `main`. They expose internal config
objects to stdout on every dataset construction.

---

## Removal / Iteration Plan

| Item | File | Action | Risk |
|------|------|--------|------|
| Commented-out `create_sim_map` (180 lines) | `model.py:347–528` | Safe delete now | None — already dead |
| `tf_debug` import | `model.py:2` | Safe delete now | None |
| Inner `elemwise_cross_entropy` function | `model.py:797` | Safe delete now | None |
| `train_combined_network()` method | `train.py:552` | Delete or restore | Medium — references removed ops |
| Debug save block | `tracker.py:1063` | Safe delete (already done in `main`) | None |
| Debug `print` statements | `kitti_simnet_dataset.py:61,63` | Safe delete | None |

---

## Branch Recommendation

The `copy` branch has several regressions compared to `main`:

| Issue | copy | main |
|-------|------|------|
| AVOD path | hardcoded Colab path | uses `get_project_root()` |
| Dataset regeneration | always (`if True:`) | conditional on path existence |
| Debug save block | present (writes to `/content/demo/`) | removed |
| AVOD checkpoint name | `avod_cars_fast` / `00093000` | `pyramid_cars_with_aug_example` / `00221000` |

Consider rebasing `copy` onto `main` or cherry-picking the relevant fixes from `main`
before merging `copy` anywhere.
