#!/usr/bin/env python3
"""
apply_dann_first_experiment_patch.py
======================================
DANN first-experiment scoping (Handoff_calibration_session_to_dann_supcon.md
sec4d/4e). Applies three edits:

  1. src/models/akida_cnn_v2.py
       - GradientReversalLayer (training-only, discarded before deploy)
       - build_seizure_cnn_v2_dann()  -- shared trunk (same layer names as
         build_seizure_cnn_v2) + main head + GRL-gated domain head
       - extract_deployable_submodel() -- copies trunk+main-head weights
         by name into a plain build_seizure_cnn_v2 instance; this is the
         model that ever sees quantize()/convert(), never the DANN model
         itself.

  2. src/preprocessing/build_dataset_multi.py
       - domain_train / domain_val arrays (domain id = index into
         --patients, i.e. 0=chb01, 1=chb02, 2=chb05 for the default pool)
       - SMOTE now runs PER DOMAIN (not pooled) so every synthetic sample's
         domain label is exactly the domain it was generated inside --
         no inheritance heuristic required.
       - domain_map recorded in multi_scaler.json for provenance.

  3. src/models/train_baseline.py
       - --dann / --dann-lambda flags
       - full DANN training path: build -> compile with dict losses ->
         fit -> extract deployable submodel -> AKD1000 v1 compat check on
         the DEPLOYABLE model (not the branching one) -> save + manifest.
       - requires a dataset rebuilt via (2) above; hard-refuses otherwise.

Run from ~/dissertation/ with akida_env activated:
    python3 apply_dann_first_experiment_patch.py

Hard-refuses (exits nonzero, writes nothing to the file it fails on) if
any anchor text isn't found exactly once -- re-run after checking whether
the file changed since this patch was written against the 30 June 2026
snapshot.
"""
import sys

def patch_file(path, replacements):
    with open(path, 'r') as f:
        content = f.read()
    for i, (old, new) in enumerate(replacements):
        n = content.count(old)
        if n == 0:
            sys.exit(f"REFUSING: anchor #{i} not found in {path}.\n"
                      "File on disk doesn't match what this patch expects "
                      "-- no changes written to this file.")
        if n > 1:
            sys.exit(f"REFUSING: anchor #{i} matches {n} times in {path} "
                      "(expected exactly 1). No changes written.")
        content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched: {path}")


# ============================================================================
# 1. akida_cnn_v2.py
# ============================================================================
AKIDA_CNN_V2_PATH = 'src/models/akida_cnn_v2.py'

akida_old_1 = """def build_patient_adapted_model(base_model, freeze_until='relu3'):"""

akida_new_1 = '''class GradientReversalLayer(keras.layers.Layer):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015) for DANN.
    Forward: identity. Backward: gradient scaled by -lambda_.

    TRAINING-ONLY -- never part of the deployed graph. Discarded (along
    with the whole domain head) before quantize()/convert() via
    extract_deployable_submodel() below. AKD1000 v1 never sees this
    layer or the branching graph it creates; the deployed model stays
    architecturally identical to build_seizure_cnn_v2 (Handoff_
    calibration_session_to_dann_supcon.md sec4b; confirmed against
    Akd1000_v1_architecture_constraints.md, which forbids branching
    only in the CONVERTED graph, not the training-time TF graph).
    """
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self._lambda = float(lambda_)

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -self._lambda * dy
            return x, grad
        return _reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self._lambda})
        return config


def build_seizure_cnn_v2_dann(n_channels=18, window_samples=512,
                               n_domains=3, grl_lambda=1.0):
    """
    DANN scoping experiment (Handoff_calibration_session_to_dann_supcon.md
    sec4d/4e), first concrete run. Domains = pool patients, in the order
    passed to build_dataset_multi.py's --patients (domain id = list
    index). For the first screen: domains = chb01/02/05 -> n_domains=3.

    Shared trunk uses IDENTICAL layer names to build_seizure_cnn_v2, so
    extract_deployable_submodel() below can copy weights by name with no
    remapping. Domain head attaches to the flattened shared features via
    a GradientReversalLayer -- training-only, discarded before deployment.
    """
    inp = keras.Input(shape=(n_channels, window_samples, 1), name='eeg_input')
    x = keras.layers.Rescaling(scale=1.0 / 255.0, name='rescaling')(inp)

    x = keras.layers.Conv2D(32, (9, 7), strides=(1, 4),
        padding='valid', use_bias=False, name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.MaxPooling2D((1, 2), padding='valid', name='pool1')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu1')(x)

    x = keras.layers.Conv2D(64, (3, 3),
        padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu2')(x)

    x = keras.layers.Conv2D(32, (3, 3),
        padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool3')(x)
    x = keras.layers.ReLU(max_value=6.0, name='relu3')(x)

    shared = keras.layers.Flatten(name='flatten')(x)

    # -- Main task head (identical to build_seizure_cnn_v2) --------------
    h = keras.layers.Dense(64, use_bias=False, name='dense1')(shared)
    h = keras.layers.ReLU(max_value=6.0, name='relu_dense')(h)
    main_out = keras.layers.Dense(2, activation='softmax', name='output')(h)

    # -- Domain classifier head (training-only, via GRL) ------------------
    d = GradientReversalLayer(lambda_=grl_lambda, name='grl')(shared)
    d = keras.layers.Dense(32, use_bias=False, name='domain_dense1')(d)
    d = keras.layers.ReLU(max_value=6.0, name='domain_relu1')(d)
    domain_out = keras.layers.Dense(n_domains, activation='softmax',
                                     name='domain_output')(d)

    return keras.Model(inputs=inp, outputs=[main_out, domain_out],
                        name='seizure_cnn_v2_dann')


def extract_deployable_submodel(dann_model, n_channels=18, window_samples=512):
    """
    Strip a trained DANN model down to plain build_seizure_cnn_v2 --
    copies weights by matching layer name for the shared trunk + main
    head, drops the GRL and domain head entirely. This is the model
    that goes through quantize()/convert(); the DANN model itself
    never does (Handoff sec4b).
    """
    deployable = build_seizure_cnn_v2(n_channels=n_channels,
                                       window_samples=window_samples)
    copied, skipped = [], []
    for layer in deployable.layers:
        try:
            src_layer = dann_model.get_layer(layer.name)
        except ValueError:
            skipped.append(layer.name)
            continue
        layer.set_weights(src_layer.get_weights())
        copied.append(layer.name)
    print(f"[extract_deployable_submodel] Copied: {copied}")
    if skipped:
        print(f"[extract_deployable_submodel] WARNING -- no source layer "
              f"for: {skipped} (expected empty; investigate before trusting "
              "this checkpoint)")
    return deployable


def build_patient_adapted_model(base_model, freeze_until='relu3'):'''

# ============================================================================
# 2. build_dataset_multi.py
# ============================================================================
BUILD_DATASET_MULTI_PATH = 'src/preprocessing/build_dataset_multi.py'

bdm_old_1 = """train_X_parts, train_y_parts = [], []
val_X_parts,   val_y_parts   = [], []
scaler_info = {}

for pat in args.patients:"""

bdm_new_1 = """train_X_parts, train_y_parts = [], []
val_X_parts,   val_y_parts   = [], []
domain_train_parts, domain_val_parts = [], []   # DANN scoping
scaler_info = {}

for domain_id, pat in enumerate(args.patients):"""

bdm_old_2 = """    train_X_parts.append(X_sub_s)
    train_y_parts.append(y_sub)
    val_X_parts.append(X_vl_s)
    val_y_parts.append(y_vl)

    del X, y, X_sub, X_sub_s, X_vl_s   # release mmap + copies"""

bdm_new_2 = """    train_X_parts.append(X_sub_s)
    train_y_parts.append(y_sub)
    val_X_parts.append(X_vl_s)
    val_y_parts.append(y_vl)
    domain_train_parts.append(np.full(len(y_sub), domain_id, dtype=np.int32))
    domain_val_parts.append(np.full(len(y_vl), domain_id, dtype=np.int32))

    del X, y, X_sub, X_sub_s, X_vl_s   # release mmap + copies"""

bdm_old_3 = """X_pool       = np.concatenate(train_X_parts, axis=0)
y_pool       = np.concatenate(train_y_parts, axis=0)
X_val_chrono = np.concatenate(val_X_parts,   axis=0)   # real, chronological,
y_val_chrono = np.concatenate(val_y_parts,   axis=0)   # but 0 seizures — kept for reference only"""

bdm_new_3 = """X_pool       = np.concatenate(train_X_parts, axis=0)
y_pool       = np.concatenate(train_y_parts, axis=0)
X_val_chrono = np.concatenate(val_X_parts,   axis=0)   # real, chronological,
y_val_chrono = np.concatenate(val_y_parts,   axis=0)   # but 0 seizures — kept for reference only
domain_pool       = np.concatenate(domain_train_parts, axis=0)   # DANN scoping
domain_val_chrono = np.concatenate(domain_val_parts,   axis=0)"""

bdm_old_4 = """from sklearn.model_selection import train_test_split
X_tr_real, X_vl_real, y_tr_real, y_vl_real = train_test_split(
    X_pool, y_pool, test_size=0.20, stratify=y_pool, random_state=args.seed
)
print(f"Real (pre-SMOTE) split: train={len(X_tr_real)} (seizure={int(y_tr_real.sum())})  "
      f"val={len(X_vl_real)} (seizure={int(y_vl_real.sum())})")"""

bdm_new_4 = """from sklearn.model_selection import train_test_split
X_tr_real, X_vl_real, y_tr_real, y_vl_real, domain_tr_real, domain_vl_real = train_test_split(
    X_pool, y_pool, domain_pool, test_size=0.20, stratify=y_pool, random_state=args.seed
)
print(f"Real (pre-SMOTE) split: train={len(X_tr_real)} (seizure={int(y_tr_real.sum())})  "
      f"val={len(X_vl_real)} (seizure={int(y_vl_real.sum())})")"""

bdm_old_5 = """# ── SMOTE — applied ONLY to the real-train portion ─────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("pip install imbalanced-learn")

n_seiz_tr  = int(y_tr_real.sum())
X_flat     = X_tr_real.reshape(len(X_tr_real), -1)
sm         = SMOTE(sampling_strategy='minority',
                   k_neighbors=min(5, n_seiz_tr - 1),
                   random_state=args.seed)
X_sm, y_sm = sm.fit_resample(X_flat, y_tr_real)
X_train    = X_sm.reshape(-1, 18, 512).astype('float32')
y_train    = y_sm.astype('int32')

print(f"After SMOTE: {len(X_train)} windows "
      f"(seizure={int(y_train.sum())}, {100*y_train.mean():.1f}%)")

idx     = rng.permutation(len(X_train))
X_train = X_train[idx]
y_train = y_train[idx]"""

bdm_new_5 = """# ── SMOTE — applied ONLY to the real-train portion, PER DOMAIN ─────────────────
# DANN scoping: pooled SMOTE has no notion of domain -- a synthetic seizure
# window interpolated between two real chb01 neighbours would need a domain
# label assigned to it with no principled inheritance rule. Running SMOTE
# separately within each domain's real-train rows sidesteps this entirely:
# every synthetic sample's domain label is exactly the domain it was
# generated inside. See Handoff_calibration_session_to_dann_supcon.md sec4d.
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("pip install imbalanced-learn")

X_train_parts, y_train_parts, domain_train_final_parts = [], [], []
for dom_id in sorted(set(domain_tr_real.tolist())):
    mask = domain_tr_real == dom_id
    X_dom, y_dom = X_tr_real[mask], y_tr_real[mask]
    n_seiz_dom = int(y_dom.sum())
    if n_seiz_dom < 2:
        raise ValueError(
            f"Domain {dom_id} ({args.patients[dom_id]}) has only "
            f"{n_seiz_dom} seizure window(s) in its real-train split after "
            "the pooled 80/20 split -- cannot SMOTE within this domain. "
            "Per-domain SMOTE requires every pool patient to individually "
            "clear this bar."
        )
    X_flat_dom = X_dom.reshape(len(X_dom), -1)
    sm_dom = SMOTE(sampling_strategy='minority',
                   k_neighbors=min(5, n_seiz_dom - 1),
                   random_state=args.seed)
    X_sm_dom, y_sm_dom = sm_dom.fit_resample(X_flat_dom, y_dom)
    X_train_parts.append(X_sm_dom.reshape(-1, 18, 512).astype('float32'))
    y_train_parts.append(y_sm_dom.astype('int32'))
    domain_train_final_parts.append(np.full(len(y_sm_dom), dom_id, dtype=np.int32))
    print(f"  Domain {dom_id} ({args.patients[dom_id]}): {len(y_dom)} real "
          f"-> {len(y_sm_dom)} post-SMOTE (seizure={int(y_sm_dom.sum())})")

X_train      = np.concatenate(X_train_parts, axis=0)
y_train      = np.concatenate(y_train_parts, axis=0)
domain_train = np.concatenate(domain_train_final_parts, axis=0)

print(f"After per-domain SMOTE: {len(X_train)} windows total "
      f"(seizure={int(y_train.sum())}, {100*y_train.mean():.1f}%)")

idx          = rng.permutation(len(X_train))
X_train      = X_train[idx]
y_train      = y_train[idx]
domain_train = domain_train[idx]"""

bdm_old_6 = """X_val = X_vl_real.astype('float32')   # real, untouched by SMOTE
y_val = y_vl_real.astype('int32')"""

bdm_new_6 = """X_val      = X_vl_real.astype('float32')   # real, untouched by SMOTE
y_val      = y_vl_real.astype('int32')
domain_val = domain_vl_real.astype('int32')       # DANN scoping"""

bdm_old_7 = """np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val,     y_val=y_val,
                    X_val_chrono=X_val_chrono, y_val_chrono=y_val_chrono)"""

bdm_new_7 = """np.savez_compressed(out_path,
                    X_train=X_train, y_train=y_train, domain_train=domain_train,
                    X_val=X_val,     y_val=y_val,     domain_val=domain_val,
                    X_val_chrono=X_val_chrono, y_val_chrono=y_val_chrono,
                    domain_val_chrono=domain_val_chrono)"""

bdm_old_8 = """with open('data/processed/multi_scaler.json', 'w') as f:
    json.dump({'patients': args.patients, 'per_patient': scaler_info}, f, indent=2)
print("Saved  : data/processed/multi_scaler.json")"""

bdm_new_8 = """with open('data/processed/multi_scaler.json', 'w') as f:
    json.dump({'patients': args.patients,
               'domain_map': {p: i for i, p in enumerate(args.patients)},
               'per_patient': scaler_info}, f, indent=2)
print("Saved  : data/processed/multi_scaler.json")
print(f"Domain map: {dict((p, i) for i, p in enumerate(args.patients))}")"""

# ============================================================================
# 3. train_baseline.py
# ============================================================================
TRAIN_BASELINE_PATH = 'src/models/train_baseline.py'

tb_old_1 = """parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()"""

tb_new_1 = """parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--dann', action='store_true',
                    help='DANN scoping experiment (Handoff_calibration_'
                         'session_to_dann_supcon.md sec4d/4e). Requires '
                         '--multi-patient and a dataset built with domain '
                         'labels (domain_train/domain_val in the npz). '
                         'Mutually exclusive with --finetune-from, --stft, '
                         '--longctx.')
parser.add_argument('--dann-lambda', type=float, default=0.1,
                    help='Domain-loss weight / GRL lambda for --dann '
                         '(default 0.1; first screen also tries 0.5)')
args = parser.parse_args()

if args.dann and not args.multi_patient:
    parser.error('--dann requires --multi-patient (domains = pool patients)')
if args.dann and (args.finetune_from or args.stft or args.longctx):
    parser.error('--dann is mutually exclusive with --finetune-from/--stft/--longctx '
                 '(first screen only)')"""

tb_old_2 = """# ── Multi-patient val fix ──────────────────────────────────────────────────────"""

tb_new_2 = """# ── DANN domain labels ──────────────────────────────────────────────────────────
if args.dann:
    if 'domain_train' not in data.files or 'domain_val' not in data.files:
        sys.exit(
            f"ERROR: {data_path} has no domain_train/domain_val arrays -- "
            "it predates the DANN-scoping patch to build_dataset_multi.py.\\n"
            "Rebuild via:\\n"
            "  python3 src/preprocessing/build_dataset_multi.py "
            "--patients chb01 chb02 chb05"
        )
    domain_train = data['domain_train']
    domain_val   = data['domain_val']
    n_domains    = int(max(domain_train.max(), domain_val.max()) + 1)
    print(f"\\n[DANN] domain_train: {len(domain_train)}  "
          f"domain_val: {len(domain_val)}  n_domains={n_domains}")

# ── Multi-patient val fix ──────────────────────────────────────────────────────"""

tb_old_3 = """os.makedirs('results', exist_ok=True)

if args.finetune_from:"""

tb_new_3 = """os.makedirs('results', exist_ok=True)

if args.dann:
    # ── DANN training path (first concrete experiment, Handoff sec4d/4e) ──
    from src.models.akida_cnn_v2 import (build_seizure_cnn_v2_dann,
                                          extract_deployable_submodel,
                                          GradientReversalLayer)
    print(f"\\n=== DANN SCOPING EXPERIMENT (lambda={args.dann_lambda}) ===")
    print(f"Domains: {n_domains} (pool patients, in --patients order)")

    dann_model = build_seizure_cnn_v2_dann(
        n_channels=18, window_samples=512,
        n_domains=n_domains, grl_lambda=args.dann_lambda)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(dann_model)
        print("[DANN] compat check ran on the branching model -- not "
              "meaningful (it will never be converted). The DEPLOYABLE "
              "submodel gets the real check after extraction, below.")
    except Exception as e:
        print(f"[DANN] compat check on branching model raised: {e} "
              "(expected/ignorable -- see extract_deployable_submodel step)")

    dann_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={'output': 'sparse_categorical_crossentropy',
              'domain_output': 'sparse_categorical_crossentropy'},
        loss_weights={'output': 1.0, 'domain_output': 1.0},
        metrics={'output': 'accuracy', 'domain_output': 'accuracy'},
    )
    dann_model.summary(print_fn=lambda x: print(f"  {x}"))

    dann_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_dann_lambda{args.dann_lambda}_TRAINING.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            dann_ckpt, monitor='val_output_loss',
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_output_loss', patience=15,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_output_loss', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1),
    ]
    dann_model.fit(
        X_train, {'output': y_train, 'domain_output': domain_train},
        validation_data=(X_val, {'output': y_val, 'domain_output': domain_val}),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, verbose=2,
    )

    best_dann = keras.models.load_model(
        dann_ckpt, custom_objects={'GradientReversalLayer': GradientReversalLayer})

    print("\\n[DANN] Extracting deployable submodel (trunk + main head only)...")
    deployable = extract_deployable_submodel(best_dann, n_channels=18, window_samples=512)

    try:
        with set_akida_version(AkidaVersion.v1):
            check_model_compatibility(deployable)
        print("AKD1000 v1 compatible (deployable submodel) ✓")
    except Exception as e:
        print(f"WARNING: compatibility check on deployable submodel raised: {e}")

    # Naming matches convert_to_snn.py's --base/--variant convention exactly:
    #   results/best_ann_<base>_v<V>_<variant>.h5
    # so conversion is a plain: --base multi --variant dann_lambda<L>
    deploy_ckpt = f'results/best_ann_{patient_tag}_v{args.model_version}_dann_lambda{args.dann_lambda}.h5'
    deployable.save(deploy_ckpt)
    print(f"Saved deployable checkpoint: {deploy_ckpt}")

    _evaluate_and_save(
        deployable, X_train, y_train, X_val, y_val,
        X_test if has_test else None, y_test if has_test else None,
        patient_tag=f'{patient_tag}_dann_lambda{args.dann_lambda}',
        model_version=args.model_version,
    )
    # patient_tag passed UNMODIFIED here (not the dann-suffixed variant) so
    # _write_ckpt_manifest resolves the correct multi_scaler.json path.
    _write_ckpt_manifest(
        deploy_ckpt, patient_tag=patient_tag,
        seed=args.seed, finetune_from=None, gradual_unfreeze=False,
        model_version=args.model_version,
    )
    print(f"\\n[DANN] Done. Next -- convert + eval on each held-out patient:")
    print(f"  python3 src/models/convert_to_snn.py --patient multi --base multi "
          f"--variant dann_lambda{args.dann_lambda} --eval-patient chb03 "
          f"--model-version {args.model_version} --cal-samples 1024")
    print(f"  # repeat --eval-patient for chb10 chb13 chb15 chb16 chb20")
    print(f"  python3 src/evaluation/eval_event_level.py --fbz "
          f"results/seizure_model_multi_dann_lambda{args.dann_lambda}"
          f"_v{args.model_version}_w4a4.fbz --eval-patient chb03  # repeat per patient")
    sys.exit(0)

if args.finetune_from:"""


if __name__ == '__main__':
    patch_file(AKIDA_CNN_V2_PATH, [(akida_old_1, akida_new_1)])
    patch_file(BUILD_DATASET_MULTI_PATH, [
        (bdm_old_1, bdm_new_1),
        (bdm_old_2, bdm_new_2),
        (bdm_old_3, bdm_new_3),
        (bdm_old_4, bdm_new_4),
        (bdm_old_5, bdm_new_5),
        (bdm_old_6, bdm_new_6),
        (bdm_old_7, bdm_new_7),
        (bdm_old_8, bdm_new_8),
    ])
    patch_file(TRAIN_BASELINE_PATH, [
        (tb_old_1, tb_new_1),
        (tb_old_2, tb_new_2),
        (tb_old_3, tb_new_3),
    ])
    print("\nAll patches applied successfully.")
    print("\nSanity checks before training:")
    print("  python3 src/models/akida_cnn_v2.py   # must still print AKD1000 v1 compatible ✓")
    print("  python3 -c \"from src.models.akida_cnn_v2 import build_seizure_cnn_v2_dann; "
          "m = build_seizure_cnn_v2_dann(); m.summary()\"")
