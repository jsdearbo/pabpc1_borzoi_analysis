import pickle
import numpy as np

with open("/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/outputs/fig3/input_seqs.pkl", "rb") as f:
    input_seqs = pickle.load(f)
with open("/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/outputs/fig3/input_seqs_rna_oriented.pkl", "rb") as f:
    oriented_input_seqs = pickle.load(f)

RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")

for i in range(10):
    orig = input_seqs[i]
    rc   = oriented_input_seqs[i]

    if isinstance(orig, str):
        is_same = (rc == orig)
        is_rc = (rc == orig.translate(RC_TABLE)[::-1])
        assert is_same or is_rc, f"Mismatch at index {i:d}"
    else:
        is_same = np.allclose(rc, orig)
        is_rc = np.allclose(rc, orig[[3,2,1,0], ::-1])
        assert is_same or is_rc, f"Mismatch at index {i:d}"

print("All checks passed!")

# verify that + strand sequences are unchanged
for i in range(10):
    orig = input_seqs[i]
    rc   = oriented_input_seqs[i]

    if isinstance(orig, str):
        is_same = (rc == orig)
        is_rc = (rc == orig.translate(RC_TABLE)[::-1])
        assert is_same or is_rc, f"Mismatch at index {i:d}"
    else:
        is_same = np.allclose(rc, orig)
        is_rc = np.allclose(rc, orig[[3,2,1,0], ::-1])
        assert is_same or is_rc, f"Mismatch at index {i:d}"

print("All checks passed!")

# check a minus stranded example for magnitude preservation of attributions

with open("/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/outputs/fig3/attributions.pkl", "rb") as f:
    attributions = pickle.load(f)
with open("/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/outputs/fig3/attributions_rna_oriented.pkl", "rb") as f:
    oriented_attributions = pickle.load(f)

for i in range(10):
    orig = attributions[i]
    rc   = oriented_attributions[i]

    print(np.abs(orig).sum(), np.abs(rc).sum())