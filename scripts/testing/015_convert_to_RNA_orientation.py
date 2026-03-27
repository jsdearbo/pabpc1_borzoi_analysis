import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RC_INDEX = np.array([3, 2, 1, 0])  # A,C,G,T -> T,G,C,A
RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement_array(arr: np.ndarray) -> np.ndarray:
    """
    Reverse-complement an array shaped (4, L) or (N, 4, L).

    Assumes channel order A,C,G,T.
    """
    if arr.ndim == 2:
        if arr.shape[0] != 4:
            raise ValueError(f"Expected shape (4, L), got {arr.shape}")
        return arr[RC_INDEX, ::-1]

    if arr.ndim == 3:
        if arr.shape[1] != 4:
            raise ValueError(f"Expected shape (N, 4, L), got {arr.shape}")
        return arr[:, RC_INDEX, ::-1]

    raise ValueError(f"Unsupported ndim={arr.ndim} for shape {arr.shape}")


def reorient_to_rna(final_attributions, input_seqs, mapping_df, coord_data, strand_col="strand"):
    """
    Reorient saved attributions and sequences into RNA/transcription orientation.

    Parameters
    ----------
    final_attributions : np.ndarray
        Shape (N, 4, L)
    input_seqs : list[np.ndarray]
        Each entry shape (4, L)
    mapping_df : pd.DataFrame
        Must contain columns: coord_index, attribution_index
    coord_data : pd.DataFrame
        Must contain strand column
    strand_col : str
        Column in coord_data containing '+' / '-'

    Returns
    -------
    oriented_attributions : np.ndarray
    oriented_input_seqs : list[np.ndarray]
    """
    if strand_col not in coord_data.columns:
        raise ValueError(f"'{strand_col}' not found in coord_data columns")

    oriented_attributions = final_attributions.copy()
    oriented_input_seqs = list(input_seqs)

    for _, row in mapping_df.iterrows():
        coord_idx = int(row["coord_index"])
        attr_idx = int(row["attribution_index"])
        strand = str(coord_data.loc[coord_idx, strand_col]).strip()

        if strand == "-":
            oriented_attributions[attr_idx] = reverse_complement_array(
                oriented_attributions[attr_idx]
            )
            seq = oriented_input_seqs[attr_idx]
            if isinstance(seq, str):
                oriented_input_seqs[attr_idx] = seq.translate(RC_TABLE)[::-1]
            else:
                oriented_input_seqs[attr_idx] = reverse_complement_array(seq)
        elif strand == "+":
            continue
        else:
            raise ValueError(f"Unexpected strand value at coord index {coord_idx}: {strand!r}")

    return oriented_attributions, oriented_input_seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--coord_file_path", required=True)
    p.add_argument("--strand_col", default="strand")
    p.add_argument("--suffix", default="_rna_oriented")
    args = p.parse_args()

    attr_path = os.path.join(args.experiment_dir, "attributions.pkl")
    seq_path = os.path.join(args.experiment_dir, "input_seqs.pkl")
    map_path = os.path.join(args.experiment_dir, "attribution_mapping.csv")

    with open(attr_path, "rb") as f:
        final_attributions = pickle.load(f)
    with open(seq_path, "rb") as f:
        input_seqs = pickle.load(f)

    mapping_df = pd.read_csv(map_path)
    coord_data = pd.read_csv(args.coord_file_path)

    logger.info(f"Loaded attributions: {np.shape(final_attributions)}")
    logger.info(f"Loaded input seqs: {len(input_seqs)}")
    logger.info(f"Loaded mapping rows: {len(mapping_df)}")
    logger.info(f"Loaded coord rows: {len(coord_data)}")

    oriented_attributions, oriented_input_seqs = reorient_to_rna(
        final_attributions=final_attributions,
        input_seqs=input_seqs,
        mapping_df=mapping_df,
        coord_data=coord_data,
        strand_col=args.strand_col,
    )

    out_attr = os.path.join(args.experiment_dir, f"attributions{args.suffix}.pkl")
    out_seq = os.path.join(args.experiment_dir, f"input_seqs{args.suffix}.pkl")

    with open(out_attr, "wb") as f:
        pickle.dump(oriented_attributions, f)
    with open(out_seq, "wb") as f:
        pickle.dump(oriented_input_seqs, f)

    logger.info(f"Saved RNA-oriented attributions -> {out_attr}")
    logger.info(f"Saved RNA-oriented sequences -> {out_seq}")


if __name__ == "__main__":
    main()