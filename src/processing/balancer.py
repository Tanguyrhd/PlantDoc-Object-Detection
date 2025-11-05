"""
Data Balancer
Balances dataset classes through duplication.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataBalancer:
    """Balances dataset by duplicating samples from underrepresented classes."""

    def __init__(self):
        """
        Initialize balancer.
        """
        pass

    def balance_by_column(
        self,
        df: pd.DataFrame,
        column: str,
        target_samples_per_class: int,
        keep_above_target: bool = True
    ) -> pd.DataFrame:
        """
        Balance dataset by duplicating samples to reach target per class.

        Args:
            df: DataFrame to balance
            column: Column name to group by (e.g., 'species', 'disease')
            target_samples_per_class: Target number of samples per class
            keep_above_target: If True, keep all samples for classes above target.
                             If False, downsample to target.

        Returns:
            Balanced DataFrame
        """
        balanced_dfs = []

        for class_value, group in df.groupby(column):
            n_samples = len(group)
            n_to_add = target_samples_per_class - n_samples

            if n_to_add > 0:
                print(f"  {class_value}: {n_samples} → {target_samples_per_class} "
                      f"(adding {n_to_add} duplicates)")

                # Keep original samples
                balanced_dfs.append(group)

                # Add duplicates with modified filenames
                duplicates_added = 0
                while duplicates_added < n_to_add:
                    # Cycle through samples
                    idx = duplicates_added % n_samples
                    sample = group.iloc[idx:idx+1].copy()

                    # Modify filename to avoid conflicts
                    original_filename = sample['filename'].values[0]
                    stem = Path(original_filename).stem
                    suffix = Path(original_filename).suffix
                    new_filename = f"{stem}_dup{duplicates_added}{suffix}"
                    sample.loc[:, 'filename'] = new_filename

                    balanced_dfs.append(sample)
                    duplicates_added += 1

            else:
                if keep_above_target:
                    print(f"  {class_value}: {n_samples} (already >= target, keeping all)")
                    balanced_dfs.append(group)
                else:
                    print(f"  {class_value}: {n_samples} → {target_samples_per_class} "
                          f"(downsampling)")
                    balanced_dfs.append(group.iloc[:target_samples_per_class])

        df_balanced = pd.concat(balanced_dfs, ignore_index=True)

        print(f"\n✓ Dataset balanced! Total samples: {len(df_balanced)}")

        return df_balanced

    @staticmethod
    def get_class_distribution(df: pd.DataFrame, column: str) -> pd.Series:
        """
        Get class distribution counts.

        Args:
            df: DataFrame
            column: Column to count

        Returns:
            Series with class counts
        """
        return df[column].value_counts().sort_index()

    @staticmethod
    def print_distribution(df: pd.DataFrame, column: str, title: str = "Distribution"):
        """
        Print class distribution.

        Args:
            df: DataFrame
            column: Column to analyze
            title: Title for the output
        """
        print(f"\n{title}:")

        counts = DataBalancer.get_class_distribution(df, column)
        for label, count in counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label:12}: {count:5} samples ({percentage:5.1f}%)")

        print(f"\n  Total training: {len(df)} samples")
