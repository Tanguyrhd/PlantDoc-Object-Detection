"""
Species Pipeline
Pipeline for plant species classification.
"""

import pandas as pd
from .base_pipeline import BasePipeline


class SpeciesPipeline(BasePipeline):
    """Pipeline for species classification (includes both healthy and diseased plants)."""

    def get_pipeline_type(self) -> str:
        """Get pipeline type identifier."""
        return 'species'

    def get_class_column(self) -> str:
        """Get the column name used for classification."""
        return 'species'

    def filter_data(self):
        """
        Filter to keep only samples with valid species.
        """
        print(f"\n{'='*60}")
        print(f"FILTERING BY SPECIES")
        print(f"{'='*60}\n")

        # Keep only samples with valid species
        df_train_filtered = self.df_train[self.df_train['species'].notna()].copy()
        df_test_filtered = self.df_test[self.df_test['species'].notna()].copy()

        removed_train = len(self.df_train) - len(df_train_filtered)
        removed_test = len(self.df_test) - len(df_test_filtered)

        print(f"Training: {len(df_train_filtered)} samples (removed {removed_train})")
        print(f"Validation: {len(df_test_filtered)} samples (removed {removed_test})")

        # Update dataframes
        self.df_train = df_train_filtered
        self.df_test = df_test_filtered

        # Show species distribution
        self.balancer.print_distribution(
            self.df_train,
            'species',
            "Training species distribution (before balancing)"
        )

        print(f"\n✓ Filtering complete")

    def balance_data(self, interactive: bool = True) -> None:
        """
        Balance the dataset by letting user choose to balance with a specific target or keep the natural balanced

        Args:
            interactive: If True, ask user for target samples. If False, use default.
        """
        print(f"\n{'='*60}")
        print("PREPARING DATASETS")
        print(f"{'='*60}\n")

        # Ask user for balancing choice

        distribution = self.df_train['species'].value_counts().sort_index()
        apply_balancing = False

        if interactive:
            print(f"\n{'-'*60}")
            print("BALANCING OPTIONS")
            print(f"{'-'*60}")
            print("Do you want to balance the training dataset?")
            print("  1. Yes, with custom target")
            print("  2. No, keep natural distribution")

            while True:
                choice = input("\nMake a choice between 1 and 2: ").strip()

                if choice == "1":
                    apply_balancing = True
                    while True:
                        try:
                            target_samples = int(input("Enter target samples per class: "))
                            max_possible = distribution.min() * 2
                            if target_samples > max_possible:
                                print(f"⚠️  Warning: Maximum possible is {max_possible} (minority class size)")
                                print(f"   Using undersampling will limit to {max_possible} per class")
                                confirm = input(f"Continue with {target_samples}? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    break
                            elif target_samples > 0:
                                break
                            else:
                                print("⚠️  Please enter a positive number")
                        except ValueError:
                            print("⚠️  Please enter a valid number")
                    break
                elif choice == '2':
                    apply_balancing = False
                    break
                else:
                    print("⚠️  Please enter a valid choice between 1 and 2")

        # Apply balancing if requested
        if apply_balancing:
            print(f"\n Balancing training dataset to {target_samples} samples per class...")
            self.df_train_processed = self.balancer.balance_by_column(
                self.df_train,
                column='species',
                target_samples_per_class=target_samples
            )

            self.balancer.print_distribution(
            self.df_train_processed,
            'species',
            "Training species distribution (after balancing)"
            )
            print("✓ Training dataset balanced successfully")
        else:
            print("\n✓ Keeping natural distribution (no balancing)")
            self.df_train_processed = self.df_train.copy()

        # Test set is never balanced
        self.df_test_processed = self.df_test.copy()
