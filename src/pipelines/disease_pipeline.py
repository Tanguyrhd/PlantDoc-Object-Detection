"""
Disease Pipeline
Pipeline for multi-class disease classification.
"""

import pandas as pd
from .base_pipeline import BasePipeline


class DiseasePipeline(BasePipeline):
    """Pipeline for disease classification (excludes healthy plants)."""

    def get_pipeline_type(self) -> str:
        """Get pipeline type identifier."""
        return 'disease'

    def get_class_column(self) -> str:
        """Get the column name used for classification."""
        return 'disease'

    def filter_data(self):
        """
        Filter out healthy samples and rare/excluded diseases.
        """
        print(f"\n{'='*60}")
        print(f"FILTERING DISEASE SAMPLES")
        print(f"{'='*60}\n")

        # Step 1: Filter out healthy samples
        df_diseases_train = self.df_train[self.df_train['disease'] != 'healthy'].copy()
        df_diseases_test = self.df_test[self.df_test['disease'] != 'healthy'].copy()

        print(f"After removing healthy samples:")
        print(f"  Training: {len(df_diseases_train)} disease samples")
        print(f"  Validation: {len(df_diseases_test)} disease samples")

        # Step 2: Identify and remove rare diseases
        disease_proportions = df_diseases_train['disease'].value_counts(normalize=True)
        rare_diseases = disease_proportions[
            disease_proportions < self.config.rare_disease_threshold
        ].index.tolist()

        print(f"\nRare diseases (< {self.config.rare_disease_threshold*100}%):")
        for disease in rare_diseases:
            count = len(df_diseases_train[df_diseases_train['disease'] == disease])
            print(f"  {disease}: {count} samples")

        # Step 3: Combine rare and manually excluded diseases
        all_excluded = list(set(rare_diseases + self.config.excluded_diseases))

        print(f"\nManually excluded diseases:")
        for disease in self.config.excluded_diseases:
            print(f"  {disease}")

        print(f"\nAll excluded diseases: {all_excluded}")

        # Step 4: Remove excluded diseases
        df_diseases_clean_train = df_diseases_train[
            ~df_diseases_train['disease'].isin(all_excluded)
        ].copy()

        df_diseases_clean_test = df_diseases_test[
            ~df_diseases_test['disease'].isin(all_excluded)
        ].copy()

        print(f"\nAfter removing rare and excluded diseases:")
        print(f"  Training: {len(df_diseases_clean_train)} samples")
        print(f"  Validation: {len(df_diseases_clean_test)} samples")

        # Update dataframes
        self.df_train = df_diseases_clean_train
        self.df_test = df_diseases_clean_test

        # Show disease distribution
        self.balancer.print_distribution(
            self.df_train,
            'disease',
            "Training disease distribution (before balancing)"
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

        distribution = self.df_train['disease'].value_counts().sort_index()
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
                column='disease',
                target_samples_per_class=target_samples
            )

            self.balancer.print_distribution(
            self.df_train_processed,
            'disease',
            "Training disease distribution (after balancing)"
            )
            print("✓ Training dataset balanced successfully")
        else:
            print("\n✓ Keeping natural distribution (no balancing)")
            self.df_train_processed = self.df_train.copy()

        # Test set is never balanced
        self.df_test_processed = self.df_test.copy()
