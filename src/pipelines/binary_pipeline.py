"""
Binary Pipeline
Pipeline for binary classification (Healthy vs Disease).
"""

import pandas as pd
from .base_pipeline import BasePipeline


class BinaryPipeline(BasePipeline):
    """Pipeline for binary classification: healthy (0) vs disease (1)."""

    def get_pipeline_type(self) -> str:
        """Get pipeline type identifier."""
        return 'binary'

    def get_class_column(self) -> str:
        """Get the column name used for classification."""
        return 'binary_class'

    def filter_data(self):
        """
        Create binary labels: 0 = healthy, 1 = disease.
        No filtering needed - we keep all samples.
        """
        print(f"\n{'='*60}")
        print(f"CREATING BINARY LABELS")
        print(f"{'='*60}\n")

        # Create binary class column
        self.df_train['binary_class'] = (self.df_train['disease'] != 'healthy').astype(int)
        self.df_test['binary_class'] = (self.df_test['disease'] != 'healthy').astype(int)

        # Show distribution
        healthy_train = (self.df_train['binary_class'] == 0).sum()
        disease_train = (self.df_train['binary_class'] == 1).sum()

        healthy_test = (self.df_test['binary_class'] == 0).sum()
        disease_test = (self.df_test['binary_class'] == 1).sum()

        print(f"Training set:")
        print(f"  Class 0 (Healthy): {healthy_train} samples")
        print(f"  Class 1 (Disease): {disease_train} samples")
        print(f"  Ratio: {healthy_train/disease_train:.2f}:1 (Healthy:Disease)")

        print(f"\nValidation set:")
        print(f"  Class 0 (Healthy): {healthy_test} samples")
        print(f"  Class 1 (Disease): {disease_test} samples")

        print(f"\n✓ Binary labels created")

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

        distribution = self.df_train['binary_class'].value_counts().sort_index()
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
                            max_possible = distribution.min()
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
                column='binary_class',
                target_samples_per_class=target_samples
            )

            # Show new distribution
            new_distribution = self.df_train_processed['binary_class'].value_counts().sort_index()
            print("\n📊 Training set - Balanced distribution:")
            for label, count in new_distribution.items():
                label_name = "Healthy" if label == 0 else "Diseased"
                percentage = (count / len(self.df_train_processed)) * 100
                print(f"  {label_name:12} (label {label}): {count:5} samples ({percentage:5.1f}%)")

            print(f"\n  Total training: {len(self.df_train_processed)} samples")
            print("✓ Training dataset balanced successfully")
        else:
            print("\n✓ Keeping natural distribution (no balancing)")
            self.df_train_processed = self.df_train.copy()

        # Test set is never balanced
        self.df_test_processed = self.df_test.copy()
