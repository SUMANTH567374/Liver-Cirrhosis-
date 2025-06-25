# src/features/feature_selection.py

import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.data_ingestion import setup_logger

def select_features(data: pd.DataFrame, target_column: str, drop_columns: list, logger=None, top_n=14):
    if logger is None:
        logger = logging.getLogger()

    try:
        logger.info("Starting feature selection using Random Forest...")

        # Drop unwanted columns before splitting
        cols_to_drop = drop_columns + ['Sex', 'Spiders', 'Edema', 'Hepatomegaly']
        data = data.drop(columns=cols_to_drop)

        X = data.drop(columns=[target_column])
        y = data[target_column].astype(int)

        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Class distribution after SMOTE: {dict(pd.Series(y_resampled).value_counts())}")

        # Create and save balanced DataFrame
        balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_data[target_column] = y_resampled

        final_dir = os.path.join(BASE_DIR, "data", "final")
        os.makedirs(final_dir, exist_ok=True)
        balanced_path = os.path.join(final_dir, "balanced_data.csv")
        balanced_data.to_csv(balanced_path, index=False)
        logger.info(f"Balanced data saved to: {balanced_path}")

        # Train Random Forest for feature selection
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_resampled, y_resampled)

        # Get and log top features
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(top_n)
        logger.info(f"Top {top_n} features:\n{top_features}")

        # Plot and save feature importance
        plt.figure(figsize=(8, 6))
        top_features.plot(kind='barh', title=f'Top {top_n} Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_dir = os.path.join(BASE_DIR, "graphs")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"top_{top_n}_features.png")
        plt.savefig(plot_path)
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.show()

        return top_features.index.tolist()

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cleaned_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    logger = setup_logger(os.path.join(BASE_DIR, "logs", "feature_selection.log"))

    if not os.path.exists(cleaned_path):
        logger.error(f"Cleaned data not found at {cleaned_path}")
        exit()

    df = pd.read_csv(cleaned_path)

    selected_features = select_features(
        data=df,
        target_column='Status',
        drop_columns=['N_Days'],
        logger=logger,
        top_n=14
    )

    # Save selected feature names
    final_dir = os.path.join(BASE_DIR, "data", "final")
    os.makedirs(final_dir, exist_ok=True)
    feature_path = os.path.join(final_dir, "selected_features.txt")

    with open(feature_path, "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    logger.info(f"Selected features saved to {feature_path}")
