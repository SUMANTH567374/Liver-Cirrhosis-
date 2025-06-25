import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.data.data_ingestion import setup_logger
import yaml

# Define base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_params():
    params_path = os.path.join(BASE_DIR, "params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def select_features(data: pd.DataFrame, target_column: str, drop_columns: list, top_n: int, logger=None):
    if logger is None:
        logger = logging.getLogger()

    try:
        logger.info("Starting feature selection using Random Forest...")

        # Drop unwanted columns
        cols_to_drop = drop_columns + ['Sex', 'Spiders', 'Edema', 'Hepatomegaly']
        data = data.drop(columns=cols_to_drop)

        X = data.drop(columns=[target_column])
        y = data[target_column].astype(int)

        # SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Class distribution after SMOTE: {dict(pd.Series(y_resampled).value_counts())}")

        # Save balanced data
        balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_data[target_column] = y_resampled

        final_dir = os.path.join(BASE_DIR, "data", "final")
        os.makedirs(final_dir, exist_ok=True)
        balanced_path = os.path.join(final_dir, "balanced_data.csv")
        balanced_data.to_csv(balanced_path, index=False)
        logger.info(f"Balanced data saved to: {balanced_path}")

        # Train Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_resampled, y_resampled)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(top_n)
        logger.info(f"Top {top_n} features:\n{top_features}")

        # Save plot to graphs/feature_selection/
        plot_dir = os.path.join(BASE_DIR, "graphs", "feature_selection")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"top_{top_n}_features.png")

        plt.figure(figsize=(8, 6))
        top_features.plot(kind='barh', title=f'Top {top_n} Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close()

        return top_features.index.tolist()

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise

if __name__ == "__main__":
    cleaned_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    logger = setup_logger(os.path.join(BASE_DIR, "logs", "feature_selection.log"))

    if not os.path.exists(cleaned_path):
        logger.error(f"Cleaned data not found at {cleaned_path}")
        exit()

    # ✅ Load nested top_n from features.top_n
    params = load_params()
    top_n = params["features"]["top_n"]

    df = pd.read_csv(cleaned_path)

    selected_features = select_features(
        data=df,
        target_column='Status',
        drop_columns=['N_Days'],
        top_n=top_n,
        logger=logger
    )

    feature_path = os.path.join(BASE_DIR, "data", "final", "selected_features.txt")
    with open(feature_path, "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    logger.info(f"Selected features saved to {feature_path}")
