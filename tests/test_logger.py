import os
import logging

def setup_logger():
    log_dir = "test_reports"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test_log.txt")

    # Clear existing handlers to avoid duplicate logs in pytest reruns
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="a"  # Append mode
    )

    # Optional: also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)

