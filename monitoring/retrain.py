import subprocess
import sys
from datetime import datetime


def retrain_model():
    """
    Automatically runs the ML training/experiment pipeline.
    experiment.py trains multiple models, logs them to MLflow,
    selects the best model, and saves it.
    """

    print("\n====================================")
    print("AUTOMATIC MODEL RETRAINING")
    print("====================================")

    print(
        "Started at:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print("\nStarting retraining pipeline...\n")

    try:

        # Use the same Python interpreter as the current virtual environment
        result = subprocess.run(
            [
                sys.executable,
                "src/experiment.py"
            ],
            capture_output=True,
            text=True
        )

        # Show experiment.py output
        print(result.stdout)

        # Check whether training succeeded
        if result.returncode == 0:

            print("\n====================================")
            print("RETRAINING SUCCESSFUL")
            print("====================================")

            print(
                "Completed at:",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            print(
                "\nBest model has been saved to:"
            )

            print(
                "models/churn_model.joblib"
            )
            
            print("\nStarting model validation...")

            promotion_result = subprocess.run(
                [
                    sys.executable,
                    "monitoring/promote.py"
                ],
                text=True
            )

            if promotion_result.returncode != 0:
                print("Model validation/promotion failed.")
                return False

            return True

        else:

            print("\n====================================")
            print("RETRAINING FAILED")
            print("====================================")

            print(result.stderr)

            return False

    except Exception as error:

        print("\nUnexpected error during retraining:")

        print(error)

        return False


# --------------------------------------------------
# RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    retrain_model()