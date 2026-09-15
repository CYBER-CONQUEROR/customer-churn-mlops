import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_mlops_pipeline():

    print("\n====================================")
    print("AUTOMATED MLOPS PIPELINE")
    print("====================================")

    print(
        "Started:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    drift_script = (
        PROJECT_ROOT / "monitoring" / "drift.py"
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(drift_script)
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )

        output = result.stdout

        if result.stderr:
            output += "\n" + result.stderr

        success = result.returncode == 0

        return {
            "success": success,
            "output": output
        }

    except Exception as error:

        return {
            "success": False,
            "output": str(error)
        }


if __name__ == "__main__":

    pipeline_result = run_mlops_pipeline()

    print(
        pipeline_result["output"]
    )

    if pipeline_result["success"]:

        print("\nMLOps pipeline completed.")

    else:

        print("\nMLOps pipeline failed.")