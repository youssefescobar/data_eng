import subprocess
try:
    subprocess.run(["bash", "./auto_commit.sh"], check=True)
    print("Auto-commit script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running the auto-commit script: {e}")