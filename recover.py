import os
import shutil

blob_dir = ".git/lost-found/other"
out_dir = "recovered_code"
os.makedirs(out_dir, exist_ok=True)

for blob in os.listdir(blob_dir):
    blob_path = os.path.join(blob_dir, blob)
    with open(blob_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        content = "".join(lines)

        # Default name with the hash
        filename = f"unknown_{blob[:6]}.txt"

        # Identify Slurm scripts and use their job name
        if "#SBATCH --job-name=" in content:
            for line in lines:
                if line.startswith("#SBATCH --job-name="):
                    job_name = line.split("=")[1].strip()
                    filename = f"{job_name}.slurm"
                    break
        # Identify Python scripts (looks for standard Python keywords)
        elif "import " in content or "def " in content or "polars" in content:
            # Try to guess the python filename from the first few lines if possible
            filename = f"python_script_{blob[:6]}.py"
        # Identify R scripts
        elif "library(" in content or "<-" in content:
            filename = f"r_script_{blob[:6]}.R"

        shutil.copy(blob_path, os.path.join(out_dir, filename))

print(f"Success! Files recovered to the '{out_dir}/' folder.")
