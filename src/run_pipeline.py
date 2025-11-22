import subprocess
import sys

def run(script):
    print("\n" + "="*70)
    print(f"Running {script}...")
    print("="*70)
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    print("\n Starting Full Resume–Job Fit Pipeline\n")
    
    # 1. Data preproccessing
    # run("data_preprocessing.py")    # Optional to run because running time is slow

    # 2. Train Fit / No Fit model 
    run("new_scripts/train_fit_model.py")

    # 3. Generate Recommendations
    run("new_scripts/recommend_jobs.py")

    # 4. Generate Visualizations
    run("new_scripts/generate_visualizations.py")

    print("\n Pipeline Complete! All results saved in datasets/ & visualizations/")
