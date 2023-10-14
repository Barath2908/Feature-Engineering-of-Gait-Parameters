import os
import multiprocessing

def run_file(file):
    os.system(f"python {file}")

if __name__ == "__main__":
    common_path = "C:/Users/ibra5/Desktop/mafese-main/"  # Replace with your common path
    files = ["Correlation.py", "Fwiz.py", "Lasso.py", "Recursive.py", "Sequential.py", "Meta.py"]  # Replace with your file names

    # Create a multiprocessing pool with the number of available CPU cores
    pool = multiprocessing.Pool()

    # Join the common path with each file name
    files = [os.path.join(common_path, file) for file in files]

    # Run each file in parallel using the pool
    pool.map(run_file, files)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
