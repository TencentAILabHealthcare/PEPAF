import subprocess
import sys
import os
def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f"Running command: {command}")

    os.system(command)

def main():
    print("Welcome to the automation script for affinity prediciton and guided evolution!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Run PepAF prediction for PDBbind")
        print("2. Run PepAF prediction for single pair")
        print("3. Run PepAF prediction for antigen-HLA")
        print("4. Run PepAF-gudied peptide evolution")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            run_command("cd PepAF && python predict.py --task pdbbind")
            print("Results were saved in PepAF/output/pdbbind.tsv")
            break
        elif choice == '2':
            run_command("cd PepAF && python predict.py --task single")
            break
        elif choice == '3':
            run_command("cd PepAF && python predict.py --task pmhc")
            print("Results were saved in PepAF/output/pmhc.tsv")
            break
        elif choice == '4':
            run_command("cd PepRL && sh run.sh")
            break
        elif choice == '5':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()