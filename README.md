
# PepAF

PepAF is a cross-domain knowledge transfer framework for protein-peptide affinity prediction, antigen-HLA affinity prediction, and guided peptide evolution.


## Installation and Setup

1. **Hardware requirements**:
   This project requires only a standard computer with enough RAM and a NVIDIA GPU to support operations. We ran the demo using the following specs:
   - CPU: 10 cores, 2.5 GHz/core
   - RAM: 40GB
   - GPU: NVIDIA TESLA P40, V100, A100
   - CUDA: 11.0

2. **System requirements**:
   This tool is supported for Linux. The tool has been tested on the following system:

   - CentOS Linux release 8.2.2.2004

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/TencentAILabHealthcare/PEPAF.git
   cd PepAF
   ```

4. **Install Required Packages**:
   The basic environment requirements are:
   - Python: 3.10
   - CUDA: 11.0

   Use the following command to install the necessary packages as specified in the `requirements.txt` file:

   ```bash
   conda create -n PepAF python==3.10
   conda activate PepAF
   pip install -r requirements.txt
   ```

5. **Download Model Weights**:

   Download the `model_weights.zip` file and extract it to the `PepAF/model_weights` directory. The model_weights.zip is available on Zenodo: <https://doi.org/10.5281/zenodo.15050396>

   After extraction, the `PepAF/model_weights` directory should contain the following:

   ```plaintext
   PepAF/model_weights/
   ├── ESM-2/
   ├── ESM-Pep/
   ├── PepAF/
   └── PepAF_pmhc
   ```

6. **Download PDBBind Data**:

   Download the `pdbbind_data.zip` file and extract it to the `PepAF/PepAF` directory, specifically to `PepAF/PepAF/pdbbind_data`. The pdbbind_data.zip is available on Zenodo: <https://doi.org/10.5281/zenodo.15050396>

   After extraction, the `PepAF/PepAF/pdbbind_data` directory should contain the following:

   ```plaintext
   PepAF/PepAF/pdbbind_data/
   ├── all_data.tsv
   ├── pep/
   └── pro/
   ```

7. **Download Antigen-HLA Data**:

   Download the `pmhc_data.zip` file and extract it to the `PepAF/PepAF` directory, specifically to `PepAF/PepAF/pmhc_data`. The pmhc_data.zip is available on Zenodo: <https://doi.org/10.5281/zenodo.15050396>

   After extraction, the `PepAF/PepAF/pmhc_data` directory should contain the following:

   ```plaintext
   PepAF/PepAF/pmhc_data/
   ├── test_data.tsv
   ├── mhc/
   └── pep/
   ```
  
8. **Download Receptor Data**:

   Download the `receptor_data.zip` file and extract it to the `PepAF/PepAF` directory, specifically to `PepAF/PepAF/receptor_data`. The receptor_data.zip is available on Zenodo: <https://doi.org/10.5281/zenodo.15050396>

   After extraction, the `PepAF/PepAF/receptor_data` directory should contain the following:

   ```plaintext
   PepAF/PepAF/receptor_data/
   ├── coordinates.json
   ├── mod_rec_seq.json
   ├── rec_interface.json
   ├── supported_receptor_pdbid.txt
   └── esm/
   ```

---

## Quick Start
   To quickly get started with PepAF, you can use the provided automation script. Follow these steps:

1. **Run the Automation Script**:
   After setting up the environment and downloading the necessary files, you can run the automation script to start an example task easily.

      ```bash
      python quick_start.py
      ```
   This will present you with a menu to select from the following options:

   ```
    Welcome to the automation script for affinity prediciton and guided evolution!

    Please choose an option:
    1. Run PepAF prediction for PDBbind
    2. Run PepAF prediction for single pair
    3. Run PepAF prediction for antigen-HLA
    4. Run PepAF-gudied peptide evolution
    5. Exit
    Enter your choice (1-5):
   ```

   1. **Run PepAF prediction for PDBbind**
   - Select option 1 allows you to perform a binding affinity prediction for the PDBBind dataset.
   ```
   Enter your choice (1-5): 1
   Running command: cd PepAF && python predict.py --task pdbbind
   Results were saved in PepAF/output/pdbbind.tsv
   ```
   2. **Run PepAF prediction for single pair**
   - Selection option 2 allows you to perform a binding affinity prediction for a given protein and a peptide. If you want to specify parameters such as the target protein and the peptide sequence, please refer to Further instructions 2.
   ```
    Enter your choice (1-5): 2
    Running command: cd PepAF && python predict.py --task single
    Starting prediction for protein: 5yqz_R, peptide: HSQGTFTSDYSKYLDSERAQEFVQWLENE
    Dataset created.
    Models loaded successfully.
    Predicted binding affinity between 5yqz_R and HSQGTFTSDYSKYLDSERAQEFVQWLENE: 10.12
   ```

   3. **Run PepAF prediction for antigen-HLA**
   - Selection option 3 allows you to perform a binding affinity prediction for the antigen-HLA test data. If you want to specify parameters such as the input data, please refer to Further instructions 3.
   ```
   Enter your choice (1-5): 3
   Running command: cd PepAF && python predict.py --task pmhc
   Building data ...
   Predicting...
   Results were saved in PepAF/output/pmhc.tsv
   ```

   4. **Run PepAF-gudied peptide evolution**
   - Select option 2 will start the Reinforcement Learning (RL) optimization process. This process will optimize the given peptide sequence targeting a specific protein (pdb id) to search better mutated sequences with higher affinity. If you want to specify parameters such as the target protein and the initial peptide sequence, please refer to Further instructions 2.
   ```
    Enter your choice (1-5): 4
    Running command: cd PepRL && sh run.sh
    Target PDB: 7lll_R, Start sequence: TFQKWAAVVVPSG
    ################## model loaded on cuda #####################
    ######### 1-th Play ###########
    Mutated seq TFRKWAAVVVPSG
    Mutated seq TFREWAAVVVPSG
    Mutated seq TFRIWAAVVVPSG
    Mutated seq PFRIWAAVVVPSG
    ######### 2-th Play ###########
    Mutated seq TFRIGAAVVVPSG
    ######### 3-th Play ###########
    Mutated seq TFRIWAAVVVSSG
    ...
   ```

   5. **Exit:**
   - Choosing this option will exit the script and terminate the current session.

   Simply enter the corresponding number to execute your desired task.

---

## Further Instructions
### 1. Run PepAF prediction for PDBbind

Navigate to the `PepAF` directory and run the example code:
```bash
cd PepAF/PepAF
python predict.py --task pdbbind
```
This will utilize five models to predict the final results.


### 2. Run PepAF prediction for single pair

A list of supported PDB IDs (35,473) can be found in `PepAF/PepAF/receptor_data/supported_receptor_pdbid.txt`.

Since PepAF requires various features for prediction, we support an approache to simply use PepAF:

Use the built-in feature extraction by specifying the target PDB ID and peptide sequence in `predict.py`:
```bash
cd PepAF/PepAF
python predict.py --task single
```

### 3. Run PepAF prediction for antigen-HLA
Navigate to the `PepAF` directory and run the example code:

```bash
cd PepAF/PepAF
python predict.py --task pmhc
```

If you want to test self-data, please use AlphaFold2 to predict the complete HLA structure, and use ESM2 and ESM-pep to extract evolutionary embeddings for HLA and antigen sequences.

### 4. Run PepAF-gudied peptide evolution

Navigate to the `PepRL` directory and run the optimization script:
```bash
cd PepAF/PepRL
sh run.sh
```

#### Key Parameter Explanations:
- `PLAYOUT_NUM=20`: The number of rounds for each game.
- `NITER=100`: The number of optimization iterations.
- `WORKDIR='./'`: The working directory.
- `PDBID='7lll_R'`: The PDB ID of the target protein.
- `SEQ='QDEEGLLLMQSLEMS'`: The peptide sequence to be optimized.
- `OUTPUT_DIR="./results/${PDBID}/${SEQ}"`: The directory for output results.

A list of supported PDB IDs can be found in `PepAF/PepAF/receptor_data/supported_receptor_pdbid.txt`.