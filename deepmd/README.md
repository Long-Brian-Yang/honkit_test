# DeePMD-kit Quick Start Tutorial

## Table of Contents

- [DeePMD-kit Quick Start Tutorial](#deepmd-kit-quick-start-tutorial)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Task](#2-task)
  - [3. Background](#3-background)
  - [4. Practice](#4-practice)
    - [4.1. Data Preparation](#41-data-preparation)
    - [4.2. Prepare Input Script](#42-prepare-input-script)
    - [4.3. Train a Model](#43-train-a-model)
    - [4.4. Freeze a Model](#44-freeze-a-model)
    - [4.5. Compress a Model (Optional)](#45-compress-a-model-optional)
    - [4.6. Test a Model](#46-test-a-model)
    - [4.7. Run MD with LAMMPS](#47-run-md-with-lammps)
  - [5. References](#5-references)

---

## 1. Overview

**DeePMD-kit** is a software tool that employs neural networks to fit potential energy surfaces (PES) based on first-principles (DFT-level) data for molecular dynamics (MD) simulations. Once a DeePMD-kit model (or **Deep Potential**, DP) is trained, it can be used seamlessly in common MD engines such as **LAMMPS**, **GROMACS**, and **OpenMM**.

By leveraging machine learning and high-performance computing (HPC), DeePMD-kit can:

- Achieve _ab initio_ accuracy at a fraction of the computational cost.
- Scale to very large systems (hundreds of millions of atoms) thanks to HPC optimizations.
- Serve as a plug-and-play force field for advanced MD simulations in physics, chemistry, materials science, and other related fields.

---

## 2. Task

In this tutorial, we demonstrate the basic workflow for using **DeePMD-kit**:

1. **Prepare a dataset** (DFT or AIMD data).
2. **Convert** it to DeePMD-kit’s compressed format (using [dpdata](https://github.com/deepmodeling/dpdata) or other methods).
3. **Train** a DP model with `dp train`.
4. **Freeze** the trained model into a `.pb` file for inference.
5. (Optional) **Compress** the model for speed and memory savings.
6. **Test** the model’s accuracy.
7. **Use** the model in your favorite MD engine (e.g., LAMMPS) to run production molecular dynamics.

By the end of this tutorial, you will:

- Understand how to configure a DeePMD-kit training input in JSON.
- Convert raw data to the DeePMD-kit format.
- Train, test, and freeze a DP model.
- Run an MD simulation in LAMMPS using `pair_style deepmd`.

**Estimated time**: ~20 minutes.

---

## 3. Background

Here, we showcase an example of _gaseous methane_ (`CH4`). We assume you already have an ab initio MD trajectory or a set of static DFT calculations. The principal steps are:

1. **Prepare data**:
   - Convert your raw DFT or AIMD outputs (e.g., from VASP, CP2K, Quantum Espresso, ABACUS, LAMMPS) into DeePMD-kit’s standard compressed NumPy format.
2. **Train**:
   - Use `dp train input.json` with your training (and validation) sets.
3. **Freeze**:
   - Convert the TensorFlow checkpoint into a single `.pb` file.
4. **Compress** (optional):
   - Optimize the `.pb` network for faster inference.
5. **Test**:
   - Evaluate energies/forces vs. reference data.
6. **Run MD**:
   - Plug into LAMMPS with `pair_style deepmd`.

DeePMD-kit’s success has been recognized widely, including earning the **2020 ACM Gordon Bell Prize** for HPC.

---

## 4. Practice

Below is a hands-on demonstration. (All commands shown assume you have **DeePMD-kit**, **dpdata**, and **LAMMPS** installed in your environment.)

### 4.1. Data Preparation

1. **Acquire or generate** first-principles data. In this example, we have an **ABACUS** MD trajectory for methane.
2. **Convert** to DeePMD-kit format using [dpdata](https://github.com/deepmodeling/dpdata). For instance:

```python
import dpdata
import numpy as np

# Load data of ABACUS MD format
data = dpdata.LabeledSystem("00.data/abacus_md", fmt="abacus/md")
print(f"Number of frames: {len(data)}")

# Randomly select validation frames
rng = np.random.default_rng()
index_val = rng.choice(len(data), size=40, replace=False)
index_train = list(set(range(len(data))) - set(index_val))

data_train = data.sub_system(index_train)
data_val   = data.sub_system(index_val)

# Save in DeePMD-kit format
data_train.to_deepmd_npy("00.data/training_data")
data_val.to_deepmd_npy("00.data/validation_data")

print(f"Training frames:   {len(data_train)}")
print(f"Validation frames: {len(data_val)}")
```

3. After this step, you should see directories like:

   ```
   00.data/
   ├── abacus_md/
   ├── training_data/
   └── validation_data/
   ```

   Each contains a `set.000` directory with compressed data, plus `type.raw` and `type_map.raw`.

---

### 4.2. Prepare Input Script

DeePMD-kit requires a **JSON** file specifying the training hyperparameters, network architecture, and file paths. Below is an **example** `input.json`:

```jsonc
{
  "model": {
    "type_map": ["H", "C"],
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "rcut_smth": 0.5,
      "sel": "auto",
      "neuron": [25, 50, 100],
      "axis_neuron": 16,
      "resnet_dt": false,
      "seed": 1
    },
    "fitting_net": {
      "neuron": [240, 240, 240],
      "resnet_dt": true,
      "seed": 1
    }
  },

  "learning_rate": {
    "type": "exp",
    "decay_steps": 50,
    "start_lr": 0.001,
    "stop_lr": 3.51e-8
  },

  "loss": {
    "type": "ener",
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0,
    "limit_pref_v": 0
  },

  "training": {
    "training_data": {
      "systems": ["../00.data/training_data"],
      "batch_size": "auto"
    },
    "validation_data": {
      "systems": ["../00.data/validation_data"],
      "batch_size": "auto",
      "numb_btch": 1
    },
    "numb_steps": 10000,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 200,
    "save_freq": 1000
  }
}
```

**Key parameters**:

- **type_map**: Mapping of atomic types (`["H", "C"]` here).
- **descriptor**: E2_a descriptor with cutoff radius = 6.0 Å, smoothing start = 0.5 Å, neural network sizes, etc.
- **fitting_net**: Fitting network architecture `[240, 240, 240]`.
- **loss**: Weighted training for energies (`pref_e`) and forces (`pref_f`).
- **training**: Number of steps (`numb_steps` = 10000), batch sizes, data paths.

---

### 4.3. Train a Model

From the directory containing your `input.json`, simply run:

```bash
dp train input.json
```

DeePMD-kit will print periodic output like:

```
DEEPMD INFO    batch    1000 training time ...
DEEPMD INFO    saved checkpoint model.ckpt
...
```

and produce a `lcurve.out` containing stepwise loss metrics. Key columns are:

1. **step**
2. **rmse_val**
3. **rmse_trn**
4. **rmse_e_val** (energy validation error per atom)
5. **rmse_e_trn** (energy training error per atom)
6. **rmse_f_val** (forces validation error)
7. **rmse_f_trn** (forces training error)
8. **learning rate**

You should see the RMS errors dropping over steps.

---

### 4.4. Freeze a Model

DeePMD-kit stores the model in TensorFlow checkpoints by default. To create a single `.pb` file:

```bash
dp freeze -o graph.pb
```

This **frozen** model is used for inference in MD codes like LAMMPS.

---

### 4.5. Compress a Model (Optional)

Model compression can speed up inference further:

```bash
dp compress -i graph.pb -o compress.pb
```

This creates a `compress.pb`. You can use it in place of `graph.pb` if desired.

---

### 4.6. Test a Model

To evaluate the model on a validation set and compare energies/forces:

```bash
dp test -m graph.pb -s ../00.data/validation_data
```

It will print average and RMS errors for energy, force, and virial:

```
Energy MAE         : x.xxxe-03 eV
Energy MAE/Natoms  : x.xxxe-04 eV
Force  MAE         : x.xxxe-02 eV/A
...
```

You can also test using Python:

```python
import dpdata
val_system = dpdata.LabeledSystem("../00.data/validation_data", fmt="deepmd/npy")
prediction = val_system.predict("graph.pb")

# For a quick correlation plot:
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(val_system["energies"], prediction["energies"], alpha=0.5)
plt.plot([min(val_system["energies"]), max(val_system["energies"])],
         [min(val_system["energies"]), max(val_system["energies"])],
         'r--', linewidth=0.5)
plt.xlabel("DFT Energy (eV)")
plt.ylabel("DP Predicted Energy (eV)")
plt.show()
```

---

### 4.7. Run MD with LAMMPS

Copy your model file (e.g., `graph.pb`) to your LAMMPS run directory.  
Then write an **in.lammps** with something like:

```lammps
units           metal
atom_style      atomic
boundary        p p p

read_data       conf.lmp

pair_style      deepmd graph.pb
pair_coeff      * *

timestep        0.001
thermo          100
run             5000
```

Finally, run LAMMPS:

```bash
lmp -i in.lammps
```

You will see LAMMPS output indicating a **Deep Potential** style, and the system will evolve under your DP potential.

---

## 5. References

- **DeePMD-kit** documentation:  
  <https://deepmd.readthedocs.io>
- **dpdata** for data conversion:  
  <https://github.com/deepmodeling/dpdata>
- Original DeePMD-kit paper:
  - Wang, Han, et al. _Comput. Phys. Commun._ **228**, 178–184 (2018).
- Gordon Bell Prize 2020 highlight:
  - <https://deepmodeling.com/gordon-bell-2020/>

For detailed installation instructions, additional examples, GPU optimization tips, and advanced features (like active learning, MLPot for automatically exploring configuration space, etc.), please consult the official [DeePMD-kit Documentation](https://deepmd.rtfd.io/).

**Happy Deep Potential Modeling!**
