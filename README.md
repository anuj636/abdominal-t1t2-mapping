# abdominal-t1t2-mapping

Example source code to perform reconstruction and post-processing for abdominal 3D T1 and T2 mapping

_Jonathan Stelter, Kilian Weiss, Lisa Steinhelfer, Jakob Meineke, Weitong Zhang, Bernhard Kainz, Rickmer F. Braren, Dimitrios C. Karampinos, Abdominal simultaneous 3D water T1 and T2 mapping using a free-breathing Cartesian acquisition with spiral profile ordering, Magnetic Resonance in Medicine, DOI: [10.1002/mrm.70040](https://doi.org/10.1002/mrm.70040)_

## 🚀 Setup

### Requirements

- Julia 1.9 (system-wide installation is recommened)
- GPU for Julia reconstruction with sufficient memory (tested with NVIDIA RTX A6000 with 48 GB of graphics memory)
- Anaconda/mamba environment with Python 3.10

### Installing

1. **Create a new mamba environment:**

    ```shell
    mamba env create --name t1t2mapping --file environment_nocuda.yml
    mamba activate t1t2mapping
    ```

2. **Open Julia and instantiate/precompile new Julia environment:**

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

3. **Run processing scripts directly from the shell:**

    ```shell
    julia -i scripts/recon.jl
    python -i scripts/postprocessing.py
    ```

## Data
Raw data for the T1 mapping phantom and the dictionary are stored [here](https://1drv.ms/f/c/fbf9e7f0b84c6ec2/Eo6aFR0d1VJHo9CT7WewBesBDQYer-YNmjBYQXYIwu5PpQ?e=FTG8px).

## Authors and acknowledgment
* Jonathan Stelter - [Body Magnetic Resonance Research Group, TUM](http://bmrr.de)

**Dual-echo water-fat separation: https://github.com/BMRRgroup/2echo-WaterFat-hmrGC**

## License
This project is licensed as given in the LICENSE file. However, used submodules / projects may be licensed differently. Please see the respective licenses.