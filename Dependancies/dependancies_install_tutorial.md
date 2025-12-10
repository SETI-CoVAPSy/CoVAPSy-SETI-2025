# CoVAPSy Project Dependencies

This repository outlines the Python and external dependencies required to run the CoVAPSy project, which involves Python dependencies for Webots simulation and hardware/simu controller interactions.

## 1. Python Dependencies (Managed by `uv`)

All core Python packages are defined in the `pyproject.toml` file at the root of this repository. We recommend using `uv` for ultra-fast dependency resolution and installation.

### Prerequisites

1.  **Install `uv`:** Ensure you have the `uv` package manager installed.
    ```bash
    # Via curl
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh 
    # Or, if you use Conda
    conda install uv -c conda-forge 
    ```

### Setup Instructions

1.  **Navigate to the project root:**
    ```bash
    cd /path/to/covapsy
    ```

2.  **Create a Virtual Environment:**
    ```bash
    uv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The following command reads the `[project.dependencies]` section in `pyproject.toml` and installs all packages from PyPI:
    ```bash
    uv pip install .
    ```

---

## 2. External / System Dependencies

Due to the project's reliance on the ROS 2 (Humble) and Webots ecosystems, several core components are not available on PyPI and must be installed separately via a Conda environment (recommended for robotic projects) or system package managers (e.g., `apt`, `dnf`).

**The following packages must be installed externally:**

| Dependency | Purpose | Recommended Installation Method |
| :--- | :--- | :--- |
| **ROS 2 Humble** | Core Robotic Operating System environment. | System Installation (see official ROS 2 docs) or Robostack Conda. |
| **`python-orocos-kdl`** | Python bindings for the Kinematics and Dynamics Library (KDL). | Conda (Robostack) or manual compilation. |
| **`webots-ros2-driver`** | Bridge between ROS 2 and Webots simulation. | System Package Manager (`apt`/`dnf`) or Conda (Robostack). |

### Installation via Conda (Recommended for KDL & Webots Drivers)

If you use Conda (with the Robostack channels configured), you can create your base environment and install these specialized packages before running `uv`:

```bash
# 1. Create your Conda environment
conda create -n covapsy-base python=3.12 
conda activate covapsy-base

# 2. Install KDL (assuming Robostack channels are set up)
conda install -c robostack-staging python-orocos-kdl=1.5.3 

# 3. Install Webots driver (or use system packages)
conda install -c robostack-staging webots-ros2-driver=2025.0.0
# OR (if using system packages) Ensure the driver is installed via your system

# 4. Install uv, then the rest of the dependencies from pyproject.toml
conda install uv -c conda-forge
uv pip install .
