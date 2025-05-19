# LoopStructural Mini GUI

## Description

A desktop application built with PySide6 and PyVista for creating, configuring, and visualizing 3D geological models using the LoopStructural library. This GUI allows users to load fault and stratigraphy data, define model parameters, build geological models, and interactively view the results in a 3D environment. This tool is specifically designed for and intended to be implemented into the PZERO project.

## Features

-   **Data Loading:**
    -   Load fault geometry data from VTK/VTP files. Each fault is managed separately.
    -   Load stratigraphy data from VTK/VTP files (points and optional normals/values).
-   **Multiple Fault Handling:**
    -   Supports loading and modeling of multiple distinct fault structures.
    -   List view for loaded faults, allowing selection and parameter inspection.
-   **Automated Fault Geometry Calculation:**
    -   Calculates fault center and axis lengths based on input point data.
    -   Uses Principal Component Analysis (PCA) for point clouds with 10 or more points.
    -   Uses a planar fitting method for point clouds with fewer than 10 points.
-   **Manual Override:**
    -   Option to override automatically calculated fault geometry parameters (center, major/minor axis lengths).
    -   Intermediate axis length for faults is manually configurable.
-   **Model Configuration:**
    -   Define model origin (X, Y, Z) and maximum extent (X, Y, Z).
-   **Interpolator Selection:**
    -   Choose interpolator type (PLI, FDI, Surfe) for both faults and foliations.
-   **Fault Parameters:**
    -   Set fault displacement.
    -   Configure `nelements` for fault discretization.
    -   Define fault buffer.
-   **Foliation Parameters:**
    -   Configure `nelements` for foliation.
-   **Interactive 3D Visualization (PyVista):**
    -   Displays loaded fault and stratigraphy input data (points, wireframes).
    -   Visualizes calculated fault axes (PCA components or planar fit axes and normal).
    -   Renders the resulting LoopStructural model, including fault surfaces and foliation isosurfaces.
    -   Real-time updates to the visualization upon model changes or fault selection.
-   **Status Logging:**
    -   A dedicated area displays status messages, warnings, and errors.

## Requirements

-   Python 3.7+
-   PySide6
-   LoopStructural
-   pandas
-   numpy
-   pyvista
-   pyvistaqt
-   scikit-learn

## Installation

1.  **Clone the repository (if applicable) or ensure you have the `loop_structural_mini_gui.py` file.**
2.  **Install the required Python packages:**
    ```bash
    pip install PySide6 LoopStructural pandas numpy pyvista pyvistaqt scikit-learn
    ```

## Usage

1.  **Run the application:**
    ```bash
    python loop_structural_mini_gui.py
    ```
2.  **Load Data:**
    -   Click "Load Fault VTK" to load one or more fault geometry files (`.vtk` or `.vtp`). You will be prompted to enter a unique name for each fault.
    -   Click "Load Strati VTK" to load a stratigraphy data file (`.vtk` or `.vtp`). This file should contain points, and can optionally include 'Normals' and 'val' (or 'vals') point data arrays.
3.  **Configure Model & Features:**
    -   Set the "Model Origin" and "Model Maximum" coordinates.
    -   Select a fault from the "Loaded Faults" list to view/edit its parameters.
    -   If needed, check "Override Calculated Fault Geometry" to manually input fault center and axis lengths for the selected fault.
    -   Adjust other fault parameters (Displacement, Nelements, Interpolator Type, Buffer).
    -   Adjust Foliation Parameters (Nelements, Interpolator Type).
4.  **Build Model:**
    -   Click the "Build Model & Features" button.
    -   The application will process the data, build the LoopStructural model, and update the PyVista 3D view.
5.  **Interact with Visualization:**
    -   The right-hand pane shows the 3D model.
    -   Input data (fault points/meshes, stratigraphy points) are shown.
    -   Calculated axes for the selected (or all, if only one) fault are displayed.
    -   Modeled fault surfaces and foliation isosurfaces are rendered.

## File Structure

-   `loop_structural_mini_gui.py`: The main Python script containing the application logic and GUI definition.
-   `README.md`: This file.
-   (User-provided VTK/VTP data files for faults and stratigraphy)

---

This GUI provides a simplified interface for common LoopStructural workflows, particularly focusing on fault and foliation modeling with visual feedback. 