import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QGridLayout,
    QComboBox, QTextEdit, QFrame, QCheckBox, QListWidget, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QMenu,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QDoubleValidator, QAction, QIntValidator

import LoopStructural as LS
# import LoopStructural.visualisation as vis # No longer used
import pandas as pd
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor # Use QtInteractor for embedding
# Removed vtk import as PyVista handles it

# Helper for PCA
from sklearn.decomposition import PCA

def compute_geometric_features(polydata):
    """
    Compute geometric features using PCA (Principal Component Analysis).
    Returns center, axes vectors (normalized), and axes lengths.
    """
    if polydata is None or polydata.n_points == 0:
        # Return default values if polydata is invalid
        return np.array([0,0,0]), np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([0,0,0])

    # Always use PCA-based method for geometric analysis
    points = polydata.points
    center = np.mean(points, axis=0)
    
    if polydata.n_points <= 1: # Not enough points for covariance
        return center, np.identity(3), np.zeros(3)

    centered_points = points - center
    if polydata.n_points == 2: # Special handling for 2 points (a line)
        vec = centered_points[1] - centered_points[0]
        length1 = np.linalg.norm(vec)
        axis1 = vec / length1 if length1 > 1e-6 else np.array([1,0,0])
        # Create two orthogonal axes
        temp_axis = np.array([0,0,1]) if np.abs(np.dot(axis1, np.array([0,0,1]))) < 0.9 else np.array([0,1,0])
        axis2 = np.cross(axis1, temp_axis)
        axis2_norm = np.linalg.norm(axis2)
        if axis2_norm > 1e-6: axis2 /= axis2_norm
        else: axis2 = np.array([0,1,0] if axis1[0]!=0 or axis1[2]!=0 else [1,0,0]) # Ensure orthogonality
        
        axis3 = np.cross(axis1, axis2)
        axis3_norm = np.linalg.norm(axis3)
        if axis3_norm > 1e-6: axis3 /= axis3_norm
        # Re-orthogonalize axis2 if necessary from axis1 and axis3
        axis2 = np.cross(axis3, axis1)
        return center, np.array([axis1, axis2, axis3]), np.array([length1, 0, 0])

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors_cols = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors_cols[:, sort_idx].T # Transpose to get axes as rows
    
    # Compute lengths based on point cloud extents along these axes
    projected_coords = centered_points @ axes.T
    lengths = np.ptp(projected_coords, axis=0)

    # Ensure axes are normalized
    for i in range(3):
        norm = np.linalg.norm(axes[i])
        if norm > 1e-6:
            axes[i] /= norm
        else: # If an axis is zero (e.g. flat data), set a default
            default_axes = np.identity(3)
            axes[i] = default_axes[i]

    # Sort axes by length (descending)
    sort_idx = np.argsort(lengths)[::-1]
    axes = axes[sort_idx]
    lengths = lengths[sort_idx]
    
    # Ensure right-handed coordinate system
    if np.dot(np.cross(axes[0], axes[1]), axes[2]) < 0:
        axes[2] = -axes[2] # Flip the smallest axis to maintain right-handedness
    
    return center, axes, lengths

class PyVistaPlotterWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.plotter = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.plotter.interactor)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

class LoopStructuralMiniGui(QMainWindow):
    # Signal to indicate that the PyVista window should be updated
    model_updated_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoopStructural Mini GUI with PyVista")
        self.setGeometry(100, 100, 1400, 900) # Wider for plotter and table

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_h_layout = QHBoxLayout(self.central_widget)
        
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.main_h_layout.addWidget(self.controls_widget, 1) 

        self.pyvista_widget = PyVistaPlotterWidget(self)
        self.main_h_layout.addWidget(self.pyvista_widget, 2) 

        self.model_data_df = None
        self.geological_model = None
        self.strati_polydata = None # Stratigraphy remains singular for now
        self.faults_data = {}
        self.current_fault_name = None
        self.fault_relationships = {} # To store {(fault1, fault2): type}

        self.double_validator = QDoubleValidator() # For float inputs

        self._create_input_widgets()
        self._create_control_buttons()
        self._create_status_area()
        
        self.controls_layout.addStretch(1) # Push status area to bottom

        self.model_updated_signal.connect(self.visualize_model_pyvista)

        self.log_status("GUI Initialized. Please load data and set parameters.")

    def _create_input_widgets(self):
        # Main grid for parameters
        param_grid_layout = QGridLayout()
        param_grid_layout.setSpacing(10)
        current_row = 0

        # Fault List
        param_grid_layout.addWidget(QLabel("<b>Loaded Faults:</b>"), current_row, 0, 1, 2)
        self.fault_list_widget = QListWidget()
        self.fault_list_widget.setFixedHeight(100)
        self.fault_list_widget.currentItemChanged.connect(self.on_fault_selection_changed)
        self.fault_list_widget.setContextMenuPolicy(Qt.CustomContextMenu) # Enable context menu
        self.fault_list_widget.customContextMenuRequested.connect(self._show_fault_list_context_menu) # Connect handler
        param_grid_layout.addWidget(self.fault_list_widget, current_row + 1, 0, 1, 2); current_row += 2
        
        # Origin
        param_grid_layout.addWidget(QLabel("<b>Model Origin (X,Y,Z):</b>"), current_row, 0, 1, 4); current_row += 1
        self.origin_x_edit = QLineEdit("0"); self.origin_x_edit.setValidator(self.double_validator)
        self.origin_y_edit = QLineEdit("0"); self.origin_y_edit.setValidator(self.double_validator)
        self.origin_z_edit = QLineEdit("0"); self.origin_z_edit.setValidator(self.double_validator)
        origin_layout = QHBoxLayout()
        origin_layout.addWidget(QLabel("X:")); origin_layout.addWidget(self.origin_x_edit)
        origin_layout.addWidget(QLabel("Y:")); origin_layout.addWidget(self.origin_y_edit)
        origin_layout.addWidget(QLabel("Z:")); origin_layout.addWidget(self.origin_z_edit)
        param_grid_layout.addLayout(origin_layout, current_row, 0, 1, 4); current_row += 1
        
        # Extent / Maximum
        param_grid_layout.addWidget(QLabel("<b>Model Maximum (X,Y,Z):</b>"), current_row, 0, 1, 4); current_row += 1
        self.max_x_edit = QLineEdit("1000"); self.max_x_edit.setValidator(self.double_validator)
        self.max_y_edit = QLineEdit("1000"); self.max_y_edit.setValidator(self.double_validator)
        self.max_z_edit = QLineEdit("1000"); self.max_z_edit.setValidator(self.double_validator)
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("X:")); max_layout.addWidget(self.max_x_edit)
        max_layout.addWidget(QLabel("Y:")); max_layout.addWidget(self.max_y_edit)
        max_layout.addWidget(QLabel("Z:")); max_layout.addWidget(self.max_z_edit)
        param_grid_layout.addLayout(max_layout, current_row, 0, 1, 4); current_row += 1

        # Fault Parameters
        param_grid_layout.addWidget(QLabel("<b>--- Fault Parameters ---</b>"), current_row, 0, 1, 4, Qt.AlignCenter); current_row += 1
        
        override_layout = QHBoxLayout() # Layout for checkbox and reset button
        self.override_fault_geom_checkbox = QCheckBox("Override Calculated Fault Geometry")
        self.override_fault_geom_checkbox.setChecked(False)
        self.override_fault_geom_checkbox.stateChanged.connect(self.toggle_fault_geom_fields)
        override_layout.addWidget(self.override_fault_geom_checkbox)

        self.reset_geom_button = QPushButton("Reset Geometry to Calculated")
        self.reset_geom_button.clicked.connect(self._on_reset_geometry_clicked)
        self.reset_geom_button.setEnabled(False) # Initially disabled
        override_layout.addWidget(self.reset_geom_button)
        param_grid_layout.addLayout(override_layout, current_row, 0, 1, 4); current_row += 1

        param_grid_layout.addWidget(QLabel("Fault Center (X,Y,Z):"), current_row, 0, 1, 1)
        self.fault_center_x_edit = QLineEdit(); self.fault_center_x_edit.setReadOnly(True); self.fault_center_x_edit.setValidator(self.double_validator)
        self.fault_center_y_edit = QLineEdit(); self.fault_center_y_edit.setReadOnly(True); self.fault_center_y_edit.setValidator(self.double_validator)
        self.fault_center_z_edit = QLineEdit(); self.fault_center_z_edit.setReadOnly(True); self.fault_center_z_edit.setValidator(self.double_validator)
        self.fault_center_x_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        self.fault_center_y_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        self.fault_center_z_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        fault_center_layout = QHBoxLayout()
        fault_center_layout.addWidget(QLabel("X:")); fault_center_layout.addWidget(self.fault_center_x_edit)
        fault_center_layout.addWidget(QLabel("Y:")); fault_center_layout.addWidget(self.fault_center_y_edit)
        fault_center_layout.addWidget(QLabel("Z:")); fault_center_layout.addWidget(self.fault_center_z_edit)
        param_grid_layout.addLayout(fault_center_layout, current_row, 1, 1, 3); current_row += 1
        
        param_grid_layout.addWidget(QLabel("Major Axis Length:"), current_row, 0)
        self.fault_major_axis_edit = QLineEdit(); self.fault_major_axis_edit.setReadOnly(True); self.fault_major_axis_edit.setValidator(self.double_validator)
        self.fault_major_axis_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        param_grid_layout.addWidget(self.fault_major_axis_edit, current_row, 1)
        param_grid_layout.addWidget(QLabel("Minor Axis Length:"), current_row, 2)
        self.fault_minor_axis_edit = QLineEdit(); self.fault_minor_axis_edit.setReadOnly(True); self.fault_minor_axis_edit.setValidator(self.double_validator)
        self.fault_minor_axis_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        param_grid_layout.addWidget(self.fault_minor_axis_edit, current_row, 3); current_row += 1
        
        param_grid_layout.addWidget(QLabel("Intermediate Axis Length:"), current_row, 0)
        self.fault_intermediate_axis_edit = QLineEdit("10"); self.fault_intermediate_axis_edit.setReadOnly(True); self.fault_intermediate_axis_edit.setValidator(self.double_validator)
        self.fault_intermediate_axis_edit.editingFinished.connect(self.on_manual_fault_geom_changed)
        param_grid_layout.addWidget(self.fault_intermediate_axis_edit, current_row, 1); current_row += 1

        param_grid_layout.addWidget(QLabel("Displacement:"), current_row, 0)
        self.fault_displacement_edit = QLineEdit("1"); self.fault_displacement_edit.setValidator(self.double_validator)
        self.fault_displacement_edit.editingFinished.connect(self.on_fault_parameter_changed)
        param_grid_layout.addWidget(self.fault_displacement_edit, current_row, 1)
        param_grid_layout.addWidget(QLabel("Nelements:"), current_row, 2)
        self.fault_nelements_edit = QLineEdit("1000"); self.fault_nelements_edit.setValidator(QDoubleValidator(0, 1000000, 0))
        self.fault_nelements_edit.editingFinished.connect(self.on_fault_parameter_changed)
        param_grid_layout.addWidget(self.fault_nelements_edit, current_row, 3); current_row +=1

        param_grid_layout.addWidget(QLabel("Interpolator Type:"), current_row, 0)
        self.fault_interpolator_combo = QComboBox(); self.fault_interpolator_combo.addItems(["PLI", "FDI", "Surfe"])
        self.fault_interpolator_combo.currentTextChanged.connect(self.on_fault_parameter_changed)
        param_grid_layout.addWidget(self.fault_interpolator_combo, current_row, 1)
        param_grid_layout.addWidget(QLabel("Fault Buffer:"), current_row, 2)
        self.fault_buffer_edit = QLineEdit("0.5"); self.fault_buffer_edit.setValidator(self.double_validator)
        self.fault_buffer_edit.editingFinished.connect(self.on_fault_parameter_changed)
        param_grid_layout.addWidget(self.fault_buffer_edit, current_row, 3); current_row +=1
        
        # Add Slip Vector section header
        param_grid_layout.addWidget(QLabel("<b>--- Slip Vector ---</b>"), current_row, 0, 1, 4, Qt.AlignCenter); current_row += 1
        
        # Rake input (angle from intermediate axis on fault plane)
        param_grid_layout.addWidget(QLabel("Rake (°):"), current_row, 0)
        self.fault_rake_edit = QLineEdit("0")  # Default 0 degrees (along intermediate axis)
        self.fault_rake_edit.setValidator(QIntValidator(-180, 180))  # Integer between -180 to 180 degrees
        self.fault_rake_edit.editingFinished.connect(self.on_slip_vector_changed)
        param_grid_layout.addWidget(self.fault_rake_edit, current_row, 1)
        
        # Add explanation label
        rake_info_label = QLabel("(Angle from strike direction on fault plane)")
        rake_info_label.setStyleSheet("font-size: 9px; color: gray;")
        param_grid_layout.addWidget(rake_info_label, current_row, 2, 1, 2); current_row += 1
        
        # Add rake conventions info
        rake_conventions_label = QLabel("0°=Sinistral, 90°=Reverse, -90°=Normal, ±180°=Dextral")
        rake_conventions_label.setStyleSheet("font-size: 8px; color: darkblue; font-weight: bold;")
        param_grid_layout.addWidget(rake_conventions_label, current_row, 0, 1, 4); current_row += 1
        
        
        
        # Axis Visualization Type
        param_grid_layout.addWidget(QLabel("Axis Visualization:"), current_row, 0)
        self.axis_visualization_combo = QComboBox()
        self.axis_visualization_combo.addItems(["OBB (Wireframe)", "2D Ellipsoid", "3D Ellipsoid"])
        self.axis_visualization_combo.setCurrentText("OBB (Wireframe)")  # Default to current implementation
        self.axis_visualization_combo.currentTextChanged.connect(self.on_axis_visualization_changed)
        param_grid_layout.addWidget(self.axis_visualization_combo, current_row, 1, 1, 3); current_row += 1
        
        # Foliation Parameters
        param_grid_layout.addWidget(QLabel("<b>--- Foliation Parameters ('strati') ---</b>"), current_row, 0, 1, 4, Qt.AlignCenter); current_row +=1
        param_grid_layout.addWidget(QLabel("Nelements:"), current_row, 0)
        self.foliation_nelements_edit = QLineEdit("1000"); self.foliation_nelements_edit.setValidator(QDoubleValidator(0,1000000,0))
        param_grid_layout.addWidget(self.foliation_nelements_edit, current_row, 1)
        param_grid_layout.addWidget(QLabel("Interpolator Type:"), current_row, 2)
        self.foliation_interpolator_combo = QComboBox(); self.foliation_interpolator_combo.addItems(["PLI", "FDI", "Surfe"])
        param_grid_layout.addWidget(self.foliation_interpolator_combo, current_row, 3); current_row +=1

        self.controls_layout.addLayout(param_grid_layout) # Add the grid of parameters

        # Fault Relationships Table
        self.relationship_group_box = QGroupBox("Fault Relationships")
        self.relationship_layout = QVBoxLayout()
        self.relationship_table = QTableWidget()
        self.relationship_table.setMinimumHeight(150) # Give it some initial height
        self.relationship_layout.addWidget(self.relationship_table)
        self.relationship_group_box.setLayout(self.relationship_layout)
        self.controls_layout.addWidget(self.relationship_group_box)
        self._update_relationship_table() # Initial call to set up empty table

    def _show_fault_list_context_menu(self, position: QPoint):
        item = self.fault_list_widget.itemAt(position)
        if not item:
            return

        fault_name = item.text()
        menu = QMenu()
        zoom_action = QAction(f"Zoom to Fault: {fault_name}", self)
        zoom_action.triggered.connect(lambda: self._zoom_to_selected_fault(fault_name))
        menu.addAction(zoom_action)
        menu.exec(self.fault_list_widget.mapToGlobal(position))

    def _zoom_to_selected_fault(self, fault_name):
        if fault_name not in self.faults_data:
            self.log_status(f"Cannot zoom: Fault data for '{fault_name}' not found.")
            return

        fault_data = self.faults_data[fault_name]
        plotter = self.pyvista_widget.plotter

        center = fault_data.get('center')
        axes_vectors = fault_data.get('axes')
        lengths = fault_data.get('lengths')

        if center is None or axes_vectors is None or lengths is None:
            self.log_status(f"Geometric data missing for fault '{fault_name}'. Cannot calculate zoom bounds.")
            # Fallback to polydata bounds if geometry is incomplete
            if fault_data.get('polydata'):
                plotter.reset_camera(bounds=fault_data['polydata'].bounds)
                self.log_status(f"Zoomed to polydata bounds of fault '{fault_name}'.")
            return

        points_for_bounds = [center.copy()]
        for i in range(3):
            vec = axes_vectors[i]
            length = lengths[i]
            # Ensure length is positive for visualization bounds
            vis_length = max(length, np.mean(lengths) * 0.01 if np.mean(lengths) > 0 else 0.1) 
            if vis_length <=0 : vis_length = 0.1 # ensure positive
            
            p_a = center - vec * (vis_length / 2.0)
            p_b = center + vec * (vis_length / 2.0)
            points_for_bounds.append(p_a)
            points_for_bounds.append(p_b)
        
        bounds_mesh = pv.PolyData(np.array(points_for_bounds))
        if bounds_mesh.n_points > 0:
            plotter.reset_camera(bounds=bounds_mesh.bounds)
            self.log_status(f"Zoomed to fault '{fault_name}'.")
        else:
            self.log_status(f"Could not determine valid bounds for fault '{fault_name}' to zoom.")

    def _on_reset_geometry_clicked(self):
        if not self.current_fault_name or not self.override_fault_geom_checkbox.isChecked():
            self.log_status("Select a fault and enable 'Override Calculated Fault Geometry' to reset.")
            return

        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return

        if 'initial_center' not in fault_data or 'initial_lengths' not in fault_data:
            self.log_status(f"No initial calculated geometry stored for fault '{self.current_fault_name}'. Cannot reset.")
            return

        # Restore from initial_... values
        fault_data['center'] = fault_data['initial_center'].copy()
        fault_data['lengths'] = fault_data['initial_lengths'].copy()
        # Axes (orientation) are from initial_axes and not changed by this override system for now
        # If axes were also overridable, we'd reset fault_data['axes'] = fault_data['initial_axes'].copy()

        # Update GUI string representations from the now reset numerical values
        fault_data['gui_center_x'] = f"{fault_data['center'][0]:.2f}"
        fault_data['gui_center_y'] = f"{fault_data['center'][1]:.2f}"
        fault_data['gui_center_z'] = f"{fault_data['center'][2]:.2f}"
        fault_data['gui_major_axis'] = f"{fault_data['lengths'][0]:.2f}"
        fault_data['gui_intermediate_axis'] = f"{fault_data['lengths'][1]:.2f}"
        fault_data['gui_minor_axis'] = f"{fault_data['lengths'][2]:.2f}"
        
        self.update_fault_gui_fields(self.current_fault_name) # Refresh QLineEdits
        self.log_status(f"Geometry for fault '{self.current_fault_name}' reset to calculated values.")
        
        # Trigger visualization update (similar to on_manual_fault_geom_changed)
        self.pyvista_widget.plotter.clear_actors()
        for name, data in self.faults_data.items():
            if data.get('polydata'):
                style_args = {'style':'wireframe', 'color':'darkgrey', 'line_width':2, 'name':f"fault_input_{name}"}
                if data['polydata'].n_points <= 10:
                    style_args = {'style':'points', 'color':'magenta', 'point_size':15, 'name':f"fault_input_points_few_{name}"}
                    self.pyvista_widget.plotter.add_mesh(data['polydata'].outline(), color='cyan', name=f"fault_input_bbox_{name}")
                self.pyvista_widget.plotter.add_mesh(data['polydata'], **style_args)
        if self.strati_polydata:
            self.pyvista_widget.plotter.add_mesh(self.strati_polydata, style='points', color='blue', point_size=5, name="strati_input_vtk_points")
        self._add_fault_axes_widget(self.pyvista_widget.plotter)
        self.pyvista_widget.plotter.render()

    def on_fault_selection_changed(self, current_item, previous_item):
        if current_item is not None:
            self.current_fault_name = current_item.text()
            self.log_status(f"Selected fault: {self.current_fault_name}")
            self.update_fault_gui_fields(self.current_fault_name) # This will also update reset_geom_button state
            # Visualization update for selection (highlighting, etc.)
            if self.faults_data.get(self.current_fault_name, {}).get('polydata'):
                self.pyvista_widget.plotter.clear_actors() 
                for name, data in self.faults_data.items():
                    if data.get('polydata'):
                        style_args = {'style':'wireframe', 'color':'darkgrey', 'line_width':2, 'name':f"fault_input_{name}"}
                        if data['polydata'].n_points <= 10:
                            style_args = {'style':'points', 'color':'magenta', 'point_size':15, 'name':f"fault_input_points_few_{name}"}
                            self.pyvista_widget.plotter.add_mesh(data['polydata'].outline(), color='cyan', name=f"fault_input_bbox_{name}")
                        self.pyvista_widget.plotter.add_mesh(data['polydata'], **style_args)
                if self.strati_polydata:
                    self.pyvista_widget.plotter.add_mesh(self.strati_polydata, style='points', color='blue', point_size=5, name="strati_input_vtk_points")
                self._add_fault_axes_widget(self.pyvista_widget.plotter) # This will draw axes for current fault differently if logic is there
                self.pyvista_widget.plotter.reset_camera()
        else:
            self.current_fault_name = None
            self.update_fault_gui_fields(None) # Clear fields and disable reset button

    def update_fault_gui_fields(self, fault_name):
        fault_data = self.faults_data.get(fault_name)
        if fault_data:
            # Geometric params
            self.fault_center_x_edit.setText(fault_data.get('gui_center_x', ""))
            self.fault_center_y_edit.setText(fault_data.get('gui_center_y', ""))
            self.fault_center_z_edit.setText(fault_data.get('gui_center_z', ""))
            self.fault_major_axis_edit.setText(fault_data.get('gui_major_axis', ""))
            self.fault_intermediate_axis_edit.setText(fault_data.get('gui_intermediate_axis', "0")) # Default if missing
            self.fault_minor_axis_edit.setText(fault_data.get('gui_minor_axis', ""))
            
            # Non-geometric fault parameters
            self.fault_displacement_edit.setText(fault_data.get('gui_displacement', "1"))
            self.fault_nelements_edit.setText(fault_data.get('gui_nelements', "1000"))
            self.fault_interpolator_combo.setCurrentText(fault_data.get('gui_interpolator_type', "PLI"))
            self.fault_buffer_edit.setText(fault_data.get('gui_fault_buffer', "0.5"))

            # Update slip vector fields
            self.fault_rake_edit.setText(fault_data.get('gui_rake', "0"))

            # Update axis visualization setting
            axis_visualization = fault_data.get('gui_axis_visualization', "OBB (Wireframe)")
            self.axis_visualization_combo.setCurrentText(axis_visualization)

            is_overriding = self.override_fault_geom_checkbox.isChecked()
            self.fault_center_x_edit.setReadOnly(not is_overriding)
            self.fault_center_y_edit.setReadOnly(not is_overriding)
            self.fault_center_z_edit.setReadOnly(not is_overriding)
            self.fault_major_axis_edit.setReadOnly(not is_overriding)
            self.fault_intermediate_axis_edit.setReadOnly(not is_overriding)
            self.fault_minor_axis_edit.setReadOnly(not is_overriding)
            self.reset_geom_button.setEnabled(is_overriding) # Enable/disable reset button

        else: # No fault selected or fault_data not found
            self.fault_center_x_edit.clear(); self.fault_center_y_edit.clear(); self.fault_center_z_edit.clear()
            self.fault_major_axis_edit.clear(); self.fault_minor_axis_edit.clear(); self.fault_intermediate_axis_edit.clear()
            # Clear other fault-specific fields
            self.fault_displacement_edit.setText("1")
            self.fault_nelements_edit.setText("1000")
            self.fault_interpolator_combo.setCurrentText("PLI")
            self.fault_buffer_edit.setText("0.5")
            self.reset_geom_button.setEnabled(False)

            # Clear slip vector fields with defaults
            self.fault_rake_edit.setText("0")

            # Reset axis visualization to default
            self.axis_visualization_combo.setCurrentText("OBB (Wireframe)")

    def toggle_fault_geom_fields(self, state):
        is_editable = (state == Qt.Checked)
        self.reset_geom_button.setEnabled(is_editable and self.current_fault_name is not None) # Also check if a fault is selected

        if self.current_fault_name:
            self.update_fault_gui_fields(self.current_fault_name) # This will set read-only based on checkbox
        else: # No fault selected, just toggle general state
            self.fault_center_x_edit.setReadOnly(not is_editable)
            self.fault_center_y_edit.setReadOnly(not is_editable)
            self.fault_center_z_edit.setReadOnly(not is_editable)
            self.fault_major_axis_edit.setReadOnly(not is_editable)
            self.fault_intermediate_axis_edit.setReadOnly(not is_editable)
            self.fault_minor_axis_edit.setReadOnly(not is_editable)
    
    def on_manual_fault_geom_changed(self):
        if not self.override_fault_geom_checkbox.isChecked() or not self.current_fault_name:
            return

        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return

        try:
            # Update GUI string representations first
            fault_data['gui_center_x'] = self.fault_center_x_edit.text()
            fault_data['gui_center_y'] = self.fault_center_y_edit.text()
            fault_data['gui_center_z'] = self.fault_center_z_edit.text()
            fault_data['gui_major_axis'] = self.fault_major_axis_edit.text()
            fault_data['gui_intermediate_axis'] = self.fault_intermediate_axis_edit.text()
            fault_data['gui_minor_axis'] = self.fault_minor_axis_edit.text()

            # Update numerical data for visualization
            new_center_x = float(self.fault_center_x_edit.text())
            new_center_y = float(self.fault_center_y_edit.text())
            new_center_z = float(self.fault_center_z_edit.text())
            fault_data['center'] = np.array([new_center_x, new_center_y, new_center_z])

            new_major_len = float(self.fault_major_axis_edit.text())
            new_inter_len = float(self.fault_intermediate_axis_edit.text())
            new_minor_len = float(self.fault_minor_axis_edit.text())
            
            # Ensure lengths are in the correct order (Major, Intermediate, Minor)
            # The 'axes' vectors are assumed to be already sorted this way from calculation.
            # If lengths were re-ordered by user, this might not perfectly match original axes meaning,
            # but for visualization, we apply lengths as given.
            fault_data['lengths'] = np.array([new_major_len, new_inter_len, new_minor_len])

            self.log_status(f"Fault '{self.current_fault_name}' geometry overridden by GUI values.")
            
            # Refresh visualization
            # Clear previous input data display before redrawing axes and points
            self.pyvista_widget.plotter.clear_actors() # More specific than clear() if other things are on plotter

            # Re-add all fault meshes (dots/wireframes)
            for name, data in self.faults_data.items():
                if data.get('polydata'):
                    style_args = {'style':'wireframe', 'color':'darkgrey', 'line_width':2, 'name':f"fault_input_{name}"}
                    if data['polydata'].n_points <= 10:
                        style_args = {'style':'points', 'color':'magenta', 'point_size':15, 'name':f"fault_input_points_few_{name}"}
                        self.pyvista_widget.plotter.add_mesh(data['polydata'].outline(), color='cyan', name=f"fault_input_bbox_{name}")
                    self.pyvista_widget.plotter.add_mesh(data['polydata'], **style_args)
            
            # Re-add strati if exists
            if self.strati_polydata:
                self.pyvista_widget.plotter.add_mesh(self.strati_polydata, style='points', color='blue', point_size=5, name="strati_input_vtk_points")

            self._add_fault_axes_widget(self.pyvista_widget.plotter)
            self.pyvista_widget.plotter.render() # Update the render window immediately

        except ValueError:
            self.log_status("ERROR: Invalid number in fault geometry field during override.")
        except Exception as e:
            self.log_status(f"ERROR during manual fault geometry update: {e}")
            import traceback
            self.log_status(traceback.format_exc())

    def on_fault_parameter_changed(self):
        if not self.current_fault_name:
            return

        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return
        
        try:
            # Update string representations in fault_data from GUI
            fault_data['gui_displacement'] = self.fault_displacement_edit.text()
            fault_data['gui_nelements'] = self.fault_nelements_edit.text()
            fault_data['gui_interpolator_type'] = self.fault_interpolator_combo.currentText()
            fault_data['gui_fault_buffer'] = self.fault_buffer_edit.text()
            
            # Update slip vector parameters
            fault_data['gui_rake'] = self.fault_rake_edit.text()
            
            # Log the change. No direct visualization update is tied to these specific parameters.
            # Find out which widget sent the signal for a more specific log message (optional)
            sender = self.sender()
            param_name = "Unknown"
            if sender == self.fault_displacement_edit: param_name = "Displacement"
            elif sender == self.fault_nelements_edit: param_name = "Nelements"
            elif sender == self.fault_interpolator_combo: param_name = "Interpolator Type"
            elif sender == self.fault_buffer_edit: param_name = "Fault Buffer"
            elif sender == self.fault_rake_edit: param_name = "Rake"

            self.log_status(f"Fault '{self.current_fault_name}' parameter '{param_name}' updated in GUI.")

        except Exception as e:
            self.log_status(f"ERROR during fault parameter update: {e}")
            import traceback
            self.log_status(traceback.format_exc())

    def _create_control_buttons(self):
        data_load_layout = QHBoxLayout()
        self.load_fault_button = QPushButton("Load Fault VTK")
        self.load_fault_button.clicked.connect(self.load_fault_vtk)
        data_load_layout.addWidget(self.load_fault_button)
        self.load_strati_button = QPushButton("Load Strati VTK")
        self.load_strati_button.clicked.connect(self.load_strati_vtk)
        data_load_layout.addWidget(self.load_strati_button)
        self.controls_layout.addLayout(data_load_layout)

        action_button_layout = QHBoxLayout()
        self.create_model_button = QPushButton("Build Model & Features")
        self.create_model_button.clicked.connect(self.create_model_and_features)
        action_button_layout.addWidget(self.create_model_button)
        # Visualize button now implicit via signal, or can be explicit too
        # self.visualize_button = QPushButton("Visualize (PyVista)")
        # self.visualize_button.clicked.connect(self.visualize_model_pyvista)
        # action_button_layout.addWidget(self.visualize_button)
        self.controls_layout.addLayout(action_button_layout)

    def _create_status_area(self):
        self.status_text_edit = QTextEdit()
        self.status_text_edit.setReadOnly(True)
        self.status_text_edit.setFixedHeight(150)
        self.controls_layout.addWidget(QLabel("<b>Status:</b>"))
        self.controls_layout.addWidget(self.status_text_edit)

    def log_status(self, message):
        self.status_text_edit.append(message)
        QApplication.processEvents()

    def get_numpy_array_from_inputs(self, x_edit, y_edit, z_edit, name="array"):
        try: return np.array([float(x_edit.text()), float(y_edit.text()), float(z_edit.text())])
        except ValueError: self.log_status(f"ERROR: Invalid number for {name}."); return None
        
    def get_float_from_input(self, line_edit, name="value", default_val=0.0):
        try: return float(line_edit.text())
        except ValueError: self.log_status(f"ERROR: Invalid number for {name}. Using default {default_val}."); return default_val
            
    def get_int_from_input(self, line_edit, name="value", default_val=0):
        try: return int(line_edit.text())
        except ValueError: self.log_status(f"ERROR: Invalid int for {name}. Using default {default_val}."); return default_val

    def _add_fault_axes_widget(self, plotter):
        # Clear existing axes
        actors_to_remove = [actor for actor in plotter.renderer.actors.keys() if actor.startswith("FAULTAXIS_")]
        for actor_name in actors_to_remove:
            plotter.remove_actor(actor_name, render=False)

        legend_entries = []

        for fault_name, fault_data in self.faults_data.items():
            if 'center' in fault_data and 'axes' in fault_data and 'lengths' in fault_data:
                center = fault_data['center']
                axes = fault_data['axes']
                lengths = fault_data['lengths']

                colors = ['red', 'green', 'blue']
                labels = ['Major', 'Intermediate', 'Minor']

                # Get the visualization type for this fault
                axis_visualization = fault_data.get('gui_axis_visualization', "OBB (Wireframe)")

                # Create an oriented bounding box using all three axes
                major_axis = axes[0]
                intermediate_axis = axes[1]
                minor_axis = axes[2]
                
                major_length = max(lengths[0], np.mean(lengths) * 0.1)
                intermediate_length = max(lengths[1], np.mean(lengths) * 0.1)
                minor_length = max(lengths[2], np.mean(lengths) * 0.1)

                # Apply the selected visualization method
                if axis_visualization == "OBB (Wireframe)":
                    self._create_obb_visualization(plotter, fault_name, center, axes, lengths, 
                                                 major_length, intermediate_length, minor_length, legend_entries)
                elif axis_visualization == "2D Ellipsoid":
                    self._create_2d_ellipsoid_visualization(plotter, fault_name, center, axes, lengths,
                                                          major_length, intermediate_length, minor_length, legend_entries)
                elif axis_visualization == "3D Ellipsoid":
                    self._create_3d_ellipsoid_visualization(plotter, fault_name, center, axes, lengths,
                                                          major_length, intermediate_length, minor_length, legend_entries)

                # Add center marker
                sphere_radius = np.mean(lengths) * 0.02
                if sphere_radius < 0.01:
                    sphere_radius = 0.1
                plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=center),
                               color='yellow', name=f'FAULTAXIS_{fault_name}_CenterMarker')

                # Now add slip vector visualization
                try:
                    # Get slip vector parameters
                    rake = float(fault_data.get('gui_rake', 0))
                    
                    # Convert rake to 3D vector (constrained to fault plane)
                    slip_vec = self.rake_to_vector(rake, fault_name)
                    
                    # Scale the slip vector for visualization (relative to fault size)
                    arrow_length = max(lengths) * 0.5  # Half the length of the longest axis
                    
                    # Create start and end points for the arrow shaft
                    start_point = center.copy()
                    end_point = start_point + slip_vec * arrow_length
                    
                    # Color based on rake angle (geological convention)
                    if abs(rake) <= 30 or abs(rake) >= 150:  # Strike-slip
                        slip_color = 'green'
                    elif rake > 30:  # Reverse/thrust
                        slip_color = 'red'
                    else:  # Normal
                        slip_color = 'blue'
                    
                    # Create simple 2D arrow using lines
                    # Main shaft line
                    main_line = pv.Line(start_point, end_point)
                    plotter.add_mesh(main_line, color=slip_color, line_width=4,
                                   name=f"SLIPVEC_{fault_name}_shaft")
                    
                    # Create arrowhead using two short lines
                    arrowhead_length = arrow_length * 0.15  # 15% of arrow length
                    arrowhead_angle = 25  # degrees
                    
                    # Calculate arrowhead directions (in the fault plane)
                    import math
                    
                    # Get two perpendicular vectors in the fault plane for arrowhead
                    # Use major and intermediate axes as basis
                    major_axis = axes[0]
                    intermediate_axis = axes[1]
                    
                    # Create arrowhead vectors by rotating the slip vector slightly
                    angle_rad = math.radians(arrowhead_angle)
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    
                    # Project slip vector onto fault plane basis
                    slip_major = np.dot(slip_vec, major_axis)
                    slip_inter = np.dot(slip_vec, intermediate_axis)
                    
                    # Create rotated vectors for arrowhead
                    arrow1_major = slip_major * cos_a - slip_inter * sin_a
                    arrow1_inter = slip_major * sin_a + slip_inter * cos_a
                    arrow2_major = slip_major * cos_a + slip_inter * sin_a  
                    arrow2_inter = -slip_major * sin_a + slip_inter * cos_a
                    
                    # Convert back to 3D
                    arrow1_vec = arrow1_major * major_axis + arrow1_inter * intermediate_axis
                    arrow2_vec = arrow2_major * major_axis + arrow2_inter * intermediate_axis
                    
                    # Normalize and scale
                    arrow1_vec = arrow1_vec / np.linalg.norm(arrow1_vec) * arrowhead_length
                    arrow2_vec = arrow2_vec / np.linalg.norm(arrow2_vec) * arrowhead_length
                    
                    # Create arrowhead lines (pointing back from tip)
                    arrowhead1_start = end_point - arrow1_vec
                    arrowhead2_start = end_point - arrow2_vec
                    
                    arrowhead1 = pv.Line(end_point, arrowhead1_start)
                    arrowhead2 = pv.Line(end_point, arrowhead2_start)
                    
                    plotter.add_mesh(arrowhead1, color=slip_color, line_width=4,
                                   name=f"SLIPVEC_{fault_name}_head1")
                    plotter.add_mesh(arrowhead2, color=slip_color, line_width=4,
                                   name=f"SLIPVEC_{fault_name}_head2")
                    
                    # Add to legend if this is the current fault
                    if fault_name == self.current_fault_name or len(self.faults_data) == 1:
                        # Determine fault type from rake
                        if abs(rake) <= 30 or abs(rake) >= 150:
                            fault_type = "Strike-slip"
                        elif rake > 30:
                            fault_type = "Reverse"
                        else:
                            fault_type = "Normal"
                        label_text = f"{fault_name} {fault_type} (rake={rake}°)"
                        legend_entries.append((label_text, slip_color))
                    
                except Exception as e:
                    self.log_status(f"Error drawing slip vector: {str(e)}")

        if legend_entries:
            plotter.add_legend(labels=legend_entries, bcolor='white', face='triangle')

    def _create_obb_visualization(self, plotter, fault_name, center, axes, lengths, 
                                major_length, intermediate_length, minor_length, legend_entries):
        """Create Oriented Bounding Box visualization"""
        try:
            # Create a unit box centered at origin
            bbox = pv.Box(bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
            
            # Create transformation matrix from the principal axes
            # Scale the box to match the axis lengths
            scale_matrix = np.diag([major_length, intermediate_length, minor_length])
            
            # Create rotation matrix from the principal axes (each axis is a column)
            rotation_matrix = np.column_stack([axes[0], axes[1], axes[2]])
            
            # Apply scaling then rotation to each point
            points = bbox.points
            transformed_points = []
            
            for point in points:
                # Scale the point
                scaled_point = scale_matrix @ point
                # Rotate the scaled point
                rotated_point = rotation_matrix @ scaled_point
                # Translate to the fault center
                final_point = rotated_point + center
                transformed_points.append(final_point)
            
            # Create the transformed box
            oriented_bbox = bbox.copy()
            oriented_bbox.points = np.array(transformed_points)
            
            # Add the oriented bounding box as wireframe
            plotter.add_mesh(oriented_bbox, style='wireframe', color='orange', 
                           line_width=3, name=f"FAULTAXIS_{fault_name}_OBB")
            
            # Also add a semi-transparent version for better visualization
            plotter.add_mesh(oriented_bbox, color='orange', opacity=0.2,
                           name=f"FAULTAXIS_{fault_name}_OBB_Fill")
            
            if fault_name == self.current_fault_name or len(self.faults_data) == 1:
                legend_entries.append((f"{fault_name} OBB", 'orange'))
                
        except Exception as e:
            self.log_status(f"Error creating OBB for {fault_name}: {e}")
            self._create_fallback_lines(plotter, fault_name, center, axes, lengths, legend_entries)

    def _create_2d_ellipsoid_visualization(self, plotter, fault_name, center, axes, lengths,
                                         major_length, intermediate_length, minor_length, legend_entries):
        """Create 2D ellipse in the major-intermediate plane with minor axis line"""
        try:
            # Create a parametric ellipse in 2D, then transform to 3D
            theta = np.linspace(0, 2*np.pi, 50)
            ellipse_2d = np.column_stack([
                (major_length/2) * np.cos(theta),
                (intermediate_length/2) * np.sin(theta),
                np.zeros_like(theta)
            ])
            
            # Create transformation matrix from axes
            transform_matrix = np.column_stack([axes[0], axes[1], axes[2]])
            
            # Transform ellipse points to 3D using the fault axes
            ellipse_3d = ellipse_2d @ transform_matrix.T + center
            
            # Create a polygon from the ellipse points
            ellipse_polydata = pv.PolyData(ellipse_3d)
            faces = []
            n_points = len(ellipse_3d)
            for i in range(n_points):
                faces.extend([3, i, (i+1) % n_points, 0])  # Triangle fan from center
            
            # Add center point
            ellipse_points = np.vstack([center, ellipse_3d])
            ellipse_polydata = pv.PolyData(ellipse_points, faces)
            
            # Add the ellipse to the plotter
            plotter.add_mesh(ellipse_polydata, color='orange', opacity=0.6, 
                           name=f"FAULTAXIS_{fault_name}_Ellipse2D")
            
            # Draw line for minor axis (perpendicular to ellipse)
            vis_length = max(minor_length, np.mean(lengths) * 0.1)
            p_a = center - axes[2] * (vis_length/2)
            p_b = center + axes[2] * (vis_length/2)
            plotter.add_mesh(pv.Line(p_a, p_b), color='blue', line_width=3, 
                           name=f"FAULTAXIS_{fault_name}_MinorAxis")
            
            if fault_name == self.current_fault_name or len(self.faults_data) == 1:
                legend_entries.append((f"{fault_name} 2D Ellipse", 'orange'))
                legend_entries.append((f"{fault_name} Minor", 'blue'))
                
        except Exception as e:
            self.log_status(f"Error creating 2D ellipse for {fault_name}: {e}")
            self._create_fallback_lines(plotter, fault_name, center, axes, lengths, legend_entries)

    def _create_3d_ellipsoid_visualization(self, plotter, fault_name, center, axes, lengths,
                                         major_length, intermediate_length, minor_length, legend_entries):
        """Create true 3D ellipsoid using all three axes"""
        try:
            # Create a parametric ellipsoid
            ellipsoid = pv.ParametricEllipsoid(xradius=major_length/2, 
                                             yradius=intermediate_length/2, 
                                             zradius=minor_length/2)
            
            # Create rotation matrix from the principal axes
            rotation_matrix = np.column_stack([axes[0], axes[1], axes[2]])
            
            # Apply rotation to ellipsoid points
            points = ellipsoid.points
            rotated_points = []
            
            for point in points:
                rotated_point = rotation_matrix @ point
                final_point = rotated_point + center
                rotated_points.append(final_point)
            
            # Create the transformed ellipsoid
            oriented_ellipsoid = ellipsoid.copy()
            oriented_ellipsoid.points = np.array(rotated_points)
            
            # Add the ellipsoid to the plotter
            plotter.add_mesh(oriented_ellipsoid, color='orange', opacity=0.7,
                           name=f"FAULTAXIS_{fault_name}_Ellipsoid3D")
            
            if fault_name == self.current_fault_name or len(self.faults_data) == 1:
                legend_entries.append((f"{fault_name} 3D Ellipsoid", 'orange'))
                
        except Exception as e:
            self.log_status(f"Error creating 3D ellipsoid for {fault_name}: {e}")
            self._create_fallback_lines(plotter, fault_name, center, axes, lengths, legend_entries)

    def _create_fallback_lines(self, plotter, fault_name, center, axes, lengths, legend_entries):
        """Fallback visualization using simple lines for each axis"""
        colors = ['red', 'green', 'blue']
        labels = ['Major', 'Intermediate', 'Minor']
        
        for i, (axis, length, color, label) in enumerate(zip(axes, lengths, colors, labels)):
            vis_length = max(length, np.mean(lengths) * 0.1)
            p_a = center - axis * (vis_length/2)
            p_b = center + axis * (vis_length/2)
            plotter.add_mesh(pv.Line(p_a, p_b), color=color, line_width=3, 
                           name=f"FAULTAXIS_{fault_name}_{label}")
            
            if fault_name == self.current_fault_name or len(self.faults_data) == 1:
                legend_entries.append((f"{fault_name} {label}", color))

    def load_fault_vtk(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Fault VTK/VTP", "", "VTK Files (*.vtk *.vtp)")
        if file_path:
            default_name = file_path.split('/')[-1].split('.')[0]
            fault_name, ok = QInputDialog.getText(self, "Fault Name", "Enter a unique name for this fault:", text=default_name)
            
            if not ok or not fault_name:
                self.log_status("Fault loading cancelled or no name provided.")
                return
            
            if fault_name in self.faults_data:
                self.log_status(f"ERROR: Fault name '{fault_name}' already exists. Please use a unique name.")
                return

            try:
                polydata = pv.read(file_path)
                initial_center, initial_axes, initial_lengths = compute_geometric_features(polydata)
                
                current_fault_entry = {
                    'polydata': polydata,
                    'initial_center': initial_center.copy(),
                    'initial_axes': initial_axes.copy(), 
                    'initial_lengths': initial_lengths.copy(),
                    'center': initial_center.copy(),
                    'axes': initial_axes.copy(), 
                    'lengths': initial_lengths.copy(),
                    'gui_center_x': f"{initial_center[0]:.2f}",
                    'gui_center_y': f"{initial_center[1]:.2f}",
                    'gui_center_z': f"{initial_center[2]:.2f}",
                    'gui_major_axis': f"{initial_lengths[0]:.2f}",
                    'gui_intermediate_axis': f"{initial_lengths[1]:.2f}",
                    'gui_minor_axis': f"{initial_lengths[2]:.2f}",
                    'gui_displacement': "1", 
                    'gui_nelements': "1000",
                    'gui_interpolator_type': "PLI", 
                    'gui_fault_buffer': "0.5",
                    # Add slip vector defaults
                    'gui_rake': "0",
                    # Add axis visualization default
                    'gui_axis_visualization': "OBB (Wireframe)"
                }
                self.faults_data[fault_name] = current_fault_entry
                # self.current_fault_name = fault_name # Set by on_fault_selection_changed via setCurrentRow
                
                self.log_status(f"Fault data '{fault_name}' loaded: {file_path}. Points: {polydata.n_points}")
                self.log_status(f"Geometric analysis for '{fault_name}':")
                self.log_status(f"  Initial Center: {initial_center}")
                self.log_status(f"  Initial Lengths (Maj,Int,Min): {initial_lengths}")

                self.fault_list_widget.addItem(fault_name)
                self.fault_list_widget.setCurrentRow(self.fault_list_widget.count() - 1) # Triggers selection change
                
                self._update_relationship_table()

                # Visualization is handled by on_fault_selection_changed
            except Exception as e:
                self.log_status(f"ERROR loading fault VTK for '{fault_name}': {e}")
                if fault_name in self.faults_data: del self.faults_data[fault_name]
                items = self.fault_list_widget.findItems(fault_name, Qt.MatchExactly)
                if items: self.fault_list_widget.takeItem(self.fault_list_widget.row(items[0]))
                # if self.current_fault_name == fault_name: self.current_fault_name = None # Handled by selection
                self._update_relationship_table()
                import traceback; self.log_status(traceback.format_exc())

    def load_strati_vtk(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Stratigraphy VTK/VTP", "", "VTK Files (*.vtk *.vtp)")
        if file_path:
            try:
                self.strati_polydata = pv.read(file_path)
                self.log_status(f"Strati data loaded: {file_path}. Points: {self.strati_polydata.n_points}")
                # Trigger PyVista update
                self.pyvista_widget.plotter.clear()
                if self.strati_polydata:
                    self.pyvista_widget.plotter.add_mesh(self.strati_polydata, style='points', color='blue', point_size=5, name="strati_input_vtk_points")
                self.pyvista_widget.plotter.reset_camera()

            except Exception as e:
                self.log_status(f"ERROR loading strati VTK: {e}"); self.strati_polydata = None

    def create_model_and_features(self):
        self.log_status("Attempting to create model and features...")
        origin_vals = self.get_numpy_array_from_inputs(self.origin_x_edit, self.origin_y_edit, self.origin_z_edit, "Origin")
        maximum_vals = self.get_numpy_array_from_inputs(self.max_x_edit, self.max_y_edit, self.max_z_edit, "Maximum")

        if origin_vals is None or maximum_vals is None: return
        if np.any(maximum_vals <= origin_vals): self.log_status("ERROR: Max < Origin."); return
        if not self.faults_data: self.log_status("ERROR: No fault data loaded."); return
        if self.strati_polydata is None : self.log_status("ERROR: Strati VTK not loaded."); return

        # Construct DataFrame for LoopStructural
        all_feature_dfs = []
        for fault_name, fault_data_item in self.faults_data.items():
            if 'polydata' not in fault_data_item:
                self.log_status(f"WARNING: No polydata for fault '{fault_name}'. Skipping.")
                continue
            fault_points = fault_data_item['polydata'].points
            fault_df = pd.DataFrame(fault_points, columns=['X', 'Y', 'Z'])
            fault_df['feature_name'] = fault_name # Use the unique fault name
            fault_df['val'] = 0 
            all_feature_dfs.append(fault_df)

        # Stratigraphy data (still singular for now)
        strati_points = self.strati_polydata.points
        strati_df = pd.DataFrame(strati_points, columns=['X', 'Y', 'Z'])
        strati_df['feature_name'] = "strati" # Fixed name for now
        if 'val' in self.strati_polydata.point_data:
             strati_df['val'] = self.strati_polydata.point_data['val']
        elif 'vals' in self.strati_polydata.point_data:
             strati_df['val'] = self.strati_polydata.point_data['vals']
        else: 
             self.log_status("WARNING: No 'val' array in strati VTK. Using dummy values.")
        if 'Normals' in self.strati_polydata.point_data:
            normals = self.strati_polydata.point_data['Normals']
            strati_df['nx'] = normals[:,0]; strati_df['ny'] = normals[:,1]; strati_df['nz'] = normals[:,2]
        else:
            self.log_status("WARNING: No 'Normals' array found in strati VTK.")
        all_feature_dfs.append(strati_df)

        if not all_feature_dfs:
            self.log_status("ERROR: No valid feature data to build model."); return
        self.model_data_df = pd.concat(all_feature_dfs, ignore_index=True)
        self.log_status(f"Combined DataFrame created. Shape: {self.model_data_df.shape}")

        try:
            self.geological_model = LS.GeologicalModel(origin_vals, maximum_vals)
            self.log_status("GeologicalModel instantiated.")
            self.geological_model.data = self.model_data_df 
            self.log_status("Model data assigned.")

            # Create Faults
            created_fault_features = []
            for fault_name, fault_data_item in self.faults_data.items():
                if 'polydata' not in fault_data_item: continue # Already warned

                # Global parameters for now (Phase 1)
                fault_displacement = self.get_float_from_input(self.fault_displacement_edit, f"Fault Disp. ({fault_name})", 1.0)
                fault_nelements = self.get_int_from_input(self.fault_nelements_edit, f"Fault Nelem. ({fault_name})", 1000)
                fault_interpolator = self.fault_interpolator_combo.currentText()
                fault_buffer_val = self.get_float_from_input(self.fault_buffer_edit, f"Fault Buf. ({fault_name})", 0.5)
                
                # Get fault geometric params from stored fault_data_item
                fc_x = float(fault_data_item.get('gui_center_x', 0))
                fc_y = float(fault_data_item.get('gui_center_y', 0))
                fc_z = float(fault_data_item.get('gui_center_z', 0))
                fault_center_val = np.array([fc_x, fc_y, fc_z])

                fault_major_axis = float(fault_data_item.get('gui_major_axis', 0))
                fault_minor_axis = float(fault_data_item.get('gui_minor_axis', 0))
                fault_intermediate_axis = float(fault_data_item.get('gui_intermediate_axis', 0))

                # Get slip vector parameters
                rake = float(fault_data_item.get('gui_rake', 0))
                
                # Create slip vector dictionary
                slip_vector = {
                    'rake': rake
                }
                
                # Log the slip vector
                self.log_status(f"Using slip vector for fault '{fault_name}': rake={rake}°")
                
                fault_params_ls = {
                    'nelements': fault_nelements, 'interpolatortype': fault_interpolator,
                    'fault_buffer': fault_buffer_val, 'fault_center': fault_center_val,
                    'major_axis': fault_major_axis, 'minor_axis': fault_minor_axis,
                    'intermediate_axis': fault_intermediate_axis, 'points': True,
                    'slip_vector': slip_vector  # Add the slip vector to parameters
                }
                if fault_data_item.get('is_planar_fit') and fault_data_item.get('plane_normal') is not None:
                    fault_params_ls['fault_normal_vector'] = fault_data_item['plane_normal']
                
                self.log_status(f"Creating fault '{fault_name}' with params: {fault_params_ls}")
                self.geological_model.create_and_add_fault(fault_name, fault_displacement, **fault_params_ls)
                self.log_status(f"Fault '{fault_name}' created.")
                created_fault_features.append(self.geological_model.get_feature_by_name(fault_name))
            
            # Create Foliation (affected by all faults for now)
            if created_fault_features:
                foliation_name = "strati"
                foliation_nelements = self.get_int_from_input(self.foliation_nelements_edit, "Fol. Nelem.", 1000)
                foliation_interpolator = self.foliation_interpolator_combo.currentText()
                self.log_status(f"Creating foliation '{foliation_name}' faulted by: {[f.name for f in created_fault_features]}")
                self.geological_model.create_and_add_foliation(
                    foliation_name, nelements=foliation_nelements, 
                    interpolatortype=foliation_interpolator, 
                    faults=created_fault_features # Pass list of actual fault features
                )
                self.log_status(f"Foliation '{foliation_name}' created.")
            else:
                self.log_status("No faults created, skipping foliation for now.")

            self.log_status(f"Model features: {[f.name for f in self.geological_model.features] if self.geological_model else 'None'}")
            self.model_updated_signal.emit() 

        except Exception as e:
            self.log_status(f"ERROR during model/feature creation: {e}")
            self.geological_model = None

    def visualize_model_pyvista(self):
        self.log_status("Visualizing model with PyVista...")
        if not self.geological_model or not self.geological_model.features:
            self.log_status("ERROR: Model or features not created."); return
        
        plotter = self.pyvista_widget.plotter
        plotter.clear()
        # self._add_fault_axes_widget(plotter) # This is now called from load_fault_vtk or selection_changed, and draws all axes

        try:
            # Grid for evaluation - remains the same
            nsteps = (30, 30, 30) 
            grid_points = self.geological_model.regular_grid(nsteps=nsteps, shuffle=False, rescale=True)
            if not self.geological_model.bounding_box: 
                self.log_status("ERROR: Model bounding_box not available for grid creation."); return
            dims = nsteps
            origin_pv, max_pv = self.geological_model.origin, self.geological_model.maximum
            spacing_pv = ( (max_pv[0]-origin_pv[0])/(dims[0]-1) if dims[0]>1 else 1,
                           (max_pv[1]-origin_pv[1])/(dims[1]-1) if dims[1]>1 else 1,
                           (max_pv[2]-origin_pv[2])/(dims[2]-1) if dims[2]>1 else 1 )
            grid = pv.StructuredGrid(); grid.dimensions = dims; grid.origin = origin_pv; grid.spacing = spacing_pv
            
            feature_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
            color_idx = 0

            for feature in self.geological_model.features:
                self.log_status(f"Processing feature for visualization: {feature.name}")
                feature_values = self.geological_model.evaluate_feature_value(feature.name, grid_points, scale=False)
                grid[feature.name] = feature_values
                current_color = feature_colors[color_idx % len(feature_colors)]
                
                if feature.type == 'fault':
                    fault_surface = grid.contour(isosurfaces=[0], scalars=feature.name)
                    if fault_surface.n_points > 0:
                        plotter.add_mesh(fault_surface, color=current_color, opacity=0.7, name=f"model_surface_{feature.name}")
                        self.log_status(f"Fault surface '{feature.name}' added to PyVista with color {current_color}.")
                        color_idx +=1
                    else: self.log_status(f"Fault surface '{feature.name}' resulted in no points.")
                elif feature.type == 'foliation': # Assuming 'strati' is a foliation type
                    min_val, max_val = np.nanmin(feature_values), np.nanmax(feature_values)
                    if not (np.isnan(min_val) or np.isnan(max_val) or min_val == max_val):
                        iso_values_strati = np.linspace(min_val, max_val, 5) 
                        strati_surfaces = grid.contour(isosurfaces=iso_values_strati, scalars=feature.name)
                        if strati_surfaces.n_points > 0:
                            plotter.add_mesh(strati_surfaces, opacity=0.5, name=f"model_surface_{feature.name}", cmap="viridis") # Could use current_color too
                            self.log_status(f"Foliation '{feature.name}' isosurfaces added to PyVista.")
                            color_idx +=1
                        else: self.log_status(f"Foliation '{feature.name}' resulted in no points for isosurfacing.")    
                    else:
                        self.log_status(f"WARNING: Could not determine value range for {feature.name} for isosurfacing.")
            
            # Add original input data for all loaded faults
            for fault_name, fault_data_item in self.faults_data.items():
                if 'polydata' in fault_data_item:
                    pd_item = fault_data_item['polydata']
                    style_args_pv = {'style':'wireframe', 'color':'darkgrey', 'line_width':1, 'name':f"fault_input_{fault_name}"}
                    if pd_item.n_points <= 10: # Display small point clouds differently
                        style_args_pv = {'style':'points', 'color':'black', 'point_size':10, 'name':f"fault_input_points_few_{fault_name}"}
                        # plotter.add_mesh(pd_item.outline(), color='grey', name=f"fault_input_bbox_{fault_name}") # Outline can be noisy
                    plotter.add_mesh(pd_item, **style_args_pv)
            
            # Add original strati input data (still singular)
            if self.strati_polydata:
                plotter.add_mesh(self.strati_polydata, style='points', color='darkblue', point_size=8, name="strati_input_vtk")

            # Add fault axes (drawn by _add_fault_axes_widget, which is called from load or selection change)
            self._add_fault_axes_widget(plotter)

            plotter.camera.focal_point = self.geological_model.origin + (self.geological_model.maximum - self.geological_model.origin) / 2.0
            plotter.camera.position = plotter.camera.focal_point + np.array([0,0, -np.max(self.geological_model.maximum - self.geological_model.origin)*2.5])
            plotter.camera.viewup = [0,1,0]
            plotter.reset_camera()
            plotter.enable_parallel_scaling()
            self.log_status("PyVista visualization updated with multiple features.")

        except Exception as e:
            self.log_status(f"ERROR during PyVista visualization: {e}")
            import traceback
            self.log_status(traceback.format_exc())

    def _update_relationship_table(self):
        self.relationship_table.clear()
        fault_names = [self.fault_list_widget.item(i).text() for i in range(self.fault_list_widget.count())]
        num_faults = len(fault_names)

        self.relationship_table.setRowCount(num_faults)
        self.relationship_table.setColumnCount(num_faults)
        self.relationship_table.setHorizontalHeaderLabels(fault_names)
        self.relationship_table.setVerticalHeaderLabels(fault_names)

        for r in range(num_faults):
            for c in range(num_faults):
                if r == c:
                    item = QTableWidgetItem("N/A") # Or leave empty/disabled
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.relationship_table.setItem(r, c, item)
                    continue

                fault_r_name = fault_names[r]
                fault_c_name = fault_names[c]

                combo = QComboBox()
                combo.addItems(['none', 'abut', 'splay']) # Add other types if needed
                
                # Retrieve stored relationship, default to 'none'
                relationship_type = self.fault_relationships.get((fault_r_name, fault_c_name), 'none')
                combo.setCurrentText(relationship_type)
                
                # Use lambda with default arguments to capture r, c, and fault names correctly
                combo.currentTextChanged.connect(
                    lambda text, r_idx=r, c_idx=c, f_r=fault_r_name, f_c=fault_c_name: \
                    self._on_relationship_changed(text, r_idx, c_idx, f_r, f_c)
                )
                self.relationship_table.setCellWidget(r, c, combo)
        
        self.relationship_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.relationship_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _on_relationship_changed(self, text_value, row_idx, col_idx, fault_row_name, fault_col_name):
        relationship_key = (fault_row_name, fault_col_name)
        if text_value == 'none':
            if relationship_key in self.fault_relationships:
                del self.fault_relationships[relationship_key]
                self.log_status(f"Relationship between {fault_row_name} and {fault_col_name} set to 'none'.")
        else:
            self.fault_relationships[relationship_key] = text_value
            self.log_status(f"Relationship: {fault_row_name} affected by {fault_col_name} as '{text_value}'.")
        # print(f"Updated relationships: {self.fault_relationships}") # For debugging

    # New helper method to convert rake to 3D vector
    def rake_to_vector(self, rake, fault_name=None):
        """
        Convert rake (in degrees) to a 3D unit vector constrained to the fault plane.
        Rake: -180 to 180 degrees measured from major axis (strike direction)
        on the fault plane defined by major and intermediate axes.
        
        Geological conventions:
        - Major axis: Strike direction (horizontal on fault plane)
        - Intermediate axis: Dip direction (down-dip on fault plane)  
        - Minor axis: Fault normal (perpendicular to fault plane)
        
        Args:
            rake: Angle in degrees from strike direction (major axis)
            fault_name: Name of the fault to get axes from (uses current fault if None)
        
        Returns: 3D unit vector [x,y,z] on the fault plane
        """
        import math
        
        # Get the current fault name if not provided
        if fault_name is None:
            fault_name = self.current_fault_name
        
        if not fault_name or fault_name not in self.faults_data:
            # Fallback to simple calculation if no fault data available
            rake_rad = math.radians(rake)
            return np.array([math.cos(rake_rad), math.sin(rake_rad), 0])
        
        fault_data = self.faults_data[fault_name]
        if 'axes' not in fault_data:
            # Fallback if no axes data
            rake_rad = math.radians(rake)
            return np.array([math.cos(rake_rad), math.sin(rake_rad), 0])
        
        # Get the principal axes from PCA using correct geological conventions
        axes = fault_data['axes']
        major_axis = axes[0]        # Strike direction (horizontal on fault plane)
        intermediate_axis = axes[1] # Dip direction (down-dip on fault plane)
        minor_axis = axes[2]        # Normal to fault plane
        
        # Convert rake to radians
        rake_rad = math.radians(rake)
        
        # Calculate slip vector on the fault plane
        # Rake = 0° means along major axis (strike direction) - sinistral
        # Rake = 90° means along intermediate axis (dip direction) - reverse
        # Rake = -90° means opposite to intermediate axis (up-dip direction) - normal
        # Rake = ±180° means opposite to major axis (strike direction) - dextral
        slip_vector = (math.cos(rake_rad) * major_axis + 
                      math.sin(rake_rad) * intermediate_axis)
        
        # Ensure the vector is normalized
        norm = np.linalg.norm(slip_vector)
        if norm > 1e-6:
            slip_vector = slip_vector / norm
        else:
            # Fallback to major axis if something went wrong
            slip_vector = major_axis / np.linalg.norm(major_axis)
        
        # Verify the slip vector is on the fault plane (perpendicular to minor axis)
        # The dot product with the normal should be close to zero
        dot_product = np.dot(slip_vector, minor_axis)
        if abs(dot_product) > 1e-6:
            # Project the slip vector onto the fault plane to ensure it's on the plane
            slip_vector = slip_vector - dot_product * minor_axis
            # Renormalize
            norm = np.linalg.norm(slip_vector)
            if norm > 1e-6:
                slip_vector = slip_vector / norm
        
        return slip_vector

    # Modify the on_slip_type_changed method to update visualization
    def on_slip_type_changed(self, button):
        if not self.current_fault_name:
            return
            
        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return
            
        slip_type = "normal" if button == self.normal_radio else "reverse"
        fault_data['gui_slip_type'] = slip_type
        self.log_status(f"Slip type for fault '{self.current_fault_name}' set to {slip_type}")
        
        # Update the visualization to reflect the changed slip type
        self._update_fault_visualization()

    # Method to update visualization after slip vector changes
    def _update_fault_visualization(self):
        """Update the PyVista visualization when slip vector parameters change"""
        if not self.current_fault_name:
            return
        
        # Similar to what we do in on_manual_fault_geom_changed
        # Clear and redraw all actors to refresh the visualization
        self.pyvista_widget.plotter.clear_actors()
        for name, data in self.faults_data.items():
            if data.get('polydata'):
                style_args = {'style':'wireframe', 'color':'darkgrey', 'line_width':2, 
                              'name':f"fault_input_{name}"}
                if data['polydata'].n_points <= 10:
                    style_args = {'style':'points', 'color':'magenta', 'point_size':15, 
                                  'name':f"fault_input_points_few_{name}"}
                    self.pyvista_widget.plotter.add_mesh(
                        data['polydata'].outline(), color='cyan', name=f"fault_input_bbox_{name}")
                self.pyvista_widget.plotter.add_mesh(data['polydata'], **style_args)
        
        if self.strati_polydata:
            self.pyvista_widget.plotter.add_mesh(
                self.strati_polydata, style='points', color='blue', point_size=5, 
                name="strati_input_vtk_points")
        
        # This will redraw the axes and slip vectors with updated parameters
        self._add_fault_axes_widget(self.pyvista_widget.plotter)
        self.pyvista_widget.plotter.render()

    # New handler for slip vector parameter changes (rake)
    def on_slip_vector_changed(self):
        """Handle changes to slip vector parameters and update visualization"""
        if not self.current_fault_name:
            return
        
        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return
        
        try:
            # Update the fault data with new values from GUI
            fault_data['gui_rake'] = self.fault_rake_edit.text()
            
            # Get the sender widget for logging
            sender = self.sender()
            param_name = "Unknown"
            if sender == self.fault_rake_edit: param_name = "Rake"
            
            self.log_status(f"Fault '{self.current_fault_name}' slip vector '{param_name}' updated in GUI.")
            
            # Update visualization to reflect the changes
            self._update_fault_visualization()
            
        except Exception as e:
            self.log_status(f"ERROR during slip vector parameter update: {e}")
            import traceback
            self.log_status(traceback.format_exc())

    # New handler for axis visualization type change
    def on_axis_visualization_changed(self):
        """Handle changes to axis visualization type and update visualization"""
        if not self.current_fault_name:
            return
        
        fault_data = self.faults_data.get(self.current_fault_name)
        if not fault_data:
            return
        
        try:
            # Update the fault data with new values from GUI
            axis_visualization = self.axis_visualization_combo.currentText()
            fault_data['gui_axis_visualization'] = axis_visualization
            
            self.log_status(f"Fault '{self.current_fault_name}' axis visualization set to {axis_visualization}")
            
            # Update visualization to reflect the changes
            self._update_fault_visualization()
            
        except Exception as e:
            self.log_status(f"ERROR during axis visualization update: {e}")
            import traceback
            self.log_status(traceback.format_exc())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = LoopStructuralMiniGui()
    gui.show()
    sys.exit(app.exec()) 