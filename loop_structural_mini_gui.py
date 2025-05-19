import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QGridLayout,
    QComboBox, QTextEdit, QFrame, QCheckBox, QListWidget, QInputDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator

import LoopStructural as LS
# import LoopStructural.visualisation as vis # No longer used
import pandas as pd
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor # Use QtInteractor for embedding

# Helper for PCA
from sklearn.decomposition import PCA

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
        self.setGeometry(100, 100, 1200, 800) # Wider for plotter

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout: controls on left, plotter on right
        self.main_h_layout = QHBoxLayout(self.central_widget)
        
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.main_h_layout.addWidget(self.controls_widget, 1) # Controls take 1/3 space

        self.pyvista_widget = PyVistaPlotterWidget(self)
        self.main_h_layout.addWidget(self.pyvista_widget, 2) # Plotter takes 2/3 space

        self.model_data_df = None
        self.geological_model = None
        self.strati_polydata = None # Stratigraphy remains singular for now

        # New structure for multiple faults
        # Key: fault_name (str)
        # Value: dict with keys:
        #   'polydata': pv.PolyData,
        #   'is_planar_fit': bool,
        #   'pca_center', 'pca_components', 'pca_extents': np.array (if PCA),
        #   'plane_center', 'plane_normal', 'in_plane_axes_vectors', 
        #   'in_plane_axes_lengths', 'planar_minor_axis_length': (if Planar Fit)
        #   'gui_center_x', 'gui_center_y', 'gui_center_z', 
        #   'gui_major_axis', 'gui_minor_axis', 'gui_intermediate_axis': str (values from GUI text fields for this fault)
        self.faults_data = {}
        self.current_fault_name = None

        self.double_validator = QDoubleValidator() # For float inputs

        self._create_input_widgets()
        self._create_control_buttons()
        self._create_status_area()
        
        self.controls_layout.addStretch(1) # Push status area to bottom

        self.model_updated_signal.connect(self.visualize_model_pyvista)

        self.log_status("GUI Initialized. Please load data and set parameters.")

    def _create_input_widgets(self):
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        current_row = 0

        # Fault List
        grid_layout.addWidget(QLabel("<b>Loaded Faults:</b>"), current_row, 0, 1, 2)
        self.fault_list_widget = QListWidget()
        self.fault_list_widget.setFixedHeight(100)
        self.fault_list_widget.currentItemChanged.connect(self.on_fault_selection_changed)
        grid_layout.addWidget(self.fault_list_widget, current_row + 1, 0, 1, 2); current_row += 2 # Span 2 rows for label and list
        
        # Origin
        grid_layout.addWidget(QLabel("<b>Model Origin (X,Y,Z):</b>"), current_row, 0, 1, 4); current_row += 1
        self.origin_x_edit = QLineEdit("0"); self.origin_x_edit.setValidator(self.double_validator)
        self.origin_y_edit = QLineEdit("0"); self.origin_y_edit.setValidator(self.double_validator)
        self.origin_z_edit = QLineEdit("0"); self.origin_z_edit.setValidator(self.double_validator)
        origin_layout = QHBoxLayout()
        origin_layout.addWidget(QLabel("X:")); origin_layout.addWidget(self.origin_x_edit)
        origin_layout.addWidget(QLabel("Y:")); origin_layout.addWidget(self.origin_y_edit)
        origin_layout.addWidget(QLabel("Z:")); origin_layout.addWidget(self.origin_z_edit)
        grid_layout.addLayout(origin_layout, current_row, 0, 1, 4); current_row += 1
        
        # Extent / Maximum
        grid_layout.addWidget(QLabel("<b>Model Maximum (X,Y,Z):</b>"), current_row, 0, 1, 4); current_row += 1
        self.max_x_edit = QLineEdit("1000"); self.max_x_edit.setValidator(self.double_validator) # Larger default for VTK data
        self.max_y_edit = QLineEdit("1000"); self.max_y_edit.setValidator(self.double_validator)
        self.max_z_edit = QLineEdit("1000"); self.max_z_edit.setValidator(self.double_validator)
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("X:")); max_layout.addWidget(self.max_x_edit)
        max_layout.addWidget(QLabel("Y:")); max_layout.addWidget(self.max_y_edit)
        max_layout.addWidget(QLabel("Z:")); max_layout.addWidget(self.max_z_edit)
        grid_layout.addLayout(max_layout, current_row, 0, 1, 4); current_row += 1

        # Fault Parameters
        grid_layout.addWidget(QLabel("<b>--- Fault Parameters ---</b>"), current_row, 0, 1, 4, Qt.AlignCenter); current_row += 1
        
        self.override_fault_geom_checkbox = QCheckBox("Override Calculated Fault Geometry")
        self.override_fault_geom_checkbox.setChecked(False)
        self.override_fault_geom_checkbox.stateChanged.connect(self.toggle_fault_geom_fields)
        grid_layout.addWidget(self.override_fault_geom_checkbox, current_row, 0, 1, 4); current_row += 1

        grid_layout.addWidget(QLabel("Fault Center (X,Y,Z):"), current_row, 0, 1, 1)
        self.fault_center_x_edit = QLineEdit(); self.fault_center_x_edit.setReadOnly(True); self.fault_center_x_edit.setValidator(self.double_validator)
        self.fault_center_y_edit = QLineEdit(); self.fault_center_y_edit.setReadOnly(True); self.fault_center_y_edit.setValidator(self.double_validator)
        self.fault_center_z_edit = QLineEdit(); self.fault_center_z_edit.setReadOnly(True); self.fault_center_z_edit.setValidator(self.double_validator)
        fault_center_layout = QHBoxLayout()
        fault_center_layout.addWidget(QLabel("X:")); fault_center_layout.addWidget(self.fault_center_x_edit)
        fault_center_layout.addWidget(QLabel("Y:")); fault_center_layout.addWidget(self.fault_center_y_edit)
        fault_center_layout.addWidget(QLabel("Z:")); fault_center_layout.addWidget(self.fault_center_z_edit)
        grid_layout.addLayout(fault_center_layout, current_row, 1, 1, 3); current_row += 1
        
        grid_layout.addWidget(QLabel("Major Axis Length:"), current_row, 0)
        self.fault_major_axis_edit = QLineEdit(); self.fault_major_axis_edit.setReadOnly(True); self.fault_major_axis_edit.setValidator(self.double_validator)
        grid_layout.addWidget(self.fault_major_axis_edit, current_row, 1)
        grid_layout.addWidget(QLabel("Minor Axis Length:"), current_row, 2)
        self.fault_minor_axis_edit = QLineEdit(); self.fault_minor_axis_edit.setReadOnly(True); self.fault_minor_axis_edit.setValidator(self.double_validator)
        grid_layout.addWidget(self.fault_minor_axis_edit, current_row, 3); current_row += 1
        
        grid_layout.addWidget(QLabel("Intermediate Axis Length:"), current_row, 0)
        self.fault_intermediate_axis_edit = QLineEdit("10"); self.fault_intermediate_axis_edit.setValidator(self.double_validator) # Remains manual/default
        grid_layout.addWidget(self.fault_intermediate_axis_edit, current_row, 1); current_row += 1


        grid_layout.addWidget(QLabel("Displacement:"), current_row, 0)
        self.fault_displacement_edit = QLineEdit("1"); self.fault_displacement_edit.setValidator(self.double_validator)
        grid_layout.addWidget(self.fault_displacement_edit, current_row, 1)
        grid_layout.addWidget(QLabel("Nelements:"), current_row, 2)
        self.fault_nelements_edit = QLineEdit("1000"); self.fault_nelements_edit.setValidator(QDoubleValidator(0, 1000000, 0)) # Integer
        grid_layout.addWidget(self.fault_nelements_edit, current_row, 3); current_row +=1

        grid_layout.addWidget(QLabel("Interpolator Type:"), current_row, 0)
        self.fault_interpolator_combo = QComboBox(); self.fault_interpolator_combo.addItems(["PLI", "FDI", "Surfe"])
        grid_layout.addWidget(self.fault_interpolator_combo, current_row, 1)
        grid_layout.addWidget(QLabel("Fault Buffer:"), current_row, 2)
        self.fault_buffer_edit = QLineEdit("0.5"); self.fault_buffer_edit.setValidator(self.double_validator)
        grid_layout.addWidget(self.fault_buffer_edit, current_row, 3); current_row +=1
        
        # Foliation Parameters
        grid_layout.addWidget(QLabel("<b>--- Foliation Parameters ('strati') ---</b>"), current_row, 0, 1, 4, Qt.AlignCenter); current_row +=1
        grid_layout.addWidget(QLabel("Nelements:"), current_row, 0)
        self.foliation_nelements_edit = QLineEdit("1000"); self.foliation_nelements_edit.setValidator(QDoubleValidator(0,1000000,0))
        grid_layout.addWidget(self.foliation_nelements_edit, current_row, 1)
        grid_layout.addWidget(QLabel("Interpolator Type:"), current_row, 2)
        self.foliation_interpolator_combo = QComboBox(); self.foliation_interpolator_combo.addItems(["PLI", "FDI", "Surfe"])
        grid_layout.addWidget(self.foliation_interpolator_combo, current_row, 3); current_row +=1
        
        self.controls_layout.addLayout(grid_layout)

    def on_fault_selection_changed(self, current_item, previous_item):
        if current_item is not None:
            self.current_fault_name = current_item.text()
            self.log_status(f"Selected fault: {self.current_fault_name}")
            self.update_fault_gui_fields(self.current_fault_name)
            # Trigger a plotter update to potentially highlight or focus on this fault's axes
            # For now, _add_fault_axes_widget will draw all, so an explicit call might not be needed yet
            # unless we want highlighting.
            if self.faults_data.get(self.current_fault_name, {}).get('polydata'):
                self.pyvista_widget.plotter.clear() # Clear plotter before redrawing, might need refinement
                # Re-add all fault meshes
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
                self.pyvista_widget.plotter.reset_camera()

        else:
            self.current_fault_name = None
            # Clear fault-specific GUI fields if no fault is selected
            self.fault_center_x_edit.clear(); self.fault_center_y_edit.clear(); self.fault_center_z_edit.clear()
            self.fault_major_axis_edit.clear(); self.fault_minor_axis_edit.clear(); self.fault_intermediate_axis_edit.clear()

    def update_fault_gui_fields(self, fault_name):
        fault_data = self.faults_data.get(fault_name)
        if fault_data:
            self.fault_center_x_edit.setText(fault_data.get('gui_center_x', ""))
            self.fault_center_y_edit.setText(fault_data.get('gui_center_y', ""))
            self.fault_center_z_edit.setText(fault_data.get('gui_center_z', ""))
            self.fault_major_axis_edit.setText(fault_data.get('gui_major_axis', ""))
            self.fault_minor_axis_edit.setText(fault_data.get('gui_minor_axis', ""))
            self.fault_intermediate_axis_edit.setText(fault_data.get('gui_intermediate_axis', ""))
            # Read-only status based on override checkbox and if data exists
            is_editable = self.override_fault_geom_checkbox.isChecked()
            self.fault_center_x_edit.setReadOnly(not is_editable or not fault_data.get('gui_center_x'))
            self.fault_center_y_edit.setReadOnly(not is_editable or not fault_data.get('gui_center_y'))
            self.fault_center_z_edit.setReadOnly(not is_editable or not fault_data.get('gui_center_z'))
            self.fault_major_axis_edit.setReadOnly(not is_editable or not fault_data.get('gui_major_axis'))
            self.fault_minor_axis_edit.setReadOnly(not is_editable or not fault_data.get('gui_minor_axis'))
            # self.fault_intermediate_axis_edit is currently always editable for its default value if no calc value

    def toggle_fault_geom_fields(self, state):
        is_editable = (state == Qt.Checked)
        self.fault_center_x_edit.setReadOnly(not is_editable)
        self.fault_center_y_edit.setReadOnly(not is_editable)
        self.fault_center_z_edit.setReadOnly(not is_editable)
        self.fault_major_axis_edit.setReadOnly(not is_editable)
        self.fault_minor_axis_edit.setReadOnly(not is_editable)
        # Intermediate axis is always editable for now
        # self.fault_intermediate_axis_edit.setReadOnly(not is_editable)


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
        # Clear all actors that might have been added by this method previously using specific prefixes
        actors_to_remove = []
        for actor in plotter.renderer.actors.keys():
            if actor.startswith("FAULTAXIS_"):
                actors_to_remove.append(actor)
        for actor_name in actors_to_remove:
            plotter.remove_actor(actor_name, render=False) # Batch removal, render at end if needed

        legend_entries = []

        for fault_name, fault_data in self.faults_data.items():
            if fault_data.get('is_planar_fit'):
                if fault_data.get('plane_center') is not None and fault_data.get('plane_normal') is not None and \
                   fault_data.get('in_plane_axes_vectors') is not None and fault_data.get('in_plane_axes_lengths') is not None:
                    
                    center = fault_data['plane_center']
                    normal_vec = fault_data['plane_normal']
                    in_plane_vecs = fault_data['in_plane_axes_vectors']
                    in_plane_lengths = fault_data['in_plane_axes_lengths']
                    minor_axis_len = fault_data.get('planar_minor_axis_length', 0.1)

                    # In-Plane Major Axis (Red)
                    if len(in_plane_vecs) > 0 and len(in_plane_lengths) > 0:
                        axis1_vec, axis1_len = in_plane_vecs[0], in_plane_lengths[0]
                        if axis1_len < 0.01 : axis1_len = 0.1
                        p_a, p_b = center - axis1_vec * (axis1_len/2), center + axis1_vec * (axis1_len/2)
                        plotter.add_mesh(pv.Line(p_a, p_b), color='red', line_width=3, name=f"FAULTAXIS_{fault_name}_InPlaneMajor")
                        if fault_name == self.current_fault_name or len(self.faults_data) == 1: legend_entries.append((f"{fault_name} In-Plane Major", "red"))

                    # In-Plane Intermediate Axis (Green)
                    if len(in_plane_vecs) > 1 and len(in_plane_lengths) > 1:
                        axis2_vec, axis2_len = in_plane_vecs[1], in_plane_lengths[1]
                        if axis2_len < 0.01 : axis2_len = 0.1
                        p_a, p_b = center - axis2_vec * (axis2_len/2), center + axis2_vec * (axis2_len/2)
                        plotter.add_mesh(pv.Line(p_a, p_b), color='green', line_width=3, name=f"FAULTAXIS_{fault_name}_InPlaneInter")
                        if fault_name == self.current_fault_name or len(self.faults_data) == 1: legend_entries.append((f"{fault_name} In-Plane Inter.", "green"))

                    # Plane Normal Axis (Blue)
                    norm_len_vis = minor_axis_len if minor_axis_len > 0.01 else 0.1
                    p_a_norm, p_b_norm = center - normal_vec * (norm_len_vis/2), center + normal_vec * (norm_len_vis/2)
                    plotter.add_mesh(pv.Line(p_a_norm, p_b_norm), color='blue', line_width=3, name=f"FAULTAXIS_{fault_name}_PlaneNormal")
                    if fault_name == self.current_fault_name or len(self.faults_data) == 1: legend_entries.append((f"{fault_name} Plane Normal", "blue"))
                    
                    sphere_radius = np.mean(in_plane_lengths) * 0.01 if len(in_plane_lengths) > 0 and np.mean(in_plane_lengths) > 0 else 0.1
                    if sphere_radius < 0.01: sphere_radius = 0.1
                    plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=center), color='yellow', name=f'FAULTAXIS_{fault_name}_CenterMarker')
                # else: self.log_status(f"Planar fit data incomplete for axes widget for {fault_name}.")
            
            elif fault_data.get('pca_center') is not None and \
                 fault_data.get('pca_components') is not None and \
                 fault_data.get('pca_extents') is not None: # Original PCA path
                
                center = fault_data['pca_center']
                pc_vectors = fault_data['pca_components'] 
                pc_lengths = fault_data['pca_extents']   

                colors = ['darkred', 'darkgreen', 'darkblue'] # Darker for PCA to distinguish from planar fit axes if shown together
                labels_base = ['PC1', 'PC2', 'PC3']

                max_overall_extent = np.max(pc_lengths) if pc_lengths.size > 0 and np.max(pc_lengths) > 1e-6 else 1.0
                for i in range(3):
                    vec, length = pc_vectors[i], pc_lengths[i]
                    MIN_LINE_LENGTH_FRACTION, FALLBACK_MIN_LENGTH = 0.02, 0.1
                    min_visual_length = max(max_overall_extent * MIN_LINE_LENGTH_FRACTION, FALLBACK_MIN_LENGTH / 10.0) if max_overall_extent > 1e-6 else FALLBACK_MIN_LENGTH
                    if length < min_visual_length: length = min_visual_length
                                    
                    p_a, p_b = center - vec * (length/2), center + vec * (length/2)
                    plotter.add_mesh(pv.Line(p_a, p_b), color=colors[i], line_width=3, name=f"FAULTAXIS_{fault_name}_{labels_base[i]}")
                    if fault_name == self.current_fault_name or len(self.faults_data) == 1: legend_entries.append((f"{fault_name} {labels_base[i]}", colors[i]))
                
                min_ext_for_sphere = np.min(pc_lengths[pc_lengths > 1e-6]) if np.any(pc_lengths > 1e-6) else 1.0
                sphere_radius = min_ext_for_sphere * 0.05 
                if sphere_radius < 0.01: sphere_radius = np.mean(pc_lengths[pc_lengths > 1e-6]) * 0.01 if np.any(pc_lengths > 1e-6) else 0.1
                if sphere_radius < 0.01: sphere_radius = 0.1 
                plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=center), color='gold', name=f'FAULTAXIS_{fault_name}_PCACenterMarker')
            # else: self.log_status(f"Fault geometry data (PCA or Planar) not available for {fault_name} for axes widget.")

        if legend_entries:
            plotter.add_legend(labels=legend_entries, bcolor='white', face='triangle')
        # self.log_status(f"Fault axes widgets updated for {len(self.faults_data)} faults.")

    def load_fault_vtk(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Fault VTK/VTP", "", "VTK Files (*.vtk *.vtp)")
        if file_path:
            default_name = file_path.split('/')[-1].split('.')[0] # Default to filename without extension
            fault_name, ok = QInputDialog.getText(self, "Fault Name", "Enter a unique name for this fault:", text=default_name)
            
            if not ok or not fault_name:
                self.log_status("Fault loading cancelled or no name provided.")
                return
            
            if fault_name in self.faults_data:
                self.log_status(f"ERROR: Fault name '{fault_name}' already exists. Please use a unique name.")
                return

            try:
                polydata = pv.read(file_path)
                self.log_status(f"Fault data '{fault_name}' loaded: {file_path}. Points: {polydata.n_points}")
                
                current_fault_entry = {'polydata': polydata}
                self.faults_data[fault_name] = current_fault_entry
                self.current_fault_name = fault_name # Make the newly loaded fault current

                points = polydata.points
                current_fault_entry['is_planar_fit'] = False # Reset flag
                # Initialize all possible geometry keys to None for this fault
                for key in ['pca_center', 'pca_components', 'pca_extents', 
                            'plane_center', 'plane_normal', 'in_plane_axes_vectors', 
                            'in_plane_axes_lengths', 'planar_minor_axis_length']:
                    current_fault_entry[key] = None

                if points.shape[0] == 0:
                    self.log_status(f"ERROR: Fault '{fault_name}' VTK contains no points.")
                    # Clean up entry if error
                    del self.faults_data[fault_name]
                    if self.current_fault_name == fault_name: self.current_fault_name = None
                    return

                calculated_center = np.mean(points, axis=0)
                # Store center for both potential cases initially, associated with this fault
                current_fault_entry['plane_center'] = calculated_center 
                current_fault_entry['pca_center'] = calculated_center
                
                # Store text representations for GUI update
                current_fault_entry['gui_center_x'] = f"{calculated_center[0]:.2f}"
                current_fault_entry['gui_center_y'] = f"{calculated_center[1]:.2f}"
                current_fault_entry['gui_center_z'] = f"{calculated_center[2]:.2f}"

                if points.shape[0] >= 10: # Use PCA for 10 or more points
                    current_fault_entry['is_planar_fit'] = False
                    self.log_status(f"Using PCA for fault '{fault_name}' geometry (>= 10 points).")
                    pca = PCA(n_components=3)
                    pca.fit(points)
                    actual_axis_lengths_along_pcs = np.ptp(pca.transform(points), axis=0)
                    
                    sorted_geometric_lengths = np.sort(actual_axis_lengths_along_pcs)[::-1]
                    L_major, L_intermediate, L_minor = sorted_geometric_lengths[0], sorted_geometric_lengths[1], sorted_geometric_lengths[2]
                    
                    current_fault_entry['gui_major_axis'] = f"{L_major:.2f}"
                    current_fault_entry['gui_minor_axis'] = f"{L_minor:.2f}"
                    current_fault_entry['gui_intermediate_axis'] = f"{L_intermediate:.2f}"
                    
                    current_fault_entry['pca_components'] = pca.components_
                    current_fault_entry['pca_extents'] = actual_axis_lengths_along_pcs
                    self.log_status(f"PCA ('{fault_name}'): Center {calculated_center}, GeoLengths(Maj,Int,Min): {L_major:.2f},{L_intermediate:.2f},{L_minor:.2f}, PCA Extents: {actual_axis_lengths_along_pcs}")

                else: # Use Planar Fit for < 10 points
                    current_fault_entry['is_planar_fit'] = True
                    self.log_status(f"Using Planar Fit for fault '{fault_name}' geometry ({points.shape[0]} points).")

                    if points.shape[0] == 1:
                        L_major, L_intermediate, L_minor = 0.0, 0.0, 0.0
                        current_fault_entry['plane_normal'] = np.array([0,0,1])
                        current_fault_entry['in_plane_axes_vectors'] = [np.array([1,0,0]), np.array([0,1,0])]
                        current_fault_entry['in_plane_axes_lengths'] = [0.0, 0.0]
                        current_fault_entry['planar_minor_axis_length'] = 0.0
                    elif points.shape[0] == 2:
                        p1, p2 = points[0], points[1]
                        vec = p2 - p1
                        L_major = np.linalg.norm(vec)
                        L_intermediate, L_minor = 0.0, 0.0
                        plane_normal_val = np.array([1,0,0]) # Default
                        if np.abs(vec[0]) > 1e-6 or np.abs(vec[1]) > 1e-6:
                            plane_normal_val = np.cross(vec, np.array([0,0,1]))
                        if np.linalg.norm(plane_normal_val) < 1e-6: 
                             plane_normal_val = np.array([1,0,0]) 
                        plane_normal_val /= np.linalg.norm(plane_normal_val)
                        current_fault_entry['plane_normal'] = plane_normal_val
                        
                        axis1_vec = vec / L_major if L_major > 1e-6 else np.array([1,0,0])
                        axis2_vec = np.cross(plane_normal_val, axis1_vec)
                        axis2_vec_norm = np.linalg.norm(axis2_vec)
                        axis2_vec = axis2_vec / axis2_vec_norm if axis2_vec_norm > 1e-6 else np.array([0,1,0])
                        current_fault_entry['in_plane_axes_vectors'] = [axis1_vec, axis2_vec]
                        current_fault_entry['in_plane_axes_lengths'] = [L_major, 0.0]
                        current_fault_entry['planar_minor_axis_length'] = 0.0
                    else: # 3 to 9 points
                        centered_points = points - calculated_center
                        try:
                            U, S, Vt = np.linalg.svd(centered_points)
                            current_fault_entry['plane_normal'] = Vt[-1, :]
                        except np.linalg.LinAlgError:
                            self.log_status(f"SVD for planar fit ('{fault_name}') failed. Using default normal.")
                            current_fault_entry['plane_normal'] = np.array([0,0,1]) 

                        in_plane_axis1_vec = Vt[0, :]
                        in_plane_axis2_vec = Vt[1, :]
                        coords_axis1 = centered_points @ in_plane_axis1_vec
                        coords_axis2 = centered_points @ in_plane_axis2_vec
                        L_major_in_plane = np.ptp(coords_axis1)
                        L_intermediate_in_plane = np.ptp(coords_axis2)
                        
                        if L_intermediate_in_plane > L_major_in_plane:
                            L_major_in_plane, L_intermediate_in_plane = L_intermediate_in_plane, L_major_in_plane
                            in_plane_axis1_vec, in_plane_axis2_vec = in_plane_axis2_vec, in_plane_axis1_vec
                        
                        L_minor_val = 0.0 
                        if S.size >=3 : L_minor_val = S[-1] * 2 
                        current_fault_entry['in_plane_axes_vectors'] = [in_plane_axis1_vec, in_plane_axis2_vec]
                        current_fault_entry['in_plane_axes_lengths'] = [L_major_in_plane, L_intermediate_in_plane]
                        current_fault_entry['planar_minor_axis_length'] = L_minor_val
                        L_major, L_intermediate, L_minor = L_major_in_plane, L_intermediate_in_plane, L_minor_val
                    
                    current_fault_entry['gui_major_axis'] = f"{L_major:.2f}"
                    current_fault_entry['gui_intermediate_axis'] = f"{L_intermediate:.2f}"
                    current_fault_entry['gui_minor_axis'] = f"{L_minor:.2f}"
                    self.log_status(f"PlanarFit ('{fault_name}'): Center {calculated_center}, GeoLengths(Maj,Int,Min): {L_major:.2f},{L_intermediate:.2f},{L_minor:.2f}")
                    if current_fault_entry['plane_normal'] is not None: self.log_status(f"PlanarFit ('{fault_name}'): Normal {current_fault_entry['plane_normal']}")
                
                # Add to list widget and update GUI for the current fault
                self.fault_list_widget.addItem(fault_name)
                self.fault_list_widget.setCurrentRow(self.fault_list_widget.count() - 1) # Select the new item
                self.update_fault_gui_fields(fault_name) # This will populate the QLineEdits

                # Refresh PyVista Plotter
                self.pyvista_widget.plotter.clear()
                for name, data_dict in self.faults_data.items():
                    if 'polydata' in data_dict:
                        pd_item = data_dict['polydata']
                        style_args_pv = {'style':'wireframe', 'color':'darkgrey', 'line_width':2, 'name':f"fault_input_{name}"}
                        if pd_item.n_points <= 10:
                            style_args_pv = {'style':'points', 'color':'magenta', 'point_size':15, 'name':f"fault_input_points_few_{name}"}
                            self.pyvista_widget.plotter.add_mesh(pd_item.outline(), color='cyan', name=f"fault_input_bbox_{name}")
                        self.pyvista_widget.plotter.add_mesh(pd_item, **style_args_pv)
                
                if self.strati_polydata: # Strati data is still singular
                     self.pyvista_widget.plotter.add_mesh(self.strati_polydata, style='points', color='blue', point_size=5, name="strati_input_vtk_points")
                
                self._add_fault_axes_widget(self.pyvista_widget.plotter) # This needs to draw axes for ALL faults
                self.pyvista_widget.plotter.reset_camera()

            except Exception as e:
                self.log_status(f"ERROR loading fault VTK for '{fault_name}': {e}")
                if fault_name in self.faults_data: # Clean up partial entry on error
                    del self.faults_data[fault_name]
                # Remove from list widget if added
                items = self.fault_list_widget.findItems(fault_name, Qt.MatchExactly)
                if items: self.fault_list_widget.takeItem(self.fault_list_widget.row(items[0]))
                if self.current_fault_name == fault_name: self.current_fault_name = None
                import traceback; self.log_status(traceback.format_exc())

    def load_strati_vtk(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Stratigraphy VTK/VTP", "", "VTK Files (*.vtk *.vtp)")
        if file_path:
            try:
                self.strati_polydata = pv.read(file_path)
                self.log_status(f"Strati data loaded: {file_path}. Points: {self.strati_polydata.n_points}")
                # Trigger PyVista update
                self.pyvista_widget.plotter.clear()
                if self.fault_polydata:
                    self.pyvista_widget.plotter.add_mesh(self.fault_polydata, style='wireframe', color='darkgrey', line_width=2, name="fault_input_vtk")
                    self._add_fault_axes_widget(self.pyvista_widget.plotter)
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

                fault_params_ls = {
                    'nelements': fault_nelements, 'interpolatortype': fault_interpolator,
                    'fault_buffer': fault_buffer_val, 'fault_center': fault_center_val,
                    'major_axis': fault_major_axis, 'minor_axis': fault_minor_axis,
                    'intermediate_axis': fault_intermediate_axis, 'points': True
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = LoopStructuralMiniGui()
    gui.show()
    sys.exit(app.exec()) 