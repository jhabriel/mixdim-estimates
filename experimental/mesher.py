import porepy as pp
import numpy as np

network = pp.fracture_importer.network_3d_from_csv("benchmark_3d_constrained.csv",
                                                   constraints=[1])
mesh_args = {"mesh_size_frac": 0.1, "mesh_size_bound": 0.1, "mesh_size_min": 0.1}
gb = network.mesh(mesh_args)
print(f"Number of 3D cells: {gb.grids_of_dimension(3)[0].num_cells}")
export = pp.Exporter(gb, file_name="geiger")
export.write_vtu()
