# Importing modules
import numpy as np
import porepy as pp

from time import time
from model_local import model_local
from model_global import model_global


# %% Functions
def make_constrained_mesh(h=0.1):
    """
    Creates unstructured mesh for a given target mesh size for the case of a
    single vertical fracture embedded in the domain

    Parameters
    ----------
    h : float, optional
        Target mesh size. The default is 0.1.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network_2d = pp.fracture_importer.network_2d_from_csv(
        "network.csv", domain=domain
    )

    # Target lengths
    target_h_bound = h
    target_h_fract = h
    mesh_args = {
        "mesh_size_bound": target_h_bound,
        "mesh_size_frac": target_h_fract,
    }
    # Construct grid bucket
    gb = network_2d.mesh(mesh_args, constraints=[1, 2])

    return gb


# %% Define numerical methods and obtain grid buckets
mesh_targets = np.array([0.05, 0.025, 0.0125, 0.00625])
num_methods = ["RT0", "MVEM", "MPFA", "TPFA"]

print("Assembling grid buckets...", end="")
tic = time()
grid_buckets = []
for h in mesh_targets:
    grid_buckets.append(make_constrained_mesh(h))
print(f"\u2713 Time {time() - tic}.\n")

#%% Create a dictionary to store all the data
d = {k: {} for k in num_methods}
for method in num_methods:
    d[method] = {
        "bulk_diffusive": [],
        "bulk_residual_NC": [],
        "bulk_residual_LC": [],
        "frac_diffusive": [],
        "frac_residual_NC": [],
        "frac_residual_LC": [],
        "mortar_left": [],
        "mortar_right": [],
        "majorant_NC": [],
        "majorant_LC": [],
        "majorant_combined_NC": [],
        "majorant_combined_LC": [],
        "efficiency_pressure_NC": [],
        "efficiency_pressure_LC": [],
        "efficiency_velocity_NC": [],
        "efficiency_velocity_LC": [],
        "efficiency_combined_NC": [],
        "efficiency_combined_LC": [],
    }

# %% Loop over the models
models = [model_global, model_local]
for model in models:
    if model.__name__ == "model_global":
        print("***Using the mixed-dimensional Poincare constant***")
    else:
        print("***Using the local Poincare constant***")

    # Populate fields (NOTE: This loop may take considerable time)
    for method in num_methods:
        for idx, gb in enumerate(grid_buckets):
            print(f"Solving with {method} for refinement level {idx+1}.")
            # Get hold of errors
            tic = time()
            out = model(gb, method)  # dictionary containing all the errors
            print(f"Done. Time {time() - tic}\n")
            # Store the relevant fields in our data dictionary d
            if model.__name__ == "model_global":
                d[method]["bulk_diffusive"].append(out["bulk"]["diffusive_error"])
                d[method]["bulk_residual_NC"].append(out["bulk"]["residual_error"])
                d[method]["frac_diffusive"].append(out["frac"]["diffusive_error"])
                d[method]["frac_residual_NC"].append(out["frac"]["residual_error"])
                d[method]["mortar_left"].append(out["mortar_left"]["error"])
                d[method]["mortar_right"].append(out["mortar_right"]["error"])
                d[method]["majorant_NC"].append(out["majorant_pressure"])
                d[method]["majorant_combined_NC"].append(out["majorant_combined"])
                d[method]["efficiency_pressure_NC"].append(out["efficiency_pressure"])
                d[method]["efficiency_velocity_NC"].append(out["efficiency_velocity"])
                d[method]["efficiency_combined_NC"].append(out["efficiency_combined"])
            else:
                d[method]["bulk_residual_LC"].append(out["bulk"]["residual_error"])
                d[method]["frac_residual_LC"].append(out["frac"]["residual_error"])
                d[method]["majorant_LC"].append(out["majorant_pressure"])
                d[method]["majorant_combined_LC"].append(out["majorant_combined"])
                d[method]["efficiency_pressure_LC"].append(out["efficiency_pressure"])
                d[method]["efficiency_velocity_LC"].append(out["efficiency_velocity"])
                d[method]["efficiency_combined_LC"].append(out["efficiency_combined"])

# %% Exporting
# Permutations
rows = len(num_methods) * len(grid_buckets)

# Initialize lists
num_method_name = []
bulk_diffusive = []
bulk_residual_NC = []
bulk_residual_LC = []
frac_diffusive = []
frac_residual_NC = []
frac_residual_LC = []
mortar_left = []
mortar_right = []
majorant_NC = []
majorant_LC = []
majorant_combined_NC = []
majorant_combined_LC = []
I_eff_pressure_NC = []
I_eff_pressure_LC = []
I_eff_velocity_NC = []
I_eff_velocity_LC = []
I_eff_combined_NC = []
I_eff_combined_LC = []

# Populate lists
for method in num_methods:
    for idx in range(mesh_targets.size):
        num_method_name.append(method)
        bulk_diffusive.append(d[method]["bulk_diffusive"][idx])
        bulk_residual_NC.append(d[method]["bulk_residual_NC"][idx])
        bulk_residual_LC.append(d[method]["bulk_residual_LC"][idx])
        frac_diffusive.append(d[method]["frac_diffusive"][idx])
        frac_residual_NC.append(d[method]["frac_residual_NC"][idx])
        frac_residual_LC.append(d[method]["frac_residual_LC"][idx])
        mortar_left.append(d[method]["mortar_left"][idx])
        mortar_right.append(d[method]["mortar_right"][idx])
        majorant_NC.append(d[method]["majorant_NC"][idx])
        majorant_LC.append(d[method]["majorant_LC"][idx])
        majorant_combined_NC.append(d[method]["majorant_combined_NC"][idx])
        majorant_combined_LC.append(d[method]["majorant_combined_LC"][idx])
        I_eff_pressure_NC.append(d[method]["efficiency_pressure_NC"][idx])
        I_eff_pressure_LC.append(d[method]["efficiency_pressure_LC"][idx])
        I_eff_velocity_NC.append(d[method]["efficiency_velocity_NC"][idx])
        I_eff_velocity_LC.append(d[method]["efficiency_velocity_LC"][idx])
        I_eff_combined_NC.append(d[method]["efficiency_combined_NC"][idx])
        I_eff_combined_LC.append(d[method]["efficiency_combined_LC"][idx])

#%% Export first table
export = np.zeros(
    rows,
    dtype=[
        ("var1", "U6"),
        ("var2", float),
        ("var3", float),
        ("var4", float),
        ("var5", float),
        ("var6", float),
        ("var7", float),
        ("var8", float),
        ("var9", float),
    ],
)

# Declaring column variables
export["var1"] = num_method_name
export["var2"] = bulk_diffusive
export["var3"] = bulk_residual_NC
export["var4"] = bulk_residual_LC
export["var5"] = frac_diffusive
export["var6"] = frac_residual_NC
export["var7"] = frac_residual_LC
export["var8"] = mortar_left
export["var9"] = mortar_right

# Formatting string
fmt = "%6s %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e"

# Headers
header = "num_method eta_DF_2d eta_R_NC_2d eta_R_LC_2d eta_DF_1d eta_R_1d_NC eta_R_1d_LC " \
         "eta_mortar_l eta_mortar_r"

# Writing into txt
np.savetxt("val_2d_a.txt", export, delimiter=",", fmt=fmt, header=header)

#%% Export second table
export = np.zeros(
    rows,
    dtype=[
        ("var1", "U6"),
        ("var2", float),
        ("var3", float),
        ("var4", float),
        ("var5", float),
        ("var6", float),
        ("var7", float),
        ("var8", float),
        ("var9", float),
        ("var10", float),
        ("var11", float),
    ],
)

# Declaring column variables
export["var1"] = num_method_name
export["var2"] = majorant_NC
export["var3"] = majorant_LC
export["var4"] = majorant_combined_NC
export["var5"] = majorant_combined_LC
export["var6"] = I_eff_pressure_NC
export["var7"] = I_eff_pressure_LC
export["var8"] = I_eff_velocity_NC
export["var9"] = I_eff_velocity_LC
export["var10"] = I_eff_combined_NC
export["var11"] = I_eff_combined_LC

# Formatting string
fmt = "%6s %2.2e %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f"

# Headers
header = "method M_NC M_LC M_pu_NC M_pu_LC I_eff_p_NC I_eff_p_LC I_eff_u_NC I_eff_u_LC " \
         "I_eff_pu_NC I_eff_pu_LC"

# Writing into text
np.savetxt("val_2d_b.txt", export, delimiter=",", fmt=fmt, header=header)


