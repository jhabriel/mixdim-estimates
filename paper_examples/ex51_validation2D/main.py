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

# %% Defining numerical methods, and obtaining grid buckets
mesh_targets = np.array([0.05, 0.025, 0.0125, 0.00625])
num_methods = ["RT0", "MVEM", "MPFA", "TPFA"]

print("Assembling grid buckets...", end="")
tic = time()
grid_buckets = []
for h in mesh_targets:
    grid_buckets.append(make_constrained_mesh(h))
print(f"\u2713 Time {time() - tic}.\n")

# %% Loop over the models
models = [model_global, model_local]
for model in models:
    if model.__name__ == "model_global":
        print("\n Using the mixed-dimensional Poincare constant \n")
    else:
        print("\n Using the local Poincare constant \n")

    # Create dictionary and initialize fields
    d = {k: {} for k in num_methods}
    for method in num_methods:
        d[method] = {
            "bulk_diffusive": [],
            "bulk_residual": [],
            "frac_diffusive": [],
            "frac_residual": [],
            "mortar_diffusive": [],
            "majorant_pressure": [],
            "majorant_velocity": [],
            "majorant_combined": [],
            "true_pressure_error": [],
            "true_velocity_error": [],
            "true_combined_error": [],
            "efficiency_pressure": [],
            "efficiency_velocity": [],
            "efficiency_combined": [],
        }
    # Populate fields (NOTE: This loop may take considerable time)
    for method in num_methods:
        for idx, gb in enumerate(grid_buckets):
            print(f"Solving with {method} for refinement level {idx+1}.")
            # Get hold of errors
            tic = time()
            out = model(gb, method)
            print(f"Done. Time {time() - tic}\n")
            # Store errors in the dictionary
            d[method]["bulk_diffusive"].append(out["bulk"]["diffusive_error"])
            d[method]["bulk_residual"].append(out["bulk"]["residual_error"])
            d[method]["frac_diffusive"].append(out["frac"]["diffusive_error"])
            d[method]["frac_residual"].append(out["frac"]["residual_error"])
            d[method]["mortar_diffusive"].append(out["mortar"]["error"])
            d[method]["majorant_pressure"].append(out["majorant_pressure"])
            d[method]["majorant_velocity"].append(out["majorant_velocity"])
            d[method]["majorant_combined"].append(out["majorant_combined"])
            d[method]["true_pressure_error"].append(out["true_pressure_error"])
            d[method]["true_velocity_error"].append(out["true_velocity_error"])
            d[method]["true_combined_error"].append(out["true_combined_error"])
            d[method]["efficiency_pressure"].append(out["efficiency_pressure"])
            d[method]["efficiency_velocity"].append(out["efficiency_velocity"])
            d[method]["efficiency_combined"].append(out["efficiency_combined"])

    # %% Exporting
    # Permutations
    rows = len(num_methods) * len(grid_buckets)

    # Initialize lists
    num_method_name = []
    bulk_diffusive = []
    bulk_residual = []
    frac_diffusive = []
    frac_residual = []
    mortar_diffusive = []
    majorant_pressure = []
    majorant_velocity = []
    majorant_combined = []
    I_eff_pressure = []
    I_eff_velocity = []
    I_eff_combined = []

    # Populate lists
    for method in num_methods:
        for idx in range(mesh_targets.size):
            num_method_name.append(method)
            bulk_diffusive.append(d[method]["bulk_diffusive"][idx])
            bulk_residual.append(d[method]["bulk_residual"][idx])
            frac_diffusive.append(d[method]["frac_diffusive"][idx])
            frac_residual.append(d[method]["frac_residual"][idx])
            mortar_diffusive.append(d[method]["mortar_diffusive"][idx])
            majorant_pressure.append(d[method]["majorant_pressure"][idx])
            majorant_velocity.append(d[method]["majorant_velocity"][idx])
            majorant_combined.append(d[method]["majorant_combined"][idx])
            I_eff_pressure.append(d[method]["efficiency_pressure"][idx])
            I_eff_velocity.append(d[method]["efficiency_velocity"][idx])
            I_eff_combined.append(d[method]["efficiency_combined"][idx])

    # Prepare for exporting
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
            ("var12", float),
        ],
    )

    # Declaring column variables
    export["var1"] = num_method_name
    export["var2"] = bulk_diffusive
    export["var3"] = bulk_residual
    export["var4"] = frac_diffusive
    export["var5"] = frac_residual
    export["var6"] = mortar_diffusive
    export["var7"] = majorant_pressure
    export["var8"] = majorant_velocity
    export["var9"] = majorant_combined
    export["var10"] = I_eff_pressure
    export["var11"] = I_eff_velocity
    export["var12"] = I_eff_combined

    # Formatting string
    fmt = "%6s %2.2e %2.2e %2.2e %2.2e %2.2e"
    fmt += " %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f"

    # Headers
    header = (
        "num_method eta_DF_2d eta_R_2d eta_DF_1d eta_R_1d eta_mortar "
    )
    header += "majorant_p majorant_u majorant_pu I_eff_p I_eff_u I_eff_pu"

    # Writing into txt
    if model.__name__ == "model_local":
        np.savetxt(
            "validation2d_LC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
    else:
        np.savetxt(
            "validation2d_NC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )

    # %% Exporting to LaTeX
    # Initialize lists
    ampersend = []
    for i in range(rows):
        ampersend.append("&")

    # Prepare for exporting
    export = np.zeros(
        rows,
        dtype=[
            ("var1", "U6"),
            ("amp1", "U6"),
            ("var2", float),
            ("amp2", "U6"),
            ("var3", float),
            ("amp3", "U6"),
            ("var4", float),
            ("amp4", "U6"),
            ("var5", float),
            ("amp5", "U6"),
            ("var6", float),
            ("amp6", "U6"),
            ("var7", float),
            ("amp7", "U6"),
            ("var8", float),
            ("amp8", "U6"),
            ("var9", float),
            ("amp9", "U6"),
            ("var10", float),
            ("amp10", "U6"),
            ("var11", float),
            ("amp11", "U6"),
            ("var12", float),
        ],
    )

    # Prepare for exporting
    export["var1"] = num_method_name
    export["var2"] = bulk_diffusive
    export["var3"] = bulk_residual
    export["var4"] = frac_diffusive
    export["var5"] = frac_residual
    export["var6"] = mortar_diffusive
    export["var7"] = majorant_pressure
    export["var8"] = majorant_velocity
    export["var9"] = majorant_combined
    export["var10"] = I_eff_pressure
    export["var11"] = I_eff_velocity
    export["var12"] = I_eff_combined
    export["amp1"] = ampersend
    export["amp2"] = ampersend
    export["amp3"] = ampersend
    export["amp4"] = ampersend
    export["amp5"] = ampersend
    export["amp6"] = ampersend
    export["amp7"] = ampersend
    export["amp8"] = ampersend
    export["amp9"] = ampersend
    export["amp10"] = ampersend
    export["amp11"] = ampersend

    # Formatting string
    fmt = "%6s "  # method
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # bulk diffusive
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # bulk residual
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # frac diffusive
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # frac residual
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # mortar diffusive
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant pressure
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant velocity
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant combined
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency pressure
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency velocity
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency combined

    # Headers
    header = "num_method & eta_DF_2d & eta_R_2d & eta_DF_1d & eta_R_1d & "
    header += "eta_DF_mortar & majorant_p & majorant_u & majorant_pu & "
    header += "I_eff_p & I_eff_u & I_eff_pu"

    if model.__name__ == "model_local":
        np.savetxt(
            "validation2dtex_LC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
    else:
        np.savetxt(
            "validation2dtex_NC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
