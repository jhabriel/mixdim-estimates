# Importing modules
import numpy as np
import porepy as pp

from time import time
from model_local import model_local
from model_global import model_global


# %% Functions
def make_constrained_mesh(mesh_size=0.2):
    """
    Creates an unstructured 3D mesh for a given target mesh size for the case
    of a  single 2D vertical fracture embedded in a 3D domain

    Parameters
    ----------
    mesh_size : float, optional
        Target mesh size. The default is 0.2.

    Returns
    -------
    gb : PorePy Object
        Porepy grid bucket object.

    """
    # Load fracture network: Fracture + Ghost Fractures
    network_3d = pp.fracture_importer.network_3d_from_csv("network.csv")

    # Create mesh_arg dictionary
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": mesh_size,
        "mesh_size_min": mesh_size / 10,
    }

    # Construct grid bucket
    ghost_fracs = list(np.arange(1, 25))  # 1 to 24
    gb = network_3d.mesh(mesh_args, constraints=ghost_fracs)

    return gb


# %% Defining mesh targets, numerical methods, and dictionary fields
mesh_targets = np.array([0.3, 0.15, 0.075, 0.0375])
num_methods = ["RT0", "MVEM", "MPFA", "TPFA"]

# %% Obtain grid buckets for each mesh size
print("Assembling grid buckets...", end="")
tic = time()
grid_buckets = []
for h in mesh_targets:
    grid_buckets.append(make_constrained_mesh(h))
print(f"\u2713 Time {time() - tic}.\n")

# %% Loop over the models
models = [model_local, model_global]
for model in models:
    # Create dictionary and initialize fields
    d = {k: {} for k in num_methods}
    for method in num_methods:
        d[method] = {
            "bulk_error": [],
            "frac_error": [],
            "mortar_error": [],
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
            print(f"Solving with {method} for refinement level {idx + 1}.")
            # Get hold of errors
            tic = time()
            out = model(gb, method)
            print(f"Done. Time {time() - tic}\n")
            # Store errors in the dictionary
            d[method]["bulk_error"].append(out["bulk"]["error"])
            d[method]["frac_error"].append(out["frac"]["error"])
            d[method]["mortar_error"].append(out["mortar"]["error"])
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
    bulk_error = []
    frac_error = []
    mortar_error = []
    majorant_pressure = []
    majorant_velocity = []
    majorant_combined = []
    true_pressure_error = []
    true_velocity_error = []
    true_combined_error = []
    I_eff_pressure = []
    I_eff_velocity = []
    I_eff_combined = []

    # Populate lists
    for method in num_methods:
        for idx in range(len(mesh_targets)):
            num_method_name.append(method)
            bulk_error.append(d[method]["bulk_error"][idx])
            frac_error.append(d[method]["frac_error"][idx])
            mortar_error.append(d[method]["mortar_error"][idx])
            majorant_pressure.append(d[method]["majorant_pressure"][idx])
            majorant_velocity.append(d[method]["majorant_velocity"][idx])
            majorant_combined.append(d[method]["majorant_combined"][idx])
            true_pressure_error.append(d[method]["true_pressure_error"][idx])
            true_velocity_error.append(d[method]["true_velocity_error"][idx])
            true_combined_error.append(d[method]["true_combined_error"][idx])
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
            ("var13", float),
        ],
    )

    # Declaring column variables
    export["var1"] = num_method_name
    export["var2"] = bulk_error
    export["var3"] = frac_error
    export["var4"] = mortar_error
    export["var5"] = majorant_pressure
    export["var6"] = majorant_velocity
    export["var7"] = majorant_combined
    export["var8"] = true_pressure_error
    export["var9"] = true_velocity_error
    export["var10"] = true_combined_error
    export["var11"] = I_eff_pressure
    export["var12"] = I_eff_velocity
    export["var13"] = I_eff_combined

    # Formatting string
    fmt = "%6s %2.2e %2.2e %2.2e %2.2e %2.2e "
    fmt += "%2.2e %2.2e %2.2e %2.2e %2.2f %2.2f %2.2f"

    # Headers
    header = (
        "num_method eta_3d eta_2d eta_mortar M_p M_u M_pu tpe tve tce"
    )
    header += "I_eff_p I_eff_u I_eff_pu"

    # Writing into txt
    if model.__name__ == "model_local":
        np.savetxt(
            "validation3d_LC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
    else:
        np.savetxt(
            "validation3d_NC.txt",
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
            ("amp12", "U6"),
            ("var13", float),
        ],
    )

    # Prepare for exporting
    export["var1"] = num_method_name
    export["var2"] = bulk_error
    export["var3"] = frac_error
    export["var4"] = mortar_error
    export["var5"] = majorant_pressure
    export["var6"] = majorant_velocity
    export["var7"] = majorant_combined
    export["var8"] = true_pressure_error
    export["var9"] = true_velocity_error
    export["var10"] = true_combined_error
    export["var11"] = I_eff_pressure
    export["var12"] = I_eff_velocity
    export["var13"] = I_eff_combined
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
    export["amp12"] = ampersend

    # Formatting string
    fmt = "%6s "  # method
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # bulk error
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # frac error
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # mortar error
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant pressure
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant velocity
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # majorant combined
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # true pressure error
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # true velocity error
    fmt += "%1s "  # amp
    fmt += "%2.2e "  # true combined error
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency pressure
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency velocity
    fmt += "%1s "  # amp
    fmt += "%2.2f "  # efficiency combined

    # Headers
    header = "num_method & eta_3d & eta_2d & eta_mortar & M_p & M_u & M_pu "
    header += "tpe & tve & & tce & I_eff_p & I_eff_u & I_eff_pu"

    if model.__name__ == "model_local":
        np.savetxt(
            "validation3dtex_LC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
    else:
        np.savetxt(
            "validation3dtex_NC.txt",
            export,
            delimiter=",",
            fmt=fmt,
            header=header,
        )
