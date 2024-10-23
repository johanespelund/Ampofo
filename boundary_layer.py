# Script for calculating cell grading when wall refinement is selected
import numpy as np
import re


def cellcount_and_thickness(r, x_wall, x_bulk):
    """
    Calculate the thickness and number of cells in the boundary layer.
    Also return the ratio (expansion) between the wall and final cell size.
    r (float): growth ratio
    x_wall (float): wall cell size
    x_bulk (float): bulk cell size
    return (int, float, float): number of cells, thickness, expansion
    """
    n = 1
    x = x_wall
    t = x
    while x < x_bulk:
        n += 1
        x *= r
        t += x

    x_final = x

    return n, t, x_final / x_wall


def foam_grading(L, x_wall, x_bulk, r, twosided=True):
    """
    Calculate the grading for a direction in a foam block.
    L (float): length of the direction
    x_wall (float): wall cell size
    x_bulk (float): bulk cell size
    r (float): growth ratio
    twosided (bool): if the grading is applied to both ends of the edge
    return (str, str): total cell number string, grading string
    """
    n_bl, l_bl, expansion = cellcount_and_thickness(r, x_wall, x_bulk)

    l_bulk = L - l_bl * (1 + twosided)
    ratio_l_bulk = l_bulk / L
    ratio_l_bl = l_bl / L

    n_bulk = np.round(l_bulk / x_bulk)
    n = n_bulk + n_bl * (1 + twosided)
    ratio_n_bulk = n_bulk / n
    ratio_n_bl = n_bl / n

    debug = False
    if debug:
        print(f"{n_bl=}, {l_bl=}, {expansion=}")
        print(f"{n_bulk=}, {l_bulk=}")
        x_bulk = l_bulk / n_bulk
        print(f"{x_bulk=}, {x_wall=}, {x_bulk/x_wall=}")

    return {
        "ratio_l_bl": ratio_l_bl,
        "ratio_n_bl": ratio_n_bl,
        "expansion": expansion,
        "ratio_l_bulk": ratio_l_bulk,
        "ratio_n_bulk": ratio_n_bulk,
        "n": n,
        "twosided": twosided,
    }


def foam_grading_string(parameters: dict, flip=False):
    """
    Return a string representation of the grading for a direction in a foam block.
    flip (bool): if the grading is flipped (only relevant for one-sided grading)
    """

    ratio_l_bl = parameters["ratio_l_bl"]
    ratio_n_bl = parameters["ratio_n_bl"]
    expansion = parameters["expansion"]
    ratio_l_bulk = parameters["ratio_l_bulk"]
    ratio_n_bulk = parameters["ratio_n_bulk"]
    n = parameters["n"]
    twosided = parameters["twosided"]

    start_BL = f"({ratio_l_bl: .4f} {ratio_n_bl: .4f} {expansion: .4f})"
    bulk = f"({ratio_l_bulk: .4f} {ratio_n_bulk: .4f} 1)"
    end_BL = f"({ratio_l_bl: .4f} {ratio_n_bl: .4f} {expansion**-1: .4f})"
    s = start_BL + bulk
    if twosided:
        s += f" ({ratio_l_bl: .4f} {ratio_n_bl: .4f} {expansion**-1: .4f})"
    if not twosided and flip:
        s = bulk + end_BL

    return str(int(n)), f"({s})"


def calc_radial_cell_distribution(parameters):
    """
    Calculate the radial cell distribution for a wall refined mesh for the LH2 case.
    parameters (dict): dictionary of LH2 case parameters
    """
    r = parameters["r_BL"]
    x_bulk = parameters["cell_size"]
    x_wall = parameters["wall_cell_size"]
    R = parameters["R"]

    # Calculate total length of wall normal edge
    innerRatio = min(0.7, 0.4388 * parameters["H_G"] / R + 0.3812)
    vo = (R * R / 3) ** 0.5
    L = R - vo * innerRatio * 1.4

    grading_params = foam_grading(L, x_wall, x_bulk, r, twosided=False)

    BL = {
        "length_ratio": grading_params["ratio_l_bl"],
        "cell_ratio": grading_params["ratio_n_bl"],
        "growth_ratio": grading_params["expansion"],
        "inverse_growth_ratio": grading_params["expansion"] ** -1,
    }

    bulk = {
        "length_ratio": grading_params["ratio_l_bulk"],
        "cell_ratio": grading_params["ratio_n_bulk"],
    }

    parameters.update({"BL_" + key: value for key, value in BL.items()})
    parameters.update({"bulk_" + key: value for key, value in bulk.items()})
    parameters.update({"nr_wall_refinement": grading_params["n"]})


def calc_vertical_cell_distribution(parameters):
    r = parameters["r_BL"]
    x = parameters["cell_size"]
    x0 = parameters["wall_cell_size"]
    R = parameters["R"]

    # Calculate parameters for boundary layer region
    N_BL = np.ceil(np.log(x / x0) / np.log(r))  # Number of cells in the BL
    A = 0.82
    innerRatio = min(0.7, 0.4388 * parameters["H_G"] / R + 0.3812)
    vo = (R * R / 3) ** 0.5
    vi = vo * innerRatio
    ymin = R - parameters["H_G"]
    ymed = max(vi, R - A * parameters["H_G"])
    L = ymed - ymin
    iterating = True
    L_BL = 0.7 * x0 * (1 - r ** (N_BL - 1)) / (1 - r)  # Length of boundary layer

    # Calculate parameters for bulk region
    L_bulk = L - L_BL
    N_bulk = np.ceil(L_bulk / x)
    N = N_BL + N_bulk
    while iterating:
        if L_bulk < x:
            N_BL -= 1
            N_bulk += 1
            L_BL = 0.7 * x0 * (1 - r ** (N_BL - 1)) / (1 - r)  # Length of BL
            L_bulk = L - L_BL
            N = N_BL + N_bulk
        else:
            iterating = False

    BL = {
        "length_ratio": L_BL / L,
        "cell_ratio": N_BL / N,
        "growth_ratio": r ** (N_BL),
        "inverse_growth_ratio": r ** -(N_BL),
    }

    bulk = {"length_ratio": L_bulk / L, "cell_ratio": N_bulk / N}

    parameters.update({"BL_vert_" + key: value for key, value in BL.items()})
    parameters.update({"bulk_vert_" + key: value for key, value in bulk.items()})
    parameters.update(
        {
            "ny_wall_refinement": N,
            "nx_wall_refinement": 2 * np.ceil(1.3 * vi / x) + 1,
            "nz_wall_refinement": 2 * np.ceil(1.3 * vi / x) + 1,
        }
    )

# Function to replace the content within the parentheses after "blocks"
def replace_blocks_content(file_path, v_string, n_string, grading_string):
    with open(file_path, 'r') as file:
        content = file.read()

    # Define the pattern to match the blocks section
    pattern = r'(blocks\s*\(\s*hex\s*\(.*?\).*\s*simpleGrading\s*\(\s*.*?\s*\)\s*\)\s*;)'
    
    # Create the new blocks content
    new_blocks_content = f"blocks\n(\n  {v_string} {n_string}\n  simpleGrading\n  {grading_string}\n  );"

    # Replace the old blocks content with the new one
    updated_content = re.sub(pattern, new_blocks_content, content, flags=re.DOTALL)

    with open(file_path, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    # for L in [1, 0.32, 1]:
    #     x_wall = 1e-3
    #     x_bulk = (10/1.5)*1e-3
    #     r = 1.15
    #     grading_params = foam_grading(L, x_wall, x_bulk, r)
    #     n, grading = foam_grading_string(grading_params)
    #     print(n)
    #     print(grading)

    L = 1
    x_wall = 0.1e-3
    x_bulk = 10.0 * 1.5**0 * 1e-3
    x_wall = x_bulk / 1.2
    r = 1.15
    grading_params = foam_grading(L, x_wall, x_bulk, r)
    n, grading = foam_grading_string(grading_params)

    v_string = f"hex (0 1 2 3 4 5 6 7)"
    n_string = f"({n} 1 {n}) // x_wall: {x_wall} x_bulk: {x_bulk} r: {r}"
    grading_string = f"(\n  {grading}\n  1\n  {grading}\n)"

    print(v_string, n_string)
    print(grading_string)

    replace_blocks_content("system/blockMeshDict", v_string, n_string, grading_string)

        ## Update system/blockMeshDict with the grading parameters
    # Change the entries withing the blocks ( ... );

    # with open("system/blockMeshDict", "r") as f:
    #     lines = f.readlines()
    # reading_blocks = False
    # for i, line in enumerate(lines):
    #     if "blocks" in line:
    #         reading_blocks = True

    #     if reading_blocks and ");" in line:
    #         reading_blocks = False

    #     if reading_blocks:
    #         

