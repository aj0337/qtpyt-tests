import os

import edpyt.utils.plotter as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_selected_quantities(
    data_folder="output", quantities=None, selected_impurities=None, iteration_list=None
):
    """
    Plots selected quantities from data files in each subfolder within the specified directory.

    Parameters:
        data_folder (str): Path to the main directory containing subfolders with data files.
        quantities (list): List of quantities to plot. Options are "delta", "gfloc", "sigma", "charge", "bath_energies",
                           "bath_couplings", "dft_transmission", and "dmft_transmission".
                           If None, all quantities will be plotted.
        selected_impurities (list of int, optional): List of impurity indices to plot for bath energies and bath couplings.
        iteration_list (list of int, optional): List of specific iteration numbers to plot for bath energies and bath couplings.
    """
    if quantities is None:
        # Plot all quantities if no specific selection is provided
        quantities = [
            "delta",
            "gfloc",
            "sigma",
            "charge",
            "bath_energies",
            "bath_couplings",
            "dft_transmission",
            "dmft_transmission",
        ]

    # Loop through each subfolder in the base directory
    for subfolder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, subfolder)

        if not os.path.isdir(folder_path):
            continue

        # Extract parameters from folder name
        parts = subfolder.split("_")
        nbaths = parts[1]
        U = parts[3]

        print(f"Processing folder: {subfolder} (nbaths={nbaths}, U={U})")

        # Set labels for impurities
        labels = [
            "N",
            "C(H)",
            "C(N)",
            "C(C)",
            "C(Radical)",
            "C(H)",
            "C(N)",
            "C(H)",
            "N",
        ]

        # Default plot parameters to include subfolder name as the title
        default_plot_params = {
            "title": f"{subfolder}",  # Default title set to the subfolder name
            "legend_loc": "upper right",
            "grid": True,
        }

        if "delta" in quantities:
            # Load and plot delta
            try:
                delta_file = os.path.join(folder_path, "dmft_delta.npy")
                delta = np.load(delta_file)
                beta = 1000
                ne = delta.shape[1]
                z_mats = 1.0j * (2 * np.arange(ne) + 1) * np.pi / beta
                delta_plot_params = default_plot_params.copy()
                delta_plot_params["title"] = delta_plot_params.get(
                    "title", f"{subfolder}"
                )  # Use subfolder name as title if not set
                pl.plot_delta(
                    delta, z_mats, labels=labels, plot_params=delta_plot_params
                )
            except Exception as e:
                print(f"Error plotting delta for {subfolder}: {e}")

        if "gfloc" in quantities:
            # Load and plot gfloc
            try:
                gfloc_file = os.path.join(folder_path, "dmft_gfloc.npy")
                gfloc = np.load(gfloc_file)
                eta = 3e-2
                energies = np.arange(-10, 10, 0.01)
                z_ret = energies + 1.0j * eta
                gfloc_plot_params = default_plot_params.copy()
                gfloc_plot_params.update(
                    {
                        "title": gfloc_plot_params.get(
                            "title", f"{subfolder}"
                        ),  # Use subfolder name as title if not set
                        "xlim": (-2.5, 2.5),
                    }
                )
                pl.plot_gfloc_spectral_function(
                    gfloc, z_ret.real, labels=labels, plot_params=gfloc_plot_params
                )
            except Exception as e:
                print(f"Error plotting gfloc for {subfolder}: {e}")

        if "sigma" in quantities:
            # Load and plot sigma trace
            try:
                sigma_file = os.path.join(folder_path, "dmft_sigma.npy")
                sigma = np.load(sigma_file)
                sigma_plot_params = default_plot_params.copy()
                sigma_plot_params.update(
                    {
                        "title": sigma_plot_params.get(
                            "title", f"{subfolder}"
                        ),  # Use subfolder name as title if not set
                        "xlim": (-2.5, 2.5),
                    }
                )
                pl.plot_trace_sigma(sigma, z_ret, plot_params=sigma_plot_params)
            except Exception as e:
                print(f"Error plotting sigma for {subfolder}: {e}")

        if "charge" in quantities:
            # Load and plot charge per impurity
            try:
                charge_file = os.path.join(folder_path, "charge_per_orbital_dft.npy")
                charge = np.load(charge_file)
                dmft_charge_file = os.path.join(
                    folder_path, "charge_per_orbital_dmft.npy"
                )
                dmft_charge = np.load(dmft_charge_file)
                charge_plot_params = default_plot_params.copy()
                charge_plot_params["title"] = charge_plot_params.get(
                    "title", f"{subfolder}"
                )  # Use subfolder name as title if not set
                pl.plot_charge_per_impurity(
                    [charge, dmft_charge],
                    labels=labels,
                    dataset_labels=["DFT", "DMFT"],
                    plot_params=charge_plot_params,
                )
            except Exception as e:
                print(f"Error plotting charge per impurity for {subfolder}: {e}")

        if "bath_energies" in quantities:
            # Load and plot bath energies
            try:
                h5_file = os.path.join(folder_path, "dmft_iterations.h5")
                bath_energy_plot_params = default_plot_params.copy()
                bath_energy_plot_params["title"] = bath_energy_plot_params.get(
                    "title", f"{subfolder}"
                )  # Use subfolder name as title if not set
                pl.plot_bath_energies(
                    h5_file,
                    labels=labels,
                    plot_params=bath_energy_plot_params,
                    selected_impurities=selected_impurities,
                    iteration_list=iteration_list,
                )
            except Exception as e:
                print(f"Error plotting bath energies for {subfolder}: {e}")

        if "bath_couplings" in quantities:
            # Load and plot bath couplings
            try:
                h5_file = os.path.join(folder_path, "dmft_iterations.h5")
                bath_coupling_plot_params = default_plot_params.copy()
                bath_coupling_plot_params["title"] = bath_coupling_plot_params.get(
                    "title", f"{subfolder}"
                )  # Use subfolder name as title if not set
                pl.plot_bath_couplings(
                    h5_file,
                    labels=labels,
                    plot_params=bath_coupling_plot_params,
                    selected_impurities=selected_impurities,
                    iteration_list=iteration_list,
                )
            except Exception as e:
                print(f"Error plotting bath couplings for {subfolder}: {e}")

        # Check if both DFT and DMFT transmission data are requested and available
        dft_transmission = None
        dmft_transmission = None
        energies = None

        if "dft_transmission" in quantities:
            dft_transmission_file = os.path.join(folder_path, "dft_transmission.npy")
            if os.path.exists(dft_transmission_file):
                energies, dft_transmission = np.load(dft_transmission_file)

        if "dmft_transmission" in quantities:
            dmft_transmission_file = os.path.join(folder_path, "dmft_transmission.npy")
            if os.path.exists(dmft_transmission_file):
                energies, dmft_transmission = np.load(dmft_transmission_file)

        # Set yscale to "log" only for DFT and DMFT transmission plots
        transmission_plot_params = default_plot_params.copy()
        transmission_plot_params["yscale"] = "log"
        transmission_plot_params["title"] = transmission_plot_params.get(
            "title", f"{subfolder}"
        )  # Use subfolder name as title if not set

        # Plot DFT and DMFT transmission on the same plot if both are available
        if dft_transmission is not None or dmft_transmission is not None:
            try:
                pl.plot_transmission(
                    energies,
                    dft_transmission,
                    dmft_transmission,
                    plot_params=transmission_plot_params,
                )
            except Exception as e:
                print(f"Error plotting transmission comparison for {subfolder}: {e}")


def compute_homo_lumo_gap(gfloc, energy_grid, plot_params=None, plot=False):
    """
    Computes the HOMO-LUMO gap across all impurities in gfloc by finding the highest peak below the Fermi level
    and the lowest peak above the Fermi level in the combined spectral function.

    Parameters:
        gfloc (np.ndarray): Array containing gfloc data with shape (n_impurities, n_energies).
        energy_grid (np.ndarray): Array of energy values corresponding to gfloc.
        plot_params (dict, optional): Dictionary of plot parameters to customize the plot appearance.

    Returns:
        dict: A dictionary containing the global HOMO and LUMO energy values and the HOMO-LUMO gap.
    """
    # Calculate spectral function: -(1/π) * Im(gfloc)
    spectral_function = -(1 / np.pi) * np.imag(gfloc)
    n_impurities = spectral_function.shape[0]

    # Find indices of energy points below and above the Fermi level (0)
    below_fermi = energy_grid < 0
    above_fermi = energy_grid > 0

    # Initialize variables to find the highest HOMO and lowest LUMO across all impurities
    homo_energy = -np.inf  # Initialize to a very low value
    lumo_energy = np.inf  # Initialize to a very high value

    # Loop through each impurity to find the highest peak below Fermi and lowest peak above Fermi
    for i in range(n_impurities):
        spectrum = spectral_function[i]

        # Highest peak below Fermi level for HOMO
        homo_index = np.argmax(spectrum[below_fermi])
        current_homo_energy = energy_grid[below_fermi][homo_index]
        homo_energy = max(homo_energy, current_homo_energy)

        # Lowest peak above Fermi level for LUMO
        lumo_index = np.argmax(spectrum[above_fermi]) + np.sum(below_fermi)
        current_lumo_energy = energy_grid[above_fermi][lumo_index - np.sum(below_fermi)]
        lumo_energy = min(lumo_energy, current_lumo_energy)

    # Calculate the HOMO-LUMO gap
    homo_lumo_gap = lumo_energy - homo_energy

    # Optional plot to visualize all impurities and mark HOMO and LUMO peaks
    if plot:
        if plot_params is None:
            plot_params = {}
        plt.figure(figsize=plot_params.get("figsize", (10, 6)))

        for i in range(n_impurities):
            plt.plot(energy_grid, spectral_function[i], label=f"Impurity {i}")

        plt.axvline(homo_energy, color="blue", linestyle="--", label="HOMO")
        plt.axvline(lumo_energy, color="red", linestyle="--", label="LUMO")
        plt.xlabel(plot_params.get("xlabel", "Energy"))
        plt.ylabel(plot_params.get("ylabel", r"$-\frac{1}{\pi} \mathrm{Im}(G_{loc})$"))
        plt.title(
            plot_params.get("title", "Combined Spectral Function and HOMO-LUMO Gap")
        )
        plt.legend(loc=plot_params.get("legend_loc", "upper right"))
        plt.grid(plot_params.get("grid", True))
        plt.tight_layout()
        plt.show()

    return {
        "HOMO_energy": homo_energy,
        "LUMO_energy": lumo_energy,
        "HOMO_LUMO_gap": homo_lumo_gap,
    }


def process_and_plot_homo_lumo(data_folder="output"):
    # Initialize lists to store parameter values and results
    nbaths_list = []
    U_list = []
    homo_energies = []
    lumo_energies = []
    homo_lumo_gaps = []

    # Loop through each subfolder in the base directory
    for subfolder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, subfolder)

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Extract parameters (nbaths and U) from folder name
        try:
            parts = subfolder.split("_")

            nbaths = int(parts[1])
            U = float(parts[3])
        except (IndexError, ValueError):
            print(f"Skipping folder {subfolder} with unexpected name format.")
            continue

        print(f"Processing folder: {subfolder} (nbaths={nbaths}, U={U})")

        # Load gfloc data and energy grid
        gfloc_file = os.path.join(folder_path, "dmft_gfloc.npy")
        if not os.path.exists(gfloc_file):
            print(f"No dmft_gfloc.npy file in {subfolder}, skipping...")
            continue

        gfloc = np.load(gfloc_file)
        eta = 3e-2
        energies = np.arange(-10, 10, 0.01)
        energy_grid = (
            energies + 1.0j * eta
        )  # Assuming this is the energy grid for the spectral function

        # Compute HOMO-LUMO gap using the combined method
        homo_lumo_data = compute_homo_lumo_gap(gfloc, energy_grid.real, plot=True)

        # Append parameters and results for visualization
        nbaths_list.append(nbaths)
        U_list.append(U)
        homo_energies.append(homo_lumo_data["HOMO_energy"])
        lumo_energies.append(homo_lumo_data["LUMO_energy"])
        homo_lumo_gaps.append(homo_lumo_data["HOMO_LUMO_gap"])

    # Convert lists to arrays for easier plotting
    nbaths_list = np.array(nbaths_list)
    U_list = np.array(U_list)
    homo_energies = np.array(homo_energies)
    lumo_energies = np.array(lumo_energies)
    homo_lumo_gaps = np.array(homo_lumo_gaps)

    # Plot HOMO, LUMO, and gap vs. U for each nbaths
    unique_nbaths = np.unique(nbaths_list)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle("HOMO, LUMO, and HOMO-LUMO Gap vs. U for Different nbaths")

    for nbaths in unique_nbaths:
        mask = nbaths_list == nbaths
        axs[0].plot(
            U_list[mask], homo_energies[mask], marker="o", label=f"nbaths = {nbaths}"
        )
        axs[1].plot(
            U_list[mask], lumo_energies[mask], marker="o", label=f"nbaths = {nbaths}"
        )
        axs[2].plot(
            U_list[mask], homo_lumo_gaps[mask], marker="o", label=f"nbaths = {nbaths}"
        )

    # Configure plots
    axs[0].set_ylabel("HOMO Energy")
    axs[1].set_ylabel("LUMO Energy")
    axs[2].set_ylabel("HOMO-LUMO Gap")
    axs[2].set_xlabel("U")

    for ax in axs:
        ax.legend(loc="upper left")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compute_total_charge(data_folder="output"):
    """
    Computes the total charge across all impurities for DFT and DMFT cases
    for each nbath, U, and other parameter combinations found in the specified data folder.

    Parameters:
        data_folder (str): Path to the main directory containing subfolders with data files.

    Returns:
        dict: Dictionary containing nbath, U, additional labels, and total charges for DFT and DMFT as lists.
    """
    nbath_values = []
    U_values = []
    total_charges_dft = []
    total_charges_dmft = []
    additional_labels = []

    # Define mutually exclusive keywords to look for in folder names
    additional_keywords = {
        "with_dc": "with_dc",
        "without_dc": "without_dc",
        "adjust_mu": "adjust_mu",
        "no_adjust_mu": "no_adjust_mu",
    }

    # Loop through each subfolder in the base directory
    for subfolder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue

        # Extract nbath and U values from folder name assuming the format is "nbath_<value>_U_<value>"
        try:
            parts = subfolder.split("_")
            nbaths = int(parts[1])
            U = float(parts[3])
        except (IndexError, ValueError) as e:
            print(f"Error extracting nbath or U for {subfolder}: {e}")
            continue

        # Check for additional mutually exclusive keywords in the folder name
        additional_info = []
        if "with_dc" in subfolder and "without_dc" not in subfolder:
            additional_info.append("with_dc_correction")
        elif "without_dc" in subfolder:
            additional_info.append("without_dc_correction")

        if "adjust_mu" in subfolder and "no_adjust_mu" not in subfolder:
            additional_info.append("adjust_mu")
        elif "no_adjust_mu" in subfolder:
            additional_info.append("no_adjust_mu")

        # Construct a label based on nbath, U, and any additional information
        label = f"({nbaths},{U}"
        if additional_info:
            label += ", " + ", ".join(additional_info)
        label += ")"
        additional_labels.append(label)

        # Load DFT and DMFT charge data
        try:
            charge_file = os.path.join(folder_path, "charge_per_orbital_dft.npy")
            dmft_charge_file = os.path.join(folder_path, "charge_per_orbital_dmft.npy")
            charge = np.load(charge_file)
            dmft_charge = np.load(dmft_charge_file)

            # Calculate total charge by summing over all impurities
            total_charge_dft = np.sum(charge)
            total_charge_dmft = np.sum(dmft_charge)

            # Append the data for plotting
            nbath_values.append(nbaths)
            U_values.append(U)
            total_charges_dft.append(total_charge_dft)
            total_charges_dmft.append(total_charge_dmft)

        except Exception as e:
            print(f"Error computing total charge for {subfolder}: {e}")
            continue

    return {
        "nbath_values": nbath_values,
        "U_values": U_values,
        "additional_labels": additional_labels,
        "total_charges_dft": total_charges_dft,
        "total_charges_dmft": total_charges_dmft,
    }


def plot_total_charge(total_charge_data):
    """
    Plots the total charge for DFT and DMFT as a histogram where x-axis represents
    each combination of (nbath, U) with additional parameters if available.

    Parameters:
        total_charge_data (dict): Dictionary containing nbath, U, additional labels, and total charges for DFT and DMFT.
    """
    # Prepare x-axis labels and positions
    x_labels = total_charge_data["additional_labels"]
    x = np.arange(len(x_labels))

    # Bar width and figure setup
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting DFT and DMFT bars side by side for each (nbath, U) combination
    ax.bar(
        x - width / 2,
        total_charge_data["total_charges_dft"],
        width,
        label="DFT Total Charge",
    )
    ax.bar(
        x + width / 2,
        total_charge_data["total_charges_dmft"],
        width,
        label="DMFT Total Charge",
    )

    # Customizing plot appearance
    ax.set_xlabel("(nbath, U, additional params)")
    ax.set_ylabel("Total Charge")
    ax.set_title(
        "Total Charge for DFT and DMFT Across Nbath, U, and Additional Parameters"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
