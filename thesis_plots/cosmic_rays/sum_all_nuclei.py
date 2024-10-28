import numpy as np
import logging
from thesis_plots.cosmic_rays.extract_crdb import write_data_to_file

# Configure logging
logger = logging.getLogger(__name__)


def get_interpolated(filename, E):
    """Interpolates log(I) vs log(E) from a file, returns interpolated values for E."""
    try:
        E_data, I_data = np.loadtxt(filename, unpack=True, usecols=(0, 1))
    except IOError as e:
        logger.error(f"Error loading data from {filename}: {e}")
        raise RuntimeError(f"Failed to load data from {filename}") from e

    try:
        # Perform interpolation in the log-log space
        lgI = np.interp(np.log(E), np.log(E_data), np.log(I_data), left=-100, right=-100)
    except Exception as e:
        logger.error(f"Error interpolating data from {filename}: {e}")
        raise RuntimeError(f"Interpolation failed for {filename}") from e
    
    return np.exp(lgI)


def sum_all_nuclei(E, file_list):
    """
    Sums the interpolated flux values from a list of files for energy E.
    
    Parameters:
    - E: array of energies for which to interpolate.
    - file_list: list of file paths containing energy and flux data.
    
    Returns:
    - Sum of interpolated flux values for all nuclei.
    """
    total_flux = np.zeros_like(E)
    
    for filename in file_list:
        try:
            interpolated_flux = get_interpolated(filename, E)
            total_flux += interpolated_flux
        except RuntimeError as e:
            logger.error(f"Failed to process file {filename}: {e}")
            raise

    return total_flux


def process_experiment(expName, file_list, filename, E):
    """General function to process an experiment and write the output to a file."""
    try:
        # Sum the flux for all nuclei from the list of files
        I = sum_all_nuclei(E, file_list)

        # Define metadata for file writing
        quantity = 'AllParticles'
        energyType = 'ETOT'

        # Prepare data for file writing
        header = (f'#source: CRDB\n'
                  f'#Quantity: {quantity}\n'
                  f'#EnergyType: {energyType}\n'
                  f'#Experiment: {expName}\n'
                  f'#E - y\n')

        data_lines = [f'{eBin:10.5e} {value:10.5e}\n' for eBin, value in zip(E, I)]

        # Write data to file
        write_data_to_file(filename, header, data_lines)
    
    except Exception as e:
        logger.error(f"An error occurred during processing {expName}: {e}")
        raise


def main(cache):
    # Define the energy range
    
    # AMS-02 data files
    ams02_files = [
        'AMS-02_H_energy.txt',
        'AMS-02_He_energy.txt',
        'AMS-02_Li_energy.txt',
        'AMS-02_Be_energy.txt',
        'AMS-02_B_energy.txt',
        'AMS-02_C_energy.txt',
        'AMS-02_N_energy.txt',
        'AMS-02_O_energy.txt',
        'AMS-02_F_energy.txt',
        'AMS-02_Ne_energy.txt',
        'AMS-02_Na_energy.txt',
        'AMS-02_Mg_energy.txt',
        'AMS-02_Si_energy.txt',
        'AMS-02_S_energy.txt',
        'AMS-02_Fe_energy.txt',
    ]
    
    # CREAM data files
    cream_files = [
        'CREAM_H_energy.txt',
        'CREAM_He_energy.txt',
        'CREAM_C_energy.txt',
        'CREAM_N_energy.txt',
        'CREAM_O_energy.txt',
        'CREAM_Ne_energy.txt',
        'CREAM_Mg_energy.txt',
        'CREAM_Si_energy.txt',
        'CREAM_Fe_energy.txt',
    ]
    
    # Process AMS-02 experiment
    E = np.logspace(0.9, 3.1, 1000)
    _ams02_files = [cache.cache_dir / f for f in ams02_files]
    process_experiment('AMS02', _ams02_files, cache.cache_dir / 'AMS-02_allParticles_energy.txt', E)
    
    # Process CREAM experiment
    E = np.logspace(3.2, 4.8, 1000)
    _cream_files = [cache.cache_dir / f for f in cream_files]
    process_experiment('CREAM', _cream_files, cache.cache_dir / 'CREAM_allParticles_energy.txt', E)


if __name__ == '__main__':
    main()
