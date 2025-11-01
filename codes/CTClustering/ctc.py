import json
import numpy as np
import os

from .clustering import CTC
def run_ctc_from_config(data,config_path='config.json'):
    """
    Initializes and runs CTC clustering by loading parameters from a JSON config file.
    The configured and fitted CTC instance is returned for further inspection.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        CTC: The configured and fitted instance of the CTC class, or None if an error occurs.
    """
    # --- 1. Load JSON Configuration File ---
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Successfully loaded configuration from {config_path}.")
    except Exception as e:
        print(f"Error: Could not load or parse the config file {config_path}. {e}")
        return None

    # --- 2. Prepare parameters ---
    init_params = config.get('init_params', {})

    # --- 3. Initialize CTC Class ---
    print("\nInitializing CTC class with the following parameters:")
    print(json.dumps(init_params, indent=4))
    ctc_instance = CTC(**init_params)

    # --- 4. Prepare and Run the fit_predict Method ---
    # Consolidate all parameter dictionaries for the fit_predict call
    fit_predict_args = {
        "data": data,
        **config.get('fit_predict_main_params', {}),
        "px_estimator_params": config.get('px_estimator_params'),
        "pxy_estimator_params": config.get('pxy_estimator_params'),
        "px_estimate_params": config.get('px_estimate_params'),
        "valley_finding_params": config.get('valley_finding_params'),
        "cal_transition_mat_params": config.get('cal_transition_mat_params'),
        "merging_params": config.get('merging_params')
    }
    
    print("\nCalling fit_predict method with the following combined parameters:")
    printable_args = {k: v for k, v in fit_predict_args.items() if k != 'data'}
    print(json.dumps(printable_args, indent=4))
    
    # Call the modified fit_predict method
    ctc_instance.fit_predict(**fit_predict_args)
    labels = ctc_instance.labels
    print(f"\nfit_predict execution finished. Found {len(np.unique(labels))-1} clusters.")
    print(f"Sample labels: {labels[:20]}")

    print("\nScript execution complete. Returning CTC instance.")
    return ctc_instance
