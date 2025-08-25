from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from qm9.ase_interface import ASEDatasetInterface
from qm9.data.dataset_class import ProcessedDataset
import os
import logging


def retrieve_dataloaders(cfg):
    # Check if ASE DB should be used
    use_ase_db = getattr(cfg, 'use_ase_db', False)
    ase_db_path = getattr(cfg, 'ase_db_path', None)
    
    if 'qm9' in cfg.dataset and use_ase_db and ase_db_path:
        # Load data from ASE database
        logging.info(f"Loading QM9 data from ASE database: {ase_db_path}")
        return retrieve_dataloaders_from_ase_db(cfg, ase_db_path)
    elif 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def retrieve_dataloaders_from_ase_db(cfg, ase_db_path):
    """
    Retrieve dataloaders from ASE database.
    
    Parameters
    ----------
    cfg : object
        Configuration object with dataset parameters
    ase_db_path : str
        Path to ASE database file
        
    Returns
    -------
    dataloaders : dict
        Dictionary of dataloaders for each split
    charge_scale : None
        Charge scale (placeholder for compatibility)
    """
    import torch
    
    if not os.path.exists(ase_db_path):
        raise FileNotFoundError(f"ASE database not found: {ase_db_path}")
    
    # Load data from ASE database
    ase_interface = ASEDatasetInterface(ase_db_path)
    molecules_data = ase_interface.load_molecules_from_db()
    ase_interface.close()
    
    if not molecules_data:
        raise ValueError("No data loaded from ASE database")
    
    # Convert units (same as NPZ version)
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}
    
    # Get species information
    all_species = []
    for split_data in molecules_data.values():
        if 'charges' in split_data:
            split_species = split_data['charges'].unique(sorted=True)
            if split_species[0] == 0:
                split_species = split_species[1:]
            all_species.append(split_species)
    
    if all_species:
        all_species = all_species[0]  # Assume all splits have same species
    else:
        all_species = torch.tensor([1, 6, 7, 8, 9])  # Default QM9 species
    
    # Create ProcessedDataset objects
    datasets = {}
    num_pts = {'train': getattr(cfg, 'num_train', -1),
               'test': getattr(cfg, 'num_test', -1), 
               'valid': getattr(cfg, 'num_valid', -1)}
    
    for split, data in molecules_data.items():
        datasets[split] = ProcessedDataset(
            data, 
            num_pts=num_pts.get(split, -1),
            included_species=all_species,
            subtract_thermo=getattr(cfg, 'subtract_thermo', False)
        )
        
        # Convert units
        datasets[split].convert_units(qm9_to_eV)
    
    # Apply atom filtering if specified
    filter_n_atoms = getattr(cfg, 'filter_n_atoms', None)
    if filter_n_atoms is not None:
        logging.info(f"Retrieving molecules with only {filter_n_atoms} atoms")
        datasets = filter_atoms(datasets, filter_n_atoms)
    
    # Create dataloaders
    batch_size = getattr(cfg, 'batch_size', 32)
    num_workers = getattr(cfg, 'num_workers', 1)
    include_charges = getattr(cfg, 'include_charges', True)
    
    preprocess = PreprocessQM9(load_charges=include_charges)
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=preprocess.collate_fn
        ) for split, dataset in datasets.items()
    }
    
    logging.info(f"Successfully loaded dataloaders from ASE database: {ase_db_path}")
    charge_scale = None  # Placeholder for compatibility
    
    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets