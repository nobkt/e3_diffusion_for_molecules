from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from qm9.ase_database import ASEDatabaseReader
import os


def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
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
    elif 'ase' in cfg.dataset or hasattr(cfg, 'ase_db_path'):
        # ASE database handling
        return retrieve_ase_dataloaders(cfg)
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


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets


def retrieve_ase_dataloaders(cfg):
    """
    Retrieve dataloaders for ASE database.
    
    Args:
        cfg: Configuration object with ASE database settings
        
    Returns:
        Tuple of (dataloaders, charge_scale)
    """
    from qm9.data.dataset_class import ProcessedDataset
    
    # Get ASE database path
    if hasattr(cfg, 'ase_db_path'):
        db_path = cfg.ase_db_path
    else:
        # Try to construct from dataset name
        db_path = os.path.join(cfg.datadir, f"{cfg.dataset}.db")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ASE database not found at {db_path}")
    
    # Initialize ASE reader
    ase_reader = ASEDatabaseReader(db_path)
    
    # Get dataset info
    dataset_info = ase_reader.get_dataset_info(
        dataset_name=cfg.dataset,
        with_h=not cfg.remove_h
    )
    
    # Create splits
    data_splits = ase_reader.create_splits(
        train_ratio=getattr(cfg, 'train_ratio', 0.8),
        valid_ratio=getattr(cfg, 'valid_ratio', 0.1),
        test_ratio=getattr(cfg, 'test_ratio', 0.1),
        random_seed=getattr(cfg, 'random_seed', 42)
    )
    
    # Convert to ProcessedDataset format
    datasets = {}
    for split_name, split_data in data_splits.items():
        datasets[split_name] = ProcessedDataset(
            split_data, 
            num_pts=-1,  # Use all data
            included_species=None,  # Will be determined automatically
            subtract_thermo=False
        )
    
    # Filter by number of atoms if specified
    if hasattr(cfg, 'filter_n_atoms') and cfg.filter_n_atoms is not None:
        print("Retrieving molecules with only %d atoms" % cfg.filter_n_atoms)
        datasets = filter_atoms(datasets, cfg.filter_n_atoms)
    
    # Create dataloaders
    preprocess = PreprocessQM9(load_charges=cfg.include_charges)
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == 'train'),
            num_workers=getattr(cfg, 'num_workers', 0),
            collate_fn=preprocess.collate_fn
        )
        for split, dataset in datasets.items()
    }
    
    # No charge scaling for ASE databases by default
    charge_scale = None
    
    return dataloaders, charge_scale