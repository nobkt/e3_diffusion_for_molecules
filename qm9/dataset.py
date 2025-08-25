from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
import os
import numpy as np


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
        for key, data_list in zip(['train', 'valid', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
    elif 'ase' in cfg.dataset:
        import build_ase_dataset
        from configs.datasets_config import get_dataset_info
        
        # Get dataset configuration
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)
        
        # Load ASE database file
        db_file = getattr(cfg, 'ase_db_file', './data/ase/molecules.db')
        max_entries = getattr(cfg, 'ase_max_entries', None)
        
        # Load and split data
        split_data = build_ase_dataset.load_split_data(
            db_file,
            val_proportion=0.1,
            test_proportion=0.1,
            filter_size=getattr(cfg, 'filter_molecule_size', None),
            max_entries=max_entries
        )
        
        # Compute n_nodes histogram from the actual data
        all_data = split_data[0] + split_data[1] + split_data[2]  # train + val + test
        n_nodes_counts = {}
        for mol_data in all_data:
            n_atoms = mol_data['geometry'].shape[0]
            n_nodes_counts[n_atoms] = n_nodes_counts.get(n_atoms, 0) + 1
        
        # Update dataset_info with computed histogram
        dataset_info = dataset_info.copy()  # Make a copy to avoid modifying the original
        dataset_info['n_nodes'] = n_nodes_counts
        print(f"Computed n_nodes distribution for ASE dataset: {n_nodes_counts}")
        
        # Compute position statistics to help with normalization factor selection
        position_spans = []
        for mol_data in all_data:
            positions = mol_data['geometry'][:, 1:]  # Last 3 columns are positions
            if positions.shape[0] > 1:  # Only for molecules with multiple atoms
                mol_span = np.max(positions) - np.min(positions)
                position_spans.append(mol_span)
        
        if position_spans:
            median_span = np.median(position_spans)
            max_span = np.max(position_spans)
            print(f"ASE dataset position analysis: median_span={median_span:.2f}Å, max_span={max_span:.2f}Å")
            
            # Store statistics for potential automatic normalization factor adjustment
            dataset_info['position_stats'] = {
                'median_span': median_span,
                'max_span': max_span,
                'spans': position_spans
            }
        
        # Create transform
        transform = build_ase_dataset.ASETransform(
            dataset_info,
            cfg.include_charges,
            cfg.device,
            cfg.sequential
        )
        
        # Create dataloaders
        dataloaders = {}
        for key, data_list in zip(['train', 'valid', 'test'], split_data):
            dataset = build_ase_dataset.ASEDataset(data_list, transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            dataloaders[key] = build_ase_dataset.ASEDataLoader(
                sequential=cfg.sequential, 
                dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle
            )
        
        # Apply unit conversion like QM9 datasets
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        
        for split_name, dataloader in dataloaders.items():
            if hasattr(dataloader.dataset, 'convert_units'):
                print(f"Converting units for {split_name} split")
                dataloader.dataset.convert_units(qm9_to_eV)
        
        del split_data
        
        # Return charge_scale consistent with QM9 - use atomic number of heaviest atom typically found
        # This ensures consistent behavior with QM9 charge scaling
        charge_scale = 4.0  # Similar to max atomic number in QM9 (F=9, but we normalize)
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