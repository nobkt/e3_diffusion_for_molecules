"""
Unit tests for symmetry analyzer

Tests SymmetryAnalyzer class following the "no fallback heuristics" principle.
All tests verify strict validation and correct symmetry analysis.
"""

import pytest
import torch
import numpy as np
import warnings
from crystal.evaluation import SymmetryAnalyzer


@pytest.fixture
def cubic_crystal():
    """Create a simple cubic crystal for testing"""
    # Simple cubic lattice
    cell = torch.eye(3) * 5.0
    
    # Positions at corners of cube
    positions_cart = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [2.5, 2.5, 0.0],
        [0.0, 0.0, 2.5],
        [2.5, 0.0, 2.5],
        [0.0, 2.5, 2.5],
        [2.5, 2.5, 2.5],
    ])
    
    positions_frac = positions_cart / 5.0
    
    # Atom types (all carbon)
    atom_types = torch.ones(8, dtype=torch.long) * 6  # Carbon
    
    return {
        'positions_cart': positions_cart,
        'positions_frac': positions_frac,
        'cell': cell,
        'atom_types': atom_types,
        'pbc': torch.ones(3, dtype=torch.bool),
    }


@pytest.fixture
def orthorhombic_crystal():
    """Create an orthorhombic crystal for testing"""
    # Orthorhombic lattice (different lengths)
    cell = torch.tensor([
        [4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 6.0],
    ])
    
    positions_frac = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    
    atom_types = torch.ones(2, dtype=torch.long) * 6
    
    return {
        'positions_frac': positions_frac,
        'cell': cell,
        'atom_types': atom_types,
        'pbc': torch.ones(3, dtype=torch.bool),
    }


class TestSymmetryAnalyzerInitialization:
    """Test SymmetryAnalyzer initialization"""
    
    def test_valid_initialization_default(self):
        """Test initialization with default parameters"""
        analyzer = SymmetryAnalyzer()
        assert analyzer.symprec == 1e-3
        assert analyzer.angle_tolerance == 5.0
    
    def test_valid_initialization_custom(self):
        """Test initialization with custom parameters"""
        analyzer = SymmetryAnalyzer(symprec=1e-4, angle_tolerance=3.0)
        assert analyzer.symprec == 1e-4
        assert analyzer.angle_tolerance == 3.0
    
    def test_negative_symprec_raises_error(self):
        """Test that negative symprec raises ValueError"""
        with pytest.raises(ValueError, match="symprec must be positive"):
            SymmetryAnalyzer(symprec=-1e-3)
    
    def test_negative_angle_tolerance_raises_error(self):
        """Test that negative angle_tolerance raises ValueError"""
        with pytest.raises(ValueError, match="angle_tolerance must be positive"):
            SymmetryAnalyzer(angle_tolerance=-5.0)
    
    def test_spglib_availability_check(self):
        """Test that spglib availability is checked"""
        analyzer = SymmetryAnalyzer()
        # has_spglib should be set based on availability
        assert isinstance(analyzer.has_spglib, bool)
        
        if analyzer.has_spglib:
            assert analyzer.spglib is not None
        else:
            assert analyzer.spglib is None


class TestDetectSpaceGroup:
    """Test space group detection"""
    
    def test_detect_space_group_without_spglib(self, cubic_crystal):
        """Test space group detection returns None without spglib"""
        analyzer = SymmetryAnalyzer()
        
        if not analyzer.has_spglib:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = analyzer.detect_space_group(cubic_crystal)
                
                assert result is None
                # At least one warning about spglib not available
                assert any("not available" in str(warning.message) for warning in w)
    
    @pytest.mark.skipif(
        not hasattr(SymmetryAnalyzer(), 'has_spglib') or not SymmetryAnalyzer().has_spglib,
        reason="spglib not available"
    )
    def test_detect_space_group_with_spglib(self, cubic_crystal):
        """Test space group detection with spglib (if available)"""
        analyzer = SymmetryAnalyzer()
        result = analyzer.detect_space_group(cubic_crystal)
        
        if result is not None:
            # Check that result has expected keys
            assert 'space_group_number' in result
            assert 'space_group_symbol' in result
            assert 'point_group' in result
            assert 'crystal_system' in result
            assert 'hall_number' in result
            
            # Check that space group number is valid
            assert 1 <= result['space_group_number'] <= 230
    
    def test_missing_cell_raises_error(self, cubic_crystal):
        """Test that missing cell raises ValueError"""
        crystal = cubic_crystal.copy()
        del crystal['cell']
        
        analyzer = SymmetryAnalyzer()
        
        # Only raises if spglib is available, otherwise just returns None
        if analyzer.has_spglib:
            with pytest.raises(ValueError, match="missing 'cell' field"):
                analyzer.detect_space_group(crystal)
        else:
            # Without spglib, just returns None
            result = analyzer.detect_space_group(crystal)
            assert result is None
    
    def test_missing_atom_types_raises_error(self, cubic_crystal):
        """Test that missing atom_types raises ValueError"""
        crystal = cubic_crystal.copy()
        del crystal['atom_types']
        
        analyzer = SymmetryAnalyzer()
        
        # Only raises if spglib is available, otherwise just returns None
        if analyzer.has_spglib:
            with pytest.raises(ValueError, match="missing 'atom_types' field"):
                analyzer.detect_space_group(crystal)
        else:
            result = analyzer.detect_space_group(crystal)
            assert result is None
    
    def test_missing_positions_raises_error(self):
        """Test that missing both position types raises ValueError"""
        crystal = {
            'cell': torch.eye(3) * 5.0,
            'atom_types': torch.ones(2, dtype=torch.long),
        }
        
        analyzer = SymmetryAnalyzer()
        
        # Only raises if spglib is available, otherwise just returns None
        if analyzer.has_spglib:
            with pytest.raises(ValueError, match="missing position data"):
                analyzer.detect_space_group(crystal)
        else:
            result = analyzer.detect_space_group(crystal)
            assert result is None


class TestStructureFingerprint:
    """Test structure fingerprinting"""
    
    def test_compute_fingerprint_basic(self, cubic_crystal):
        """Test basic fingerprint computation"""
        analyzer = SymmetryAnalyzer()
        fingerprint = analyzer.compute_structure_fingerprint(cubic_crystal)
        
        # Check shape and properties
        assert fingerprint.shape == (100,)  # Default n_bins
        assert torch.all(fingerprint >= 0)
        assert torch.isclose(fingerprint.sum(), torch.tensor(1.0), atol=1e-5)  # Normalized
    
    def test_compute_fingerprint_custom_bins(self, cubic_crystal):
        """Test fingerprint with custom number of bins"""
        analyzer = SymmetryAnalyzer()
        fingerprint = analyzer.compute_structure_fingerprint(
            cubic_crystal, n_bins=50, r_max=8.0
        )
        
        assert fingerprint.shape == (50,)
    
    def test_invalid_n_bins_raises_error(self, cubic_crystal):
        """Test that invalid n_bins raises ValueError"""
        analyzer = SymmetryAnalyzer()
        
        with pytest.raises(ValueError, match="n_bins must be positive"):
            analyzer.compute_structure_fingerprint(cubic_crystal, n_bins=0)
    
    def test_invalid_r_max_raises_error(self, cubic_crystal):
        """Test that invalid r_max raises ValueError"""
        analyzer = SymmetryAnalyzer()
        
        with pytest.raises(ValueError, match="r_max must be positive"):
            analyzer.compute_structure_fingerprint(cubic_crystal, r_max=-1.0)
    
    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValueError"""
        crystal = {'positions_cart': torch.randn(5, 3)}
        analyzer = SymmetryAnalyzer()
        
        with pytest.raises(ValueError, match="missing required field"):
            analyzer.compute_structure_fingerprint(crystal)
    
    def test_no_distances_within_rmax_warning(self):
        """Test warning when no distances within r_max"""
        # Crystal with very small r_max
        crystal = {
            'positions_cart': torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            'cell': torch.eye(3) * 20.0,
            'pbc': torch.ones(3, dtype=torch.bool),
        }
        
        analyzer = SymmetryAnalyzer()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fingerprint = analyzer.compute_structure_fingerprint(crystal, r_max=1.0)
            
            assert len(w) > 0
            assert "No distances found" in str(w[-1].message)
            assert torch.all(fingerprint == 0)
    
    def test_identical_structures_same_fingerprint(self, cubic_crystal):
        """Test that identical structures produce same fingerprint"""
        analyzer = SymmetryAnalyzer()
        
        fp1 = analyzer.compute_structure_fingerprint(cubic_crystal)
        fp2 = analyzer.compute_structure_fingerprint(cubic_crystal)
        
        assert torch.allclose(fp1, fp2)


class TestCompareFingerprints:
    """Test fingerprint comparison"""
    
    def test_compare_identical_fingerprints_l2(self):
        """Test L2 distance for identical fingerprints"""
        analyzer = SymmetryAnalyzer()
        
        fp = torch.randn(100)
        fp = fp / fp.sum()  # Normalize
        
        distance = analyzer.compare_fingerprints(fp, fp, metric='l2')
        assert distance < 1e-6
    
    def test_compare_identical_fingerprints_l1(self):
        """Test L1 distance for identical fingerprints"""
        analyzer = SymmetryAnalyzer()
        
        fp = torch.randn(100)
        fp = fp / fp.sum()
        
        distance = analyzer.compare_fingerprints(fp, fp, metric='l1')
        assert distance < 1e-6
    
    def test_compare_identical_fingerprints_cosine(self):
        """Test cosine similarity for identical fingerprints"""
        analyzer = SymmetryAnalyzer()
        
        fp = torch.randn(100)
        fp = fp / fp.sum()
        
        similarity = analyzer.compare_fingerprints(fp, fp, metric='cosine')
        assert abs(similarity - 1.0) < 1e-6
    
    def test_compare_different_fingerprints(self):
        """Test comparison of different fingerprints"""
        analyzer = SymmetryAnalyzer()
        
        fp1 = torch.randn(100).abs()
        fp1 = fp1 / fp1.sum()
        
        fp2 = torch.randn(100).abs()
        fp2 = fp2 / fp2.sum()
        
        distance = analyzer.compare_fingerprints(fp1, fp2, metric='l2')
        assert distance > 0
    
    def test_mismatched_shapes_raise_error(self):
        """Test that mismatched fingerprint shapes raise ValueError"""
        analyzer = SymmetryAnalyzer()
        
        fp1 = torch.randn(100)
        fp2 = torch.randn(50)
        
        with pytest.raises(ValueError, match="must have same shape"):
            analyzer.compare_fingerprints(fp1, fp2)
    
    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError"""
        analyzer = SymmetryAnalyzer()
        
        fp = torch.randn(100)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            analyzer.compare_fingerprints(fp, fp, metric='invalid')
    
    def test_zero_norm_fingerprints_cosine(self):
        """Test cosine similarity with zero norm fingerprints"""
        analyzer = SymmetryAnalyzer()
        
        fp1 = torch.zeros(100)
        fp2 = torch.randn(100)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2, metric='cosine')
        assert similarity == 0.0


class TestAnalyzeLatticeSymmetry:
    """Test lattice symmetry analysis"""
    
    def test_cubic_lattice(self):
        """Test identification of cubic lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'cubic'
        assert result['is_orthogonal'] == True
    
    def test_tetragonal_lattice(self):
        """Test identification of tetragonal lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([5.0, 5.0, 6.0, 90.0, 90.0, 90.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'tetragonal'
        assert result['is_orthogonal'] == True
    
    def test_orthorhombic_lattice(self):
        """Test identification of orthorhombic lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0, 90.0, 90.0, 90.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'orthorhombic'
        assert result['is_orthogonal'] == True
    
    def test_hexagonal_lattice(self):
        """Test identification of hexagonal lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([5.0, 5.0, 6.0, 90.0, 90.0, 120.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'hexagonal'
        assert result['is_orthogonal'] == False
    
    def test_monoclinic_lattice(self):
        """Test identification of monoclinic lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0, 90.0, 95.0, 90.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'monoclinic'
        assert result['is_orthogonal'] == False
    
    def test_triclinic_lattice(self):
        """Test identification of triclinic lattice"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0, 85.0, 95.0, 100.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert result['lattice_type'] == 'triclinic'
        assert result['is_orthogonal'] == False
    
    def test_length_ratios_computed(self):
        """Test that length ratios are computed"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0, 90.0, 90.0, 90.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert 'length_ratios' in result
        assert 'b_over_a' in result['length_ratios']
        assert 'c_over_a' in result['length_ratios']
        assert 'c_over_b' in result['length_ratios']
        
        # Check values
        assert abs(result['length_ratios']['b_over_a'] - 1.25) < 1e-6
        assert abs(result['length_ratios']['c_over_a'] - 1.5) < 1e-6
        assert abs(result['length_ratios']['c_over_b'] - 1.2) < 1e-6
    
    def test_angles_returned(self):
        """Test that angles are returned in result"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0, 85.0, 95.0, 100.0])
        
        result = analyzer.analyze_lattice_symmetry(cell_params)
        
        assert 'angles' in result
        assert result['angles']['alpha'] == 85.0
        assert result['angles']['beta'] == 95.0
        assert result['angles']['gamma'] == 100.0
    
    def test_invalid_shape_raises_error(self):
        """Test that invalid cell_params shape raises ValueError"""
        analyzer = SymmetryAnalyzer()
        cell_params = torch.tensor([4.0, 5.0, 6.0])  # Wrong shape
        
        with pytest.raises(ValueError, match="must have shape"):
            analyzer.analyze_lattice_symmetry(cell_params)


class TestGetCrystalSystem:
    """Test _get_crystal_system helper method"""
    
    def test_triclinic_range(self):
        """Test triclinic crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(1) == 'triclinic'
        assert analyzer._get_crystal_system(2) == 'triclinic'
    
    def test_monoclinic_range(self):
        """Test monoclinic crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(3) == 'monoclinic'
        assert analyzer._get_crystal_system(10) == 'monoclinic'
        assert analyzer._get_crystal_system(15) == 'monoclinic'
    
    def test_orthorhombic_range(self):
        """Test orthorhombic crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(16) == 'orthorhombic'
        assert analyzer._get_crystal_system(50) == 'orthorhombic'
        assert analyzer._get_crystal_system(74) == 'orthorhombic'
    
    def test_tetragonal_range(self):
        """Test tetragonal crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(75) == 'tetragonal'
        assert analyzer._get_crystal_system(100) == 'tetragonal'
        assert analyzer._get_crystal_system(142) == 'tetragonal'
    
    def test_trigonal_range(self):
        """Test trigonal crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(143) == 'trigonal'
        assert analyzer._get_crystal_system(167) == 'trigonal'
    
    def test_hexagonal_range(self):
        """Test hexagonal crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(168) == 'hexagonal'
        assert analyzer._get_crystal_system(194) == 'hexagonal'
    
    def test_cubic_range(self):
        """Test cubic crystal system range"""
        analyzer = SymmetryAnalyzer()
        assert analyzer._get_crystal_system(195) == 'cubic'
        assert analyzer._get_crystal_system(220) == 'cubic'
        assert analyzer._get_crystal_system(230) == 'cubic'
    
    def test_out_of_range_raises_error(self):
        """Test that out of range space group raises ValueError"""
        analyzer = SymmetryAnalyzer()
        
        with pytest.raises(ValueError, match="must be between 1 and 230"):
            analyzer._get_crystal_system(0)
        
        with pytest.raises(ValueError, match="must be between 1 and 230"):
            analyzer._get_crystal_system(231)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
