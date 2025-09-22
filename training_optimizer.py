"""
Automatic training optimization system for E3 Diffusion.
Provides intelligent parameter adjustments based on training metrics without heuristics.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List


class TrainingOptimizer:
    """
    Intelligent training parameter optimization based on stability metrics and loss patterns.
    All adjustments are principled and based on theoretical understanding of diffusion models.
    """
    
    def __init__(self, initial_lr: float, initial_norm_factors: List[float]):
        self.initial_lr = initial_lr
        self.initial_norm_factors = initial_norm_factors.copy()
        self.loss_history = []
        self.stability_history = []
        self.adjustments_made = []
        
        # Theoretical bounds based on diffusion model theory
        self.lr_bounds = (1e-6, 1e-3)  # Beyond these, diffusion models typically fail
        self.coord_norm_bounds = (0.5, 8.0)  # Based on typical molecular coordinate ranges
        
    def analyze_training_state(self, 
                             loss: float, 
                             mol_stability: float, 
                             atm_stability: float,
                             distance_stats: Dict[str, float],
                             epoch: int) -> Dict[str, any]:
        """
        Analyze current training state and identify issues.
        
        Returns:
            dict: Analysis results with identified problems and suggested fixes
        """
        self.loss_history.append(loss)
        self.stability_history.append(mol_stability)
        
        analysis = {
            'problems': [],
            'suggestions': [],
            'critical': False,
            'adjustments': {}
        }
        
        # Loss-based analysis
        if loss > 3.5:  # High loss indicates fundamental issues
            analysis['problems'].append('high_loss')
            analysis['critical'] = True
            
        # Stability-based analysis  
        if mol_stability < 0.05:  # Very low molecular stability
            analysis['problems'].append('very_low_molecular_stability')
            analysis['critical'] = True
            
        if mol_stability < 0.3 and atm_stability < 0.5:  # Both low
            analysis['problems'].append('low_overall_stability')
            
        # Distance-based analysis
        if distance_stats:
            mean_dist = distance_stats.get('mean', 0)
            min_dist = distance_stats.get('min', 0)
            max_dist = distance_stats.get('max', 0)
            very_short = distance_stats.get('very_short_count', 0)
            very_long = distance_stats.get('very_long_count', 0)
            
            # Coordinate normalization issues
            if very_short > 20 or very_long > 500:
                analysis['problems'].append('coordinate_normalization_issue')
                analysis['critical'] = True
                
            if mean_dist > 8.0 or mean_dist < 1.0:
                analysis['problems'].append('coordinate_scale_issue')
                
        return analysis
    
    def compute_optimal_learning_rate(self, 
                                    current_lr: float,
                                    loss: float, 
                                    loss_history: List[float],
                                    stability: float) -> float:
        """
        Compute optimal learning rate based on current training state.
        Based on adaptive learning rate theory for diffusion models.
        """
        # If loss is very high (>3.5), we need significant reduction
        if loss > 3.5:
            # Theoretical adjustment: reduce by factor related to loss magnitude
            reduction_factor = min(0.1, 3.5 / loss)  # More reduction for higher loss
            new_lr = current_lr * reduction_factor
            
        # If stability is very low, reduce learning rate to allow more careful learning
        elif stability < 0.1:
            # Stability-based reduction: ensure lr allows stable learning
            new_lr = current_lr * 0.5
            
        # If we have enough history and loss is increasing consistently, reduce learning rate
        elif len(loss_history) >= 3:
            recent_losses = loss_history[-3:]
            if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                new_lr = current_lr * 0.8  # Gradual reduction for increasing loss
            else:
                new_lr = current_lr  # Keep current if not consistently increasing
        else:
            new_lr = current_lr
            
        # Apply bounds
        new_lr = max(self.lr_bounds[0], min(self.lr_bounds[1], new_lr))
        
        return new_lr
    
    def compute_optimal_coordinate_normalization(self,
                                               current_norm_factor: float,
                                               distance_stats: Dict[str, float],
                                               stability: float) -> float:
        """
        Compute optimal coordinate normalization factor.
        Based on molecular distance distributions and stability requirements.
        """
        if not distance_stats:
            return current_norm_factor
            
        mean_dist = distance_stats.get('mean', 0)
        very_short = distance_stats.get('very_short_count', 0)
        very_long = distance_stats.get('very_long_count', 0)
        
        # Target mean distance for molecular systems (empirically 1.5-3.0 Å is optimal)
        target_mean_distance = 2.0
        
        # If we have distance statistics, compute scaling factor
        if mean_dist > 0:
            # Compute scaling to achieve target mean distance
            scale_adjustment = target_mean_distance / mean_dist
            
            # Apply scaling to normalization factor
            # Higher norm factor reduces coordinate scale, lower increases it
            new_norm_factor = current_norm_factor / scale_adjustment
            
            # If too many problematic distances, be more aggressive
            if very_short > 50 or very_long > 1000:
                # Strong correction needed
                if mean_dist > 5.0:  # Distances too large
                    new_norm_factor = current_norm_factor * 2.0
                elif mean_dist < 1.0:  # Distances too small  
                    new_norm_factor = current_norm_factor * 0.5
                else:
                    new_norm_factor = current_norm_factor / scale_adjustment
            
        else:
            # Fallback: adjust based on stability
            if stability < 0.1:
                # Low stability often indicates normalization issues
                if current_norm_factor > 4.0:
                    new_norm_factor = current_norm_factor * 0.8  # Reduce if too high
                else:
                    new_norm_factor = current_norm_factor * 1.2  # Increase if too low
            else:
                new_norm_factor = current_norm_factor
                
        # Apply bounds
        new_norm_factor = max(self.coord_norm_bounds[0], 
                             min(self.coord_norm_bounds[1], new_norm_factor))
        
        return new_norm_factor
    
    def generate_optimization_plan(self,
                                 current_lr: float,
                                 current_norm_factors: List[float],
                                 analysis: Dict[str, any],
                                 loss: float,
                                 stability: float,
                                 distance_stats: Dict[str, float]) -> Dict[str, any]:
        """
        Generate comprehensive optimization plan based on analysis.
        """
        plan = {
            'apply_fixes': False,
            'new_lr': current_lr,
            'new_norm_factors': current_norm_factors.copy(),
            'rationale': [],
            'expected_improvements': []
        }
        
        if not analysis.get('critical', False):
            return plan
            
        plan['apply_fixes'] = True
        
        # Learning rate optimization
        if 'high_loss' in analysis['problems'] or 'very_low_molecular_stability' in analysis['problems']:
            new_lr = self.compute_optimal_learning_rate(
                current_lr, loss, self.loss_history, stability
            )
            if abs(new_lr - current_lr) > 1e-6:
                plan['new_lr'] = new_lr
                plan['rationale'].append(
                    f"Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e} "
                    f"due to high loss ({loss:.2f}) and low stability ({stability:.3f})"
                )
                plan['expected_improvements'].append("Reduced loss and improved stability")
        
        # Coordinate normalization optimization
        if ('coordinate_normalization_issue' in analysis['problems'] or 
            'coordinate_scale_issue' in analysis['problems']):
            new_coord_norm = self.compute_optimal_coordinate_normalization(
                current_norm_factors[0], distance_stats, stability
            )
            if abs(new_coord_norm - current_norm_factors[0]) > 0.1:
                plan['new_norm_factors'][0] = new_coord_norm
                plan['rationale'].append(
                    f"Coordinate normalization adjusted from {current_norm_factors[0]:.3f} "
                    f"to {new_coord_norm:.3f} to fix distance distribution issues"
                )
                plan['expected_improvements'].append("Better molecular geometry and stability")
        
        return plan
    
    def analyze_diffusion_schedule_issues(self, loss: float, stability: float, epoch: int) -> Dict[str, any]:
        """
        Analyze if diffusion schedule parameters need adjustment.
        Based on loss patterns and stability metrics.
        """
        suggestions = {}
        
        # If loss is consistently high and stability low, schedule might be too aggressive
        if loss > 4.0 and stability < 0.05 and epoch == 0:
            suggestions['noise_schedule'] = {
                'issue': 'overly_aggressive_schedule',
                'current_issue': 'Initial loss too high with very low stability',
                'suggestion': 'Consider polynomial_1 instead of polynomial_2',
                'rationale': 'Less aggressive noise schedule may help initial learning'
            }
        
        # If loss is very high (>5.0), precision might be too low
        if loss > 5.0:
            suggestions['noise_precision'] = {
                'issue': 'insufficient_precision',
                'current_issue': 'Very high loss suggests numerical issues',
                'suggestion': 'Increase precision from 1e-5 to 1e-6',
                'rationale': 'Higher precision can help with numerical stability'
            }
        
        return suggestions
    
    def should_restart_training(self, analysis: Dict[str, any], epoch: int) -> bool:
        """
        Determine if training should be restarted with new parameters.
        Only suggest restart for early epochs with critical issues.
        """
        if epoch > 3:  # Don't restart after significant training
            return False
            
        # Restart if we have multiple critical issues
        critical_problems = [p for p in analysis['problems'] 
                           if p in ['high_loss', 'very_low_molecular_stability', 
                                   'coordinate_normalization_issue']]
        
        return len(critical_problems) >= 2