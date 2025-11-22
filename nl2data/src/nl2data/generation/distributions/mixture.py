"""Mixture distribution sampler."""

import numpy as np
from typing import Any, Dict, Optional
from .base import BaseSampler
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def _normalize_weights(weights: list[float]) -> list[float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("Total weight must be positive")
    return [w / total for w in weights]


def _create_component_samplers(components: list[Any], **kwargs: Any) -> list[BaseSampler]:
    """Create samplers for each component distribution."""
    # Lazy import to avoid circular dependency
    from nl2data.generation.distributions.factory import get_sampler
    
    samplers = []
    for comp in components:
        sampler = get_sampler(comp.distribution, **kwargs)
        samplers.append(sampler)
    return samplers


def _evaluate_condition(condition: Optional[Dict[str, Any]], context: Optional[Dict[str, np.ndarray]] = None) -> Optional[np.ndarray]:
    """
    Evaluate a condition against context data.
    
    Args:
        condition: Condition dict with format like {"column": "category", "op": "in", "value": ["electronics", "travel"]}
        context: Dictionary of column name -> array of values for current samples
                 Note: For fact table columns that depend on dimension lookups, context must be provided
                 after dimension joins. Currently, this requires architectural changes to pass context
                 during fact column sampling.
    
    Returns:
        Boolean array indicating which samples match the condition, or None if condition is None
    """
    if condition is None:
        return None
    
    if context is None:
        # No context available, can't evaluate condition
        logger.warning("Condition specified but no context provided. Using unconditional sampling.")
        return None
    
    # Extract condition components
    col_name = condition.get("column")
    op = condition.get("op", "in")
    value = condition.get("value")
    
    if col_name is None or col_name not in context:
        logger.warning(f"Condition column '{col_name}' not found in context. Using unconditional sampling.")
        return None
    
    col_values = context[col_name]
    
    # Evaluate condition based on operator
    if op == "in":
        if not isinstance(value, list):
            value = [value]
        mask = np.isin(col_values, value)
    elif op == "eq":
        mask = col_values == value
    elif op == "ne":
        mask = col_values != value
    else:
        logger.warning(f"Unsupported condition operator '{op}'. Using unconditional sampling.")
        return None
    
    return mask


class MixtureSampler(BaseSampler):
    """Mixture distribution sampler (multi-modal distributions)."""

    def __init__(self, components: list[Any]):
        """
        Initialize mixture sampler.

        Args:
            components: List of DistMixtureComponent objects
        """
        if not components:
            raise ValueError("Mixture must have at least one component")
        
        # Extract and normalize weights
        weights = [comp.weight for comp in components]
        self.normalized_weights = _normalize_weights(weights)
        self.components = components
        
        # Check if any components have conditions
        self.has_conditions = any(comp.condition is not None for comp in components)
        
        logger.debug(
            f"Initialized MixtureSampler with {len(self.components)} components, "
            f"weights: {self.normalized_weights}, "
            f"has_conditions: {self.has_conditions}"
        )

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """
        Generate n samples from the mixture distribution.

        For conditional mixtures:
        1. Evaluate conditions for each component
        2. Select component based on condition matches (with fallback to weights)
        3. Sample from selected component

        For unconditional mixtures:
        1. Select component based on weights
        2. Sample from selected component
        """
        rng = kwargs.get("rng", np.random.default_rng())
        context = kwargs.get("context")  # Optional context dict with column arrays
        
        # Create samplers for each component
        samplers = _create_component_samplers(self.components, **kwargs)
        
        if self.has_conditions and context is not None:
            # Conditional mixture: evaluate conditions per sample
            samples = np.zeros(n, dtype=np.float64)
            component_masks = {}
            
            # Evaluate conditions for each component
            for i, comp in enumerate(self.components):
                if comp.condition is not None:
                    mask = _evaluate_condition(comp.condition, context)
                    if mask is not None:
                        component_masks[i] = mask
            
            # Assign samples to components based on conditions
            # For samples that don't match any condition, use weight-based selection
            assigned = np.zeros(n, dtype=bool)
            component_indices = np.zeros(n, dtype=int)
            
            # First, assign samples that match conditions
            for i, comp in enumerate(self.components):
                if i in component_masks:
                    mask = component_masks[i] & ~assigned
                    component_indices[mask] = i
                    assigned[mask] = True
            
            # For unassigned samples, use weight-based selection
            unassigned_mask = ~assigned
            if np.any(unassigned_mask):
                unassigned_count = np.sum(unassigned_mask)
                component_indices[unassigned_mask] = rng.choice(
                    len(self.components),
                    size=unassigned_count,
                    p=self.normalized_weights
                )
            
            # Sample from each component
            for i, sampler in enumerate(samplers):
                mask = component_indices == i
                count = np.sum(mask)
                if count > 0:
                    samples[mask] = sampler.sample(count, **kwargs)
        else:
            # Unconditional mixture: use weight-based selection
            component_indices = rng.choice(
                len(self.components),
                size=n,
                p=self.normalized_weights
            )
            
            # Sample from each component
            samples = np.zeros(n, dtype=np.float64)
            for i, sampler in enumerate(samplers):
                mask = component_indices == i
                count = np.sum(mask)
                if count > 0:
                    samples[mask] = sampler.sample(count, **kwargs)
        
        return samples

