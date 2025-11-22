"""Event effect evaluation and application."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from nl2data.ir.generation import EventSpec, EventEffect, GenerationIR
from nl2data.ir.logical import TableSpec
from nl2data.ir.dataset import DatasetIR
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def parse_event_time(time_str: str, total_rows: int, time_column: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Parse event time string to determine which rows are affected.
    
    Supports:
    - ISO datetime strings: "2024-01-15T00:00:00Z"
    - Relative percentages: "50%" (middle of dataset)
    - Relative row numbers: "1000" (row 1000)
    
    Args:
        time_str: Time specification string
        total_rows: Total number of rows in the dataset
        time_column: Optional column name for time-based matching
        df: Optional DataFrame for time-based matching
    
    Returns:
        Boolean Series indicating which rows are affected (if time_column provided)
        or integer row index (if relative/absolute)
    """
    time_str = time_str.strip()
    
    # Check for percentage
    if time_str.endswith('%'):
        pct = float(time_str.rstrip('%')) / 100.0
        row_idx = int(total_rows * pct)
        return row_idx
    
    # Check for absolute row number
    try:
        row_idx = int(time_str)
        return row_idx
    except ValueError:
        pass
    
    # Try to parse as ISO datetime
    try:
        # Remove timezone info if present
        dt_str = time_str.replace('Z', '+00:00')
        dt = datetime.fromisoformat(dt_str)
        # If we have a time column, match against it
        if time_column and df is not None and time_column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[time_column]):
                return df[time_column] >= dt
            else:
                logger.warning(f"Time column '{time_column}' is not datetime type. Cannot match event time.")
                return pd.Series([False] * len(df), index=df.index)
        else:
            # No time column - can't match, return None
            logger.warning(f"Event time '{time_str}' is datetime but no time column available for matching.")
            return None
    except ValueError:
        logger.warning(f"Unable to parse event time '{time_str}'. Expected ISO datetime, percentage, or row number.")
        return None


def get_active_events(
    events: List[EventSpec],
    row_indices: pd.Index,
    time_column: Optional[str] = None,
    df: Optional[pd.DataFrame] = None
) -> List[EventSpec]:
    """
    Get events that are active for the given row indices.
    
    Args:
        events: List of event specifications
        row_indices: Row indices to check
        time_column: Optional time column name for time-based events
        df: Optional DataFrame for time-based matching
    
    Returns:
        List of active events
    """
    active = []
    total_rows = len(row_indices) if df is None else len(df)
    
    for event in events:
        # Parse start time
        start_match = parse_event_time(event.start_time, total_rows, time_column, df)
        
        if start_match is None:
            continue
        
        # Check if event has ended
        if event.end_time:
            end_match = parse_event_time(event.end_time, total_rows, time_column, df)
            if end_match is None:
                continue
            
            # For time-based matching, check if any rows fall within the range
            if isinstance(start_match, pd.Series) and isinstance(end_match, pd.Series):
                if (start_match & ~end_match).any():
                    active.append(event)
            elif isinstance(start_match, int) and isinstance(end_match, int):
                # Row-based range
                if any(start_match <= idx < end_match for idx in row_indices):
                    active.append(event)
        else:
            # No end time - event continues indefinitely from start
            if isinstance(start_match, pd.Series):
                if start_match.any():
                    active.append(event)
            elif isinstance(start_match, int):
                if any(idx >= start_match for idx in row_indices):
                    active.append(event)
    
    return active


def apply_event_effect(
    df: pd.DataFrame,
    effect: EventEffect,
    event: EventSpec,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Apply a single event effect to a DataFrame.
    
    Args:
        df: DataFrame to modify
        effect: Event effect specification
        event: Parent event specification (for context)
        rng: Optional random number generator
    
    Returns:
        Modified DataFrame
    """
    result_df = df.copy()
    
    # Determine which rows are affected
    # For now, apply to all rows in the DataFrame
    # In the future, we could add "when" conditions to filter rows
    affected_mask = pd.Series([True] * len(df), index=df.index)
    
    # If column is specified, only modify that column
    if effect.column and effect.column in result_df.columns:
        col = effect.column
        original_values = result_df[col].copy()
        
        if effect.effect_type == "multiply_distribution":
            # Multiply values by the effect value
            multiplier = float(effect.value) if isinstance(effect.value, (int, float, str)) else 1.0
            try:
                multiplier = float(effect.value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid multiplier value for effect: {effect.value}. Using 1.0.")
                multiplier = 1.0
            
            result_df.loc[affected_mask, col] = original_values[affected_mask] * multiplier
            
        elif effect.effect_type == "add_offset":
            # Add offset to values
            offset = float(effect.value) if isinstance(effect.value, (int, float, str)) else 0.0
            try:
                offset = float(effect.value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid offset value for effect: {effect.value}. Using 0.0.")
                offset = 0.0
            
            result_df.loc[affected_mask, col] = original_values[affected_mask] + offset
            
        elif effect.effect_type == "set_value":
            # Set values to a specific value (with optional probability)
            if isinstance(effect.value, dict):
                # Complex value specification: {"value": X, "probability": 0.35}
                set_value = effect.value.get("value", effect.value)
                probability = effect.value.get("probability", 1.0)
                
                if rng is None:
                    logger.warning("set_value with probability requires RNG. Applying to all rows.")
                    probability = 1.0
                
                if probability < 1.0:
                    # Apply with probability
                    mask = affected_mask & (rng.random(len(df)) < probability)
                    result_df.loc[mask, col] = set_value
                else:
                    result_df.loc[affected_mask, col] = set_value
            else:
                # Simple value
                result_df.loc[affected_mask, col] = effect.value
            
        elif effect.effect_type == "change_distribution":
            # Change the distribution (future enhancement - would need to re-sample)
            logger.warning("change_distribution effect type not yet implemented. Skipping.")
        else:
            logger.warning(f"Unknown effect type: {effect.effect_type}. Skipping.")
    else:
        if effect.column:
            logger.warning(f"Event effect targets column '{effect.column}' which does not exist in DataFrame. Available columns: {list(result_df.columns)}")
        else:
            logger.warning("Event effect has no column specified. Table-wide effects not yet implemented.")
    
    return result_df


def apply_events_to_chunk(
    df: pd.DataFrame,
    table: TableSpec,
    ir: DatasetIR,
    chunk_start_idx: int,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Apply all active events to a DataFrame chunk.
    
    Args:
        df: DataFrame chunk to modify
        table: Table specification
        ir: Dataset IR containing events
        chunk_start_idx: Starting row index for this chunk (for relative time matching)
        rng: Optional random number generator
    
    Returns:
        Modified DataFrame
    """
    if not ir.generation.events:
        return df
    
    result_df = df.copy()
    
    # Find time column (if any) for time-based event matching
    time_column = None
    for col in table.columns:
        if col.sql_type in ('TIMESTAMP', 'DATETIME', 'DATE') or 'time' in col.name.lower() or 'date' in col.name.lower():
            time_column = col.name
            break
    
    # Get active events for this chunk
    # For now, we'll check all events and apply them if they match
    # In a more sophisticated implementation, we'd parse the event time ranges
    # and only apply events that are active for this chunk's time range
    
    for event in ir.generation.events:
        # Check if event is active for this chunk
        # Simple heuristic: if event has effects for this table, apply them
        table_effects = [e for e in event.effects if e.table == table.name]
        
        if table_effects:
            # Apply all effects for this table
            for effect in table_effects:
                try:
                    result_df = apply_event_effect(result_df, effect, event, rng=rng)
                    logger.debug(
                        f"Applied event '{event.name}' effect to table '{table.name}' "
                        f"(effect type: {effect.effect_type}, column: {effect.column})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply event '{event.name}' effect to table '{table.name}': {e}"
                    )
    
    return result_df

