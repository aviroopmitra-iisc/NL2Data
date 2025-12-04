"""LLM integration for conflict resolution and decision making."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "nl2data" / "src"))
sys.path.insert(0, str(project_root))

from nl2data.ir.logical import LogicalIR, ColumnSpec
from nl2data.ir.constraint_ir import FDConstraint


class LLMClient:
    """Simple LLM client interface. Can be extended with actual API calls."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", **kwargs):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key for LLM service (if needed)
            model: Model name to use
            **kwargs: Additional configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self.prompts_dir = Path(__file__).parent / "prompts"
    
    def _load_prompt(self, prompt_name: str) -> str:
        """Load prompt template from file."""
        prompt_path = self.prompts_dir / prompt_name
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM API with prompt.
        
        This is a placeholder - should be replaced with actual API call.
        For now, returns empty dict to allow testing without API.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            JSON response as dictionary
        """
        # TODO: Implement actual LLM API call
        # Example:
        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return json.loads(response.choices[0].message.content)
        
        # Placeholder: return empty dict (will cause fallback to defaults)
        print(f"[LLM] Would call LLM with prompt (length: {len(prompt)})")
        return {}
    
    def resolve_type_conflict(
        self,
        table_name: str,
        column: ColumnSpec,
        numeric_stats: Dict[str, Any],
        categorical_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve conflict when column has both numeric and categorical stats.
        
        Args:
            table_name: Name of the table
            column: Column specification
            numeric_stats: Numeric statistics
            categorical_stats: Categorical statistics
            
        Returns:
            Decision dictionary with "decision" and "reasoning"
        """
        prompt_template = self._load_prompt("type_conflict_resolution.txt")
        
        # Prepare input
        input_data = {
            "table_name": table_name,
            "column_name": column.name,
            "sql_type": column.sql_type,
            "numeric_summary": {
                "mean": numeric_stats.get("mean"),
                "std": numeric_stats.get("std"),
                "min": numeric_stats.get("min"),
                "max": numeric_stats.get("max"),
                "distribution_fit": numeric_stats.get("distribution_fit", {})
            },
            "categorical_summary": {
                "cardinality": categorical_stats.get("cardinality"),
                "top_1_share": categorical_stats.get("top_1_share"),
                "top_5_share": categorical_stats.get("top_5_share"),
                "value_counts": dict(list(categorical_stats.get("value_counts", {}).items())[:10])
            }
        }
        
        # Format prompt
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "decision" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in type conflict resolution: {e}")
        
        # Fallback: default to numeric if SQL type is numeric
        return {
            "decision": "numeric" if column.sql_type in ["INT", "FLOAT"] else "categorical",
            "reasoning": "Fallback: based on SQL type"
        }
    
    def select_distribution(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        statistical_properties: Dict[str, Any],
        distribution_fits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select best distribution when multiple fits have similar p-values.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            column_type: Column SQL type
            statistical_properties: Statistical properties (mean, std, etc.)
            distribution_fits: List of distribution fits with p-values
            
        Returns:
            Decision dictionary with "selected_distribution", "reasoning", "parameters"
        """
        prompt_template = self._load_prompt("distribution_selection.txt")
        
        input_data = {
            "table_name": table_name,
            "column_name": column_name,
            "column_type": column_type,
            "statistical_properties": statistical_properties,
            "distribution_fits": distribution_fits
        }
        
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "selected_distribution" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in distribution selection: {e}")
        
        # Fallback: use the first (best p-value) distribution
        if distribution_fits:
            best_fit = distribution_fits[0]
            return {
                "selected_distribution": best_fit["name"],
                "reasoning": "Fallback: using best statistical fit",
                "parameters": best_fit.get("parameters", {})
            }
        
        return {
            "selected_distribution": "uniform",
            "reasoning": "Fallback: no fits available",
            "parameters": {}
        }
    
    def decide_categorical_vs_zipf(
        self,
        table_name: str,
        column_name: str,
        categorical_stats: Dict[str, Any],
        zipf_fit: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decide between categorical and Zipf distributions.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            categorical_stats: Categorical statistics
            zipf_fit: Optional Zipf fit parameters
            
        Returns:
            Decision dictionary with "distribution_type", "parameters", "reasoning"
        """
        prompt_template = self._load_prompt("categorical_vs_zipf.txt")
        
        input_data = {
            "table_name": table_name,
            "column_name": column_name,
            "categorical_stats": categorical_stats,
            "zipf_fit": zipf_fit
        }
        
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "distribution_type" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in categorical vs zipf decision: {e}")
        
        # Fallback: use categorical if cardinality < 1000, else zipf
        cardinality = categorical_stats.get("cardinality", 0)
        if cardinality > 1000:
            return {
                "distribution_type": "zipf",
                "parameters": {"s": 1.2, "n": cardinality},
                "reasoning": "Fallback: high cardinality, using zipf"
            }
        else:
            return {
                "distribution_type": "categorical",
                "parameters": {"values": [], "probs": None},
                "reasoning": "Fallback: low cardinality, using categorical"
            }
    
    def suggest_provider(
        self,
        table_name: str,
        column: ColumnSpec,
        distribution_summary: Dict[str, Any],
        logical_ir: LogicalIR,
        discovered_fds: List[FDConstraint]
    ) -> Dict[str, Any]:
        """
        Suggest data provider for a column.
        
        Args:
            table_name: Name of the table
            column: Column specification
            distribution_summary: Distribution summary
            logical_ir: Full logical schema
            discovered_fds: List of discovered functional dependencies
            
        Returns:
            Decision dictionary with "use_provider", "provider_name", "reasoning"
        """
        prompt_template = self._load_prompt("provider_selection.txt")
        
        # Build full schema representation
        schema_dict = {
            "tables": {},
            "constraints": {
                "fds": [
                    {
                        "table": fd.table,
                        "lhs": fd.lhs,
                        "rhs": fd.rhs
                    }
                    for fd in discovered_fds
                ]
            }
        }
        
        for tname, table in logical_ir.tables.items():
            schema_dict["tables"][tname] = {
                "name": table.name,
                "columns": [
                    {
                        "name": col.name,
                        "sql_type": col.sql_type,
                        "role": col.role,
                        "references": col.references
                    }
                    for col in table.columns
                ],
                "primary_key": table.primary_key,
                "foreign_keys": [
                    {
                        "column": fk.column,
                        "ref_table": fk.ref_table,
                        "ref_column": fk.ref_column
                    }
                    for fk in table.foreign_keys
                ]
            }
        
        input_data = {
            "table_name": table_name,
            "column_name": column.name,
            "sql_type": column.sql_type,
            "distribution_summary": distribution_summary,
            "column_metadata": {
                "nullable": column.nullable,
                "unique": column.unique,
                "role": column.role,
                "references": column.references
            },
            "full_schema": schema_dict
        }
        
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "use_provider" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in provider suggestion: {e}")
        
        # Fallback: use lookup provider for foreign keys
        if column.references:
            ref_parts = column.references.split(".")
            if len(ref_parts) == 2:
                return {
                    "use_provider": True,
                    "provider_name": f"lookup.{ref_parts[0]}.{ref_parts[1]}",
                    "reasoning": "Fallback: foreign key detected"
                }
        
        return {
            "use_provider": False,
            "provider_name": None,
            "reasoning": "Fallback: no provider needed"
        }
    
    def infer_missing_statistics(
        self,
        table_name: str,
        column: ColumnSpec,
        logical_ir: LogicalIR,
        discovered_fds: List[FDConstraint]
    ) -> Dict[str, Any]:
        """
        Infer distribution for column with no statistics.
        
        Args:
            table_name: Name of the table
            column: Column specification
            logical_ir: Full logical schema
            discovered_fds: List of discovered functional dependencies
            
        Returns:
            Decision dictionary with "distribution", "provider", "reasoning"
        """
        prompt_template = self._load_prompt("missing_statistics_inference.txt")
        
        # Get related columns in same table
        table = logical_ir.tables.get(table_name)
        related_columns = []
        if table:
            related_columns = [
                {
                    "name": col.name,
                    "sql_type": col.sql_type,
                    "role": col.role
                }
                for col in table.columns
                if col.name != column.name
            ]
        
        # Get related FDs
        related_fds = [
            {
                "table": fd.table,
                "lhs": fd.lhs,
                "rhs": fd.rhs
            }
            for fd in discovered_fds
            if fd.table == table_name and (column.name in fd.lhs or column.name in fd.rhs)
        ]
        
        # Build full schema
        schema_dict = {
            "tables": {},
            "constraints": {
                "fds": [
                    {
                        "table": fd.table,
                        "lhs": fd.lhs,
                        "rhs": fd.rhs
                    }
                    for fd in discovered_fds
                ]
            }
        }
        
        for tname, t in logical_ir.tables.items():
            schema_dict["tables"][tname] = {
                "name": t.name,
                "columns": [
                    {
                        "name": col.name,
                        "sql_type": col.sql_type,
                        "role": col.role,
                        "references": col.references
                    }
                    for col in t.columns
                ],
                "primary_key": t.primary_key,
                "foreign_keys": [
                    {
                        "column": fk.column,
                        "ref_table": fk.ref_table,
                        "ref_column": fk.ref_column
                    }
                    for fk in t.foreign_keys
                ]
            }
        
        input_data = {
            "table_name": table_name,
            "column_name": column.name,
            "column_metadata": {
                "sql_type": column.sql_type,
                "nullable": column.nullable,
                "unique": column.unique,
                "role": column.role,
                "references": column.references
            },
            "related_columns": related_columns,
            "related_fds": related_fds,
            "full_schema": schema_dict
        }
        
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "distribution" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in missing statistics inference: {e}")
        
        # Fallback: basic inference
        if column.references:
            ref_parts = column.references.split(".")
            if len(ref_parts) == 2:
                return {
                    "distribution": None,
                    "provider": {
                        "name": f"lookup.{ref_parts[0]}.{ref_parts[1]}",
                        "config": {}
                    },
                    "reasoning": "Fallback: foreign key, using lookup"
                }
        
        if column.sql_type in ["INT", "FLOAT"]:
            return {
                "distribution": {
                    "kind": "uniform",
                    "low": 0.0,
                    "high": 1000.0
                },
                "provider": None,
                "reasoning": "Fallback: numeric type, using uniform"
            }
        else:
            return {
                "distribution": {
                    "kind": "categorical",
                    "domain": {
                        "values": ["fake_value"],
                        "probs": None
                    }
                },
                "provider": None,
                "reasoning": "Fallback: text type, using categorical"
            }
    
    def select_primary_key(
        self,
        table_name: str,
        candidate_keys: List[List[str]],
        logical_ir: LogicalIR
    ) -> Dict[str, Any]:
        """
        Select primary key from candidate keys using LLM.
        
        Args:
            table_name: Name of the table
            candidate_keys: List of candidate key sets (each is a list of column names)
            logical_ir: Full logical schema
            
        Returns:
            Decision dictionary with "selected_primary_key", "reasoning", "confidence"
        """
        if not candidate_keys:
            return {
                "selected_primary_key": [],
                "reasoning": "No candidate keys available",
                "confidence": 0.0
            }
        
        # If only one candidate key, return it
        if len(candidate_keys) == 1:
            return {
                "selected_primary_key": candidate_keys[0],
                "reasoning": "Only one candidate key available",
                "confidence": 1.0
            }
        
        prompt_template = self._load_prompt("primary_key_selection.txt")
        
        # Get table schema
        table = logical_ir.tables.get(table_name)
        if not table:
            return {
                "selected_primary_key": candidate_keys[0],
                "reasoning": "Table not found in schema",
                "confidence": 0.5
            }
        
        # Build table schema dict
        table_schema = {
            "name": table.name,
            "columns": [
                {
                    "name": col.name,
                    "sql_type": col.sql_type,
                    "nullable": col.nullable,
                    "unique": col.unique,
                    "role": col.role,
                    "references": col.references
                }
                for col in table.columns
            ],
            "primary_key": table.primary_key,
            "foreign_keys": [
                {
                    "column": fk.column,
                    "ref_table": fk.ref_table,
                    "ref_column": fk.ref_column
                }
                for fk in table.foreign_keys
            ]
        }
        
        # Build full schema dict
        schema_dict = {
            "tables": {
                name: {
                    "name": t.name,
                    "columns": [
                        {
                            "name": col.name,
                            "sql_type": col.sql_type,
                            "nullable": col.nullable,
                            "unique": col.unique,
                            "role": col.role,
                            "references": col.references
                        }
                        for col in t.columns
                    ],
                    "primary_key": t.primary_key,
                    "foreign_keys": [
                        {
                            "column": fk.column,
                            "ref_table": fk.ref_table,
                            "ref_column": fk.ref_column
                        }
                        for fk in t.foreign_keys
                    ]
                }
                for name, t in logical_ir.tables.items()
            }
        }
        
        # Prepare input
        input_data = {
            "table_name": table_name,
            "table_schema": table_schema,
            "candidate_keys": candidate_keys,
            "full_schema": schema_dict
        }
        
        prompt = f"""{prompt_template}

## Input Data

```json
{json.dumps(input_data, indent=2)}
```

Please provide your response in the specified JSON format."""
        
        try:
            response = self._call_llm(prompt)
            if response and "selected_primary_key" in response:
                return response
        except Exception as e:
            print(f"[LLM] Error in primary key selection: {e}")
        
        # Fallback: choose smallest candidate key
        smallest_key = min(candidate_keys, key=len)
        return {
            "selected_primary_key": smallest_key,
            "reasoning": "Fallback: selected smallest candidate key",
            "confidence": 0.5
        }

