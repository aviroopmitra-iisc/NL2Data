"""Multi-table evaluator."""

from typing import Dict
import pandas as pd
from nl2data.ir.logical import LogicalIR
from nl2data.ir.dataset import DatasetIR
from nl2data.evaluation.config import MultiTableEvalConfig
from nl2data.evaluation.models.multi_table import (
    MultiTableEvaluationReport,
    SchemaMatchResult,
    TableMatch,
    ColumnMatch,
    TableScore,
    RelationshipScore,
)
from nl2data.evaluation.matching.table_matcher import match_tables
from nl2data.evaluation.matching.column_matcher import match_columns
from nl2data.evaluation.aggregation.schema_score import (
    compute_schema_score,
    compute_schema_score_schema_only,
)
from nl2data.evaluation.aggregation.structure_score import (
    compute_primary_key_score_schema_only,
    compute_foreign_key_score_schema_only,
    compute_functional_dependency_score_schema_only,
)
from nl2data.evaluation.matching.column_matcher import _find_best_column_mapping_f1
from nl2data.evaluation.aggregation.utility_score import compute_utility_score
from nl2data.evaluation.metrics.schema.coverage import compute_coverage_factors
from nl2data.evaluation.quality import compute_quality_scores
from nl2data.evaluation.models.multi_table import QualityScore
from nl2data.config.logging import get_logger

logger = get_logger(__name__)


def evaluate_multi_table(
    real_ir: LogicalIR,
    synth_ir: LogicalIR,
    real_dfs: Dict[str, pd.DataFrame],
    synth_dfs: Dict[str, pd.DataFrame],
    config: MultiTableEvalConfig,
) -> MultiTableEvaluationReport:
    """
    Complete multi-table evaluation pipeline.
    
    Steps:
    1. Schema matching (tables + columns)
    2. Compute S_schema
    3. Compute S_structure,intra
    4. Compute S_structure,inter
    5. Compute S_utility (if config includes utility tasks)
    6. Compute S_global (optional)
    
    Args:
        real_ir: Real LogicalIR
        synth_ir: Synthetic LogicalIR
        real_dfs: Real DataFrames
        synth_dfs: Synthetic DataFrames
        config: Multi-table evaluation configuration
        
    Returns:
        MultiTableEvaluationReport with all scores
    """
    logger.info("Starting multi-table evaluation")
    
    # Check if we have data or need schema-only mode
    has_data = False
    for table_name, df in real_dfs.items():
        if len(df) > 0 and len(df.columns) > 0:
            has_data = True
            break
    
    if not has_data:
        logger.info("No data detected - using schema-only evaluation mode")
    
    # Step 1: Schema matching
    logger.info("Step 1: Matching schemas...")
    table_mapping = match_tables(
        real_ir,
        synth_ir,
        real_dfs,
        synth_dfs,
        threshold=0.0,  # No threshold - Hungarian finds best alignment
        use_hungarian=config.matching.use_hungarian,
    )
    
    logger.info(f"Matched {len(table_mapping)} tables")
    
    # Column matching for each matched table
    column_mappings = {}
    table_matches = []
    unmatched_real_tables = []
    unmatched_synth_tables = []
    
    for real_table_name, synth_table_name in table_mapping.items():
        real_table = real_ir.tables.get(real_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        
        if not real_table or not synth_table:
            continue
        
        real_df = real_dfs.get(real_table_name, pd.DataFrame())
        synth_df = synth_dfs.get(synth_table_name, pd.DataFrame())
        
        # Match columns (with schema-only support)
        col_matches, col_similarities = match_columns(
            real_table,
            synth_table,
            real_df,
            synth_df,
            threshold=0.0,  # No threshold - Hungarian finds best alignment (only datatype mismatch filtered)
            use_hungarian=config.matching.use_hungarian,
            real_ir=real_ir,
            synth_ir=synth_ir,
            real_table_name=real_table_name,
            synth_table_name=synth_table_name,
        )
        
        column_mappings[real_table_name] = col_matches
        # Store similarities for later use
        if not hasattr(evaluate_multi_table, '_column_similarities'):
            evaluate_multi_table._column_similarities = {}
        evaluate_multi_table._column_similarities[real_table_name] = col_similarities
        
        # Compute similarity for table match
        from nl2data.evaluation.matching.table_matcher import compute_table_similarity
        similarity = compute_table_similarity(
            real_table_name,
            synth_table_name,
            real_df,
            synth_df,
            real_ir,
            synth_ir,
        )
        
        table_matches.append(
            TableMatch(
                real_table=real_table_name,
                synth_table=synth_table_name,
                similarity=similarity,
            )
        )
    
    # Find unmatched tables
    all_real_tables = set(real_ir.tables.keys())
    all_synth_tables = set(synth_ir.tables.keys())
    matched_real = set(table_mapping.keys())
    matched_synth = set(table_mapping.values())
    
    unmatched_real_tables = list(all_real_tables - matched_real)
    unmatched_synth_tables = list(all_synth_tables - matched_synth)
    
    # Find unmatched columns
    unmatched_real_columns = {}
    unmatched_synth_columns = {}
    
    for real_table_name, real_table in real_ir.tables.items():
        if real_table_name in table_mapping:
            col_matches = column_mappings.get(real_table_name, {})
            all_real_cols = {c.name for c in real_table.columns}
            matched_real_cols = set(col_matches.keys())
            unmatched_real_columns[real_table_name] = list(all_real_cols - matched_real_cols)
            
            synth_table_name = table_mapping[real_table_name]
            synth_table = synth_ir.tables.get(synth_table_name)
            if synth_table:
                all_synth_cols = {c.name for c in synth_table.columns}
                matched_synth_cols = set(col_matches.values())
                unmatched_synth_columns[synth_table_name] = list(all_synth_cols - matched_synth_cols)
    
    # Build column match objects with actual similarity scores
    column_match_objects = {}
    column_similarities = getattr(evaluate_multi_table, '_column_similarities', {})
    for table_name, col_matches in column_mappings.items():
        table_similarities = column_similarities.get(table_name, {})
        column_match_objects[table_name] = [
            ColumnMatch(
                real_column=r, 
                synth_column=s, 
                similarity=table_similarities.get((r, s), 0.0)  # Use actual similarity or 0.0 if not found
            )
            for r, s in col_matches.items()
        ]
    
    # Compute coverage factors
    coverage_factors = compute_coverage_factors(
        table_mapping, column_mappings, real_ir
    )
    
    schema_match = SchemaMatchResult(
        table_matches=table_matches,
        column_matches=column_match_objects,
        unmatched_real_tables=unmatched_real_tables,
        unmatched_synth_tables=unmatched_synth_tables,
        unmatched_real_columns=unmatched_real_columns,
        unmatched_synth_columns=unmatched_synth_columns,
        table_coverage=coverage_factors["table_coverage"],
        column_coverage=coverage_factors["column_coverage"],
    )
    
    # Step 2: Quality evaluation (SD Metrics) - if enabled
    data_quality_score = None
    table_quality_scores = None
    column_quality_scores = None
    
    if config.quality.enabled:
        logger.info("Step 2: Computing quality scores (SD Metrics)...")
        try:
            quality_results = compute_quality_scores(
                real_ir=real_ir,
                synth_ir=synth_ir,
                real_dfs=real_dfs,
                synth_dfs=synth_dfs,
                schema_match_result=schema_match
            )
            
            data_quality_score = quality_results["overall_quality"]
            table_quality_scores = {
                table: result["overall_score"]
                for table, result in quality_results["table_quality"].items()
            }
            column_quality_scores = {
                table: {
                    col: col_info["score"]
                    for col, col_info in result["column_scores"].items()
                }
                for table, result in quality_results["table_quality"].items()
            }
            
            # Update schema_match with quality scores
            quality_scores_dict = {}
            for table, result in quality_results["table_quality"].items():
                # Convert pair_scores from tuple keys to string keys
                pair_scores_str = {
                    f"{col1},{col2}": pair_info["score"]
                    for (col1, col2), pair_info in result["pair_scores"].items()
                }
                quality_scores_dict[table] = QualityScore(
                    overall_score=result["overall_score"],
                    column_scores={
                        col: col_info["score"]
                        for col, col_info in result["column_scores"].items()
                    },
                    pair_scores=pair_scores_str
                )
            
            schema_match.quality_scores = quality_scores_dict
            schema_match.overall_quality = data_quality_score
            
            logger.info(f"Data quality score = {data_quality_score:.4f}")
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            # Continue without quality scores
    
    # Step 3: Table score (F1) - average of table F1 scores from column matching
    logger.info("Step 3: Computing table score (F1)...")
    table_f1_scores = []
    for real_table_name, synth_table_name in table_mapping.items():
        real_table = real_ir.tables.get(real_table_name)
        synth_table = synth_ir.tables.get(synth_table_name)
        if real_table and synth_table:
            _, table_f1 = _find_best_column_mapping_f1(real_table, synth_table)
            table_f1_scores.append(table_f1)
    S_table = sum(table_f1_scores) / len(table_f1_scores) if table_f1_scores else 0.0
    logger.info(f"Table Score (F1) = {S_table:.4f}")
    
    # Step 4: Column score (F1) - sum of column F1 scores / number of matched tables
    logger.info("Step 4: Computing column score (F1)...")
    # Column F1 scores are already computed above (same as table F1 scores)
    S_column = S_table  # Column score is the same as table score (average of column F1 per table)
    logger.info(f"Column Score (F1) = {S_column:.4f}")
    
    # Step 5: Primary key score (F1)
    logger.info("Step 5: Computing primary key score (F1)...")
    S_pk = compute_primary_key_score_schema_only(
        real_ir,
        synth_ir,
        table_mapping,
        column_mappings,
    )
    logger.info(f"PK Score (F1) = {S_pk:.4f}")
    
    # Step 6: Foreign key score (F1)
    logger.info("Step 6: Computing foreign key score (F1)...")
    S_fk = compute_foreign_key_score_schema_only(
        real_ir,
        synth_ir,
        table_mapping,
        column_mappings,
    )
    logger.info(f"FK Score (F1) = {S_fk:.4f}")
    
    # Step 7: Functional dependency score (F1)
    logger.info("Step 7: Computing functional dependency score (F1)...")
    S_fd = compute_functional_dependency_score_schema_only(
        real_ir,
        synth_ir,
        table_mapping,
        column_mappings,
    )
    logger.info(f"FD Score (F1) = {S_fd:.4f}")
    
    # Step 6: Utility (if enabled)
    S_utility = None
    utility_local = None
    utility_relational = None
    utility_queries = None
    
    if config.compute_utility:
        logger.info("Step 6: Computing utility score...")
        S_utility = compute_utility_score(
            real_ir,
            synth_ir,
            real_dfs,
            synth_dfs,
            table_mapping,
            column_mappings,
            config,
        )
        logger.info(f"S_utility = {S_utility:.4f}")
        
        # Extract component scores (simplified - would need to return from compute_utility_score)
        if config.ml_target_columns:
            utility_local = compute_local_utility(
                real_dfs,
                synth_dfs,
                table_mapping,
                column_mappings,
                config.ml_target_columns,
            )
        if config.query_workload:
            utility_queries = compute_query_utility(
                real_dfs, synth_dfs, config.query_workload, table_mapping
            )
    
    # Step 7: Global score (if enabled)
    S_global = None
    if config.compute_global_score:
        logger.info("Step 7: Computing global score...")
        S_global = (
            config.global_score.schema_weight * S_schema
            + config.global_score.structure_intra_weight * S_structure_intra
            + config.global_score.structure_inter_weight * S_structure_inter
            + config.global_score.utility_weight * (S_utility if S_utility is not None else 0.0)
        )
        logger.info(f"S_global = {S_global:.4f}")
    
    # Build report
    report = MultiTableEvaluationReport(
        schema_match=schema_match,
        # New F1-based scores
        table_score=S_table,
        column_score=S_column,
        primary_key_score=S_pk,
        foreign_key_score=S_fk,
        functional_dependency_score=S_fd,
        # Legacy fields (for backward compatibility)
        schema_score=S_column,
        structure_intra_score=S_table,
        structure_inter_score=S_fk,
        utility_score=S_utility,
        global_score=S_global,
        data_quality_score=data_quality_score,
        table_quality_scores=table_quality_scores,
        column_quality_scores=column_quality_scores,
        table_scores=[],  # TODO: populate with detailed table scores
        relationship_scores=[],  # TODO: populate with detailed relationship scores
        utility_local=utility_local,
        utility_relational=utility_relational,
        utility_queries=utility_queries,
        config=config.model_dump(),
    )
    
    logger.info("Multi-table evaluation completed")
    return report
