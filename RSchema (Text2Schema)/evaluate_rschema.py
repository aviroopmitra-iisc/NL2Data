"""
RSchema Evaluation Script

Evaluates NL to LogicalIR generation against gold DDL schemas.
Uses WordNet, semantic similarity, and LCS for alignment.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import hashlib

# Add nl2data to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nl2data" / "src"))

from nl2data.agents.base import Blackboard
from nl2data.agents.orchestrator import Orchestrator
from nl2data.utils.agent_factory import create_agent_list
from nl2data.config.logging import setup_logging
from nl2data.ir.logical import LogicalIR, TableSpec, ColumnSpec, ForeignKeySpec, SQLType
from nl2data.ir.constraint_ir import ConstraintSpec

# Try to import optional dependencies
try:
    from nltk.corpus import wordnet as wn
    from nltk import download as nltk_download
    WORDNET_AVAILABLE = True
    try:
        wn.synsets('test')
    except LookupError:
        print("Downloading WordNet data...")
        nltk_download('wordnet', quiet=True)
        nltk_download('omw-1.4', quiet=True)
except ImportError:
    WORDNET_AVAILABLE = False
    print("Warning: WordNet not available. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Global semantic model (lazy loaded)
_semantic_model = None


def get_semantic_model():
    """Lazy load semantic similarity model."""
    global _semantic_model
    if _semantic_model is None and SEMANTIC_AVAILABLE:
        print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _semantic_model


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    return name.lower().strip().replace("_", "").replace(" ", "")


def get_wordnet_synonyms(word: str) -> Set[str]:
    """Get WordNet synonyms for a word."""
    if not WORDNET_AVAILABLE:
        return set()
    
    synonyms = set()
    word_lower = word.lower()
    
    # Get synsets for the word
    for syn in wn.synsets(word_lower):
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ').lower()
            synonyms.add(lemma_name)
            synonyms.add(lemma_name.replace(' ', ''))
    
    # Also add the original word
    synonyms.add(word_lower)
    synonyms.add(word_lower.replace(' ', ''))
    
    return synonyms


def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using sentence-transformers."""
    model = get_semantic_model()
    if model is None:
        return 0.0
    
    try:
        embeddings = model.encode([text1, text2])
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(similarity)
    except Exception as e:
        print(f"  Warning: Semantic similarity failed: {e}")
        return 0.0


def longest_common_substring(s1: str, s2: str) -> int:
    """Calculate length of longest common substring."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    
    m, n = len(s1_lower), len(s2_lower)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1_lower[i-1] == s2_lower[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    
    return max_len


def lcs_ratio(s1: str, s2: str, threshold: float = 0.75) -> bool:
    """Check if LCS ratio exceeds threshold."""
    lcs_len = longest_common_substring(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return False
    ratio = lcs_len / max_len
    return ratio >= threshold


def parse_ddl_to_logical_ir(ddl: str) -> LogicalIR:
    """
    Parse SQL DDL CREATE TABLE statements to LogicalIR.
    
    Args:
        ddl: SQL DDL string with CREATE TABLE statements
        
    Returns:
        LogicalIR object
    """
    tables = {}
    
    # Find all CREATE TABLE statements
    create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?\s*\((.*?)\)\s*;'
    matches = list(re.finditer(create_table_pattern, ddl, re.IGNORECASE | re.DOTALL))
    
    if not matches:
        # Try without semicolon
        create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?\s*\((.*?)\)\s*(?=\n\n|CREATE|$)'
        matches = list(re.finditer(create_table_pattern, ddl, re.IGNORECASE | re.DOTALL))
    
    for match in matches:
        table_name = match.group(1)
        table_body = match.group(2)
        
        columns = []
        primary_key = []
        foreign_keys = []
        
        # Split by commas, handling nested parentheses
        parts = []
        current = ""
        depth = 0
        for char in table_body:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        
        # Parse each part
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for PRIMARY KEY
            if re.match(r'PRIMARY\s+KEY', part, re.IGNORECASE):
                pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', part, re.IGNORECASE)
                if pk_match:
                    pk_cols = [c.strip().strip('`') for c in pk_match.group(1).split(',')]
                    primary_key.extend(pk_cols)
                else:
                    # Inline PRIMARY KEY (e.g., "col INTEGER PRIMARY KEY")
                    pk_match = re.search(r'`?(\w+)`?\s+.*?\s+PRIMARY\s+KEY', part, re.IGNORECASE)
                    if pk_match:
                        primary_key.append(pk_match.group(1))
                continue
            
            # Check for FOREIGN KEY
            fk_match = re.search(
                r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+`?(\w+)`?\s*\(([^)]+)\)',
                part, re.IGNORECASE
            )
            if fk_match:
                fk_col = fk_match.group(1).strip().strip('`')
                ref_table = fk_match.group(2)
                ref_col = fk_match.group(3).strip().strip('`')
                foreign_keys.append({
                    'column': fk_col,
                    'ref_table': ref_table,
                    'ref_column': ref_col
                })
                continue
            
            # Parse column definition
            col_match = re.match(
                r'`?(\w+)`?\s+(\w+(?:\([^)]+\))?)\s*(.*)',
                part, re.IGNORECASE
            )
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                col_attrs = col_match.group(3)
                
                # Check for inline PRIMARY KEY
                is_pk = 'PRIMARY KEY' in col_attrs.upper()
                if is_pk and col_name not in primary_key:
                    primary_key.append(col_name)
                
                # Determine nullable
                nullable = 'NOT NULL' not in col_attrs.upper()
                
                # Determine SQL type
                sql_type = ddl_type_to_sql_type(col_type)
                
                columns.append({
                    'name': col_name,
                    'type': sql_type,
                    'nullable': nullable
                })
        
        tables[table_name] = {
            'columns': columns,
            'primary_key': primary_key,
            'foreign_keys': foreign_keys
        }
    
    # Convert to LogicalIR
    logical_tables = {}
    for table_name, table_info in tables.items():
        cols = []
        for col_info in table_info['columns']:
            col = ColumnSpec(
                name=col_info['name'],
                sql_type=col_info['type'],
                nullable=col_info['nullable']
            )
            cols.append(col)
        
        fks = []
        for fk_info in table_info['foreign_keys']:
            fk = ForeignKeySpec(
                column=fk_info['column'],
                ref_table=fk_info['ref_table'],
                ref_column=fk_info['ref_column']
            )
            fks.append(fk)
        
        table = TableSpec(
            name=table_name,
            columns=cols,
            primary_key=table_info['primary_key'],
            foreign_keys=fks
        )
        logical_tables[table_name] = table
    
    logical_ir = LogicalIR(
        tables=logical_tables,
        constraints=ConstraintSpec()
    )
    
    return logical_ir


def ddl_type_to_sql_type(ddl_type: str) -> SQLType:
    """Convert DDL type to SQLType."""
    ddl_type = ddl_type.upper().strip()
    
    # Extract base type (before parentheses)
    base_type = re.split(r'[\(\)]', ddl_type)[0].strip()
    
    if base_type in ['INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'MEDIUMINT']:
        return "INT"
    elif base_type in ['FLOAT', 'DOUBLE', 'REAL', 'DECIMAL', 'NUMERIC']:
        return "FLOAT"
    elif base_type == 'DATE':
        return "DATE"
    elif base_type in ['DATETIME', 'TIMESTAMP', 'TIME']:
        return "DATETIME"
    elif base_type in ['BOOLEAN', 'BOOL', 'BIT']:
        return "BOOL"
    else:
        return "TEXT"


def generate_logical_ir_from_nl(description: str) -> Optional[LogicalIR]:
    """
    Generate LogicalIR from natural language description (no retry loop).
    
    Args:
        description: Natural language description
        
    Returns:
        LogicalIR or None if generation failed
    """
    try:
        setup_logging()
        agents = create_agent_list(description)
        query_id = hashlib.md5(description.encode()).hexdigest()[:8]
        
        # Disable repair to avoid retry loops
        board = Orchestrator(
            agents,
            query_id=query_id,
            query_text=description,
            enable_repair=False
        ).execute(Blackboard())
        
        if board.dataset_ir and board.dataset_ir.logical:
            return board.dataset_ir.logical
        return None
    except Exception as e:
        print(f"  ERROR: NL to IR generation failed: {e}")
        return None


def align_names(
    gold_names: Set[str],
    pred_names: Set[str],
    use_wordnet: bool = True,
    semantic_threshold: float = 0.6,
    lcs_threshold: float = 0.75
) -> Dict[str, str]:
    """
    Align predicted names to gold names using WordNet, semantic similarity, and LCS.
    
    Returns:
        Dict mapping predicted_name -> gold_name
    """
    matches = {}
    used_gold = set()
    
    # Step 1: WordNet synonym matching
    if use_wordnet and WORDNET_AVAILABLE:
        for pred_name in pred_names:
            if pred_name in matches:
                continue
            
            pred_synonyms = get_wordnet_synonyms(pred_name)
            best_match = None
            best_score = 0.0
            
            for gold_name in gold_names:
                if gold_name in used_gold:
                    continue
                
                gold_normalized = normalize_name(gold_name)
                pred_normalized = normalize_name(pred_name)
                
                # Check if gold is in synonyms
                if gold_normalized in pred_synonyms or pred_normalized == gold_normalized:
                    matches[pred_name] = gold_name
                    used_gold.add(gold_name)
                    break
    
    # Step 2: Semantic similarity (for unmatched gold names)
    if SEMANTIC_AVAILABLE:
        unmatched_pred = [p for p in pred_names if p not in matches]
        unmatched_gold = [g for g in gold_names if g not in used_gold]
        
        for pred_name in unmatched_pred:
            if pred_name in matches:
                continue
            
            best_match = None
            best_score = 0.0
            
            for gold_name in unmatched_gold:
                if gold_name in used_gold:
                    continue
                
                score = semantic_similarity(pred_name, gold_name)
                if score >= semantic_threshold and score > best_score:
                    best_match = gold_name
                    best_score = score
            
            if best_match:
                matches[pred_name] = best_match
                used_gold.add(best_match)
    
    # Step 3: LCS matching (for remaining unmatched gold names)
    unmatched_pred = [p for p in pred_names if p not in matches]
    unmatched_gold = [g for g in gold_names if g not in used_gold]
    
    for pred_name in unmatched_pred:
        if pred_name in matches:
            continue
        
        best_match = None
        best_lcs_ratio = 0.0
        
        for gold_name in unmatched_gold:
            if gold_name in used_gold:
                continue
            
            if lcs_ratio(pred_name, gold_name, lcs_threshold):
                lcs_len = longest_common_substring(pred_name, gold_name)
                max_len = max(len(pred_name), len(gold_name))
                ratio = lcs_len / max_len if max_len > 0 else 0.0
                
                if ratio > best_lcs_ratio:
                    best_match = gold_name
                    best_lcs_ratio = ratio
        
        if best_match:
            matches[pred_name] = best_match
            used_gold.add(best_match)
    
    return matches


def align_tables(gold_ir: LogicalIR, pred_ir: LogicalIR) -> Dict[str, str]:
    """Align predicted tables to gold tables."""
    gold_tables = set(gold_ir.tables.keys())
    pred_tables = set(pred_ir.tables.keys())
    return align_names(gold_tables, pred_tables)


def align_columns(
    gold_columns: List[ColumnSpec],
    pred_columns: List[ColumnSpec]
) -> Dict[str, str]:
    """Align predicted columns to gold columns within matched tables."""
    gold_col_names = {col.name for col in gold_columns}
    pred_col_names = {col.name for col in pred_columns}
    return align_names(gold_col_names, pred_col_names)


def calculate_table_metrics(
    gold_ir: LogicalIR,
    pred_ir: LogicalIR,
    table_matches: Dict[str, str]
) -> Dict[str, float]:
    """Calculate table-level precision, recall, F1, and accuracy."""
    gold_tables = set(gold_ir.tables.keys())
    pred_tables = set(pred_ir.tables.keys())
    matched_gold = set(table_matches.values())
    
    intersection = len(matched_gold)
    precision = intersection / len(pred_tables) if pred_tables else 0.0
    recall = intersection / len(gold_tables) if gold_tables else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    accuracy = 1.0 if f1 == 1.0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def calculate_column_metrics(
    gold_ir: LogicalIR,
    pred_ir: LogicalIR,
    table_matches: Dict[str, str]
) -> Dict[str, float]:
    """Calculate column-level F1 and accuracy."""
    gold_tables = set(gold_ir.tables.keys())
    table_f1_scores = []
    
    for gold_table in gold_tables:
        if gold_table not in table_matches.values():
            # Unmapped table gets F1 = 0
            table_f1_scores.append(0.0)
            continue
        
        # Find predicted table that maps to this gold table
        pred_table = None
        for p, g in table_matches.items():
            if g == gold_table:
                pred_table = p
                break
        
        if pred_table is None:
            table_f1_scores.append(0.0)
            continue
        
        # Get columns
        gold_cols = gold_ir.tables[gold_table].columns
        pred_cols = pred_ir.tables[pred_table].columns
        
        # Align columns
        col_matches = align_columns(gold_cols, pred_cols)
        
        gold_col_names = {col.name for col in gold_cols}
        pred_col_names = {col.name for col in pred_cols}
        matched_gold_cols = set(col_matches.values())
        
        intersection = len(matched_gold_cols)
        precision = intersection / len(pred_col_names) if pred_col_names else 0.0
        recall = intersection / len(gold_col_names) if gold_col_names else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        table_f1_scores.append(f1)
    
    # Net F1 score
    net_f1 = sum(table_f1_scores) / len(gold_tables) if gold_tables else 0.0
    accuracy = 1.0 if net_f1 == 1.0 else 0.0
    
    return {
        'f1': net_f1,
        'accuracy': accuracy
    }


def calculate_pk_fk_accuracy(
    gold_ir: LogicalIR,
    pred_ir: LogicalIR,
    table_matches: Dict[str, str]
) -> float:
    """Calculate accuracy for primary keys and foreign keys."""
    if not table_matches:
        return 0.0
    
    # Reverse mapping: gold_table -> pred_table
    gold_to_pred = {g: p for p, g in table_matches.items()}
    
    total_checks = 0
    correct_checks = 0
    
    for gold_table, pred_table in gold_to_pred.items():
        gold_table_spec = gold_ir.tables[gold_table]
        pred_table_spec = pred_ir.tables[pred_table]
        
        # Check primary key
        gold_pk = set(gold_table_spec.primary_key)
        pred_pk = set(pred_table_spec.primary_key)
        
        # Align column names
        col_matches = align_columns(gold_table_spec.columns, pred_table_spec.columns)
        
        # Map predicted PK to gold column names
        mapped_pred_pk = set()
        for pred_col in pred_pk:
            if pred_col in col_matches:
                mapped_pred_pk.add(col_matches[pred_col])
            else:
                mapped_pred_pk.add(pred_col)
        
        total_checks += 1
        if gold_pk == mapped_pred_pk:
            correct_checks += 1
        
        # Check foreign keys
        for gold_fk in gold_table_spec.foreign_keys:
            total_checks += 1
            
            # Find matching FK in predicted
            found_match = False
            for pred_fk in pred_table_spec.foreign_keys:
                # Check if columns match (after alignment)
                pred_fk_col_mapped = col_matches.get(pred_fk.column, pred_fk.column)
                if pred_fk_col_mapped == gold_fk.column:
                    # Check if ref_table matches
                    pred_ref_table = table_matches.get(pred_fk.ref_table, pred_fk.ref_table)
                    if pred_ref_table == gold_fk.ref_table:
                        # Check if ref_column matches (need to align columns in ref table)
                        ref_gold_table = gold_ir.tables[gold_fk.ref_table]
                        ref_pred_table = pred_ir.tables[pred_fk.ref_table]
                        ref_col_matches = align_columns(ref_gold_table.columns, ref_pred_table.columns)
                        pred_ref_col_mapped = ref_col_matches.get(pred_fk.ref_column, pred_fk.ref_column)
                        if pred_ref_col_mapped == gold_fk.ref_column:
                            correct_checks += 1
                            found_match = True
                            break
            
            if not found_match:
                # FK not matched
                pass
    
    return correct_checks / total_checks if total_checks > 0 else 0.0


def calculate_datatype_accuracy(
    gold_ir: LogicalIR,
    pred_ir: LogicalIR,
    table_matches: Dict[str, str]
) -> float:
    """Calculate accuracy for data types and constraints."""
    if not table_matches:
        return 0.0
    
    gold_to_pred = {g: p for p, g in table_matches.items()}
    
    total_checks = 0
    correct_checks = 0
    
    for gold_table, pred_table in gold_to_pred.items():
        gold_table_spec = gold_ir.tables[gold_table]
        pred_table_spec = pred_ir.tables[pred_table]
        
        # Align columns
        gold_cols = {col.name: col for col in gold_table_spec.columns}
        pred_cols = {col.name: col for col in pred_table_spec.columns}
        col_matches = align_columns(gold_table_spec.columns, pred_table_spec.columns)
        
        # Check each matched column
        for pred_col_name, gold_col_name in col_matches.items():
            pred_col = pred_cols[pred_col_name]
            gold_col = gold_cols[gold_col_name]
            
            # Check data type
            total_checks += 1
            if pred_col.sql_type == gold_col.sql_type:
                correct_checks += 1
            
            # Check if PK column has nullable=False
            if gold_col_name in gold_table_spec.primary_key:
                total_checks += 1
                if pred_col_name in pred_table_spec.primary_key:
                    # Check nullable (PK should be NOT NULL)
                    if not pred_col.nullable and not gold_col.nullable:
                        correct_checks += 1
    
    return correct_checks / total_checks if total_checks > 0 else 0.0


def evaluate_single_entry(entry: dict) -> Optional[Dict]:
    """Evaluate a single entry from annotation_ddl.jsonl."""
    entry_id = entry.get('id', 'unknown')
    question = entry.get('question', '')
    answer = entry.get('answer', '')
    
    print(f"\nProcessing entry {entry_id}...")
    
    # Generate LogicalIR from NL
    print("  Generating LogicalIR from NL...")
    pred_ir = generate_logical_ir_from_nl(question)
    if pred_ir is None:
        print("  ERROR: Failed to generate LogicalIR from NL")
        return None
    
    # Parse DDL to LogicalIR
    print("  Parsing DDL to LogicalIR...")
    try:
        gold_ir = parse_ddl_to_logical_ir(answer)
    except Exception as e:
        print(f"  ERROR: Failed to parse DDL: {e}")
        return None
    
    # Align tables
    print("  Aligning tables...")
    table_matches = align_tables(gold_ir, pred_ir)
    
    # Calculate metrics
    print("  Calculating metrics...")
    table_metrics = calculate_table_metrics(gold_ir, pred_ir, table_matches)
    column_metrics = calculate_column_metrics(gold_ir, pred_ir, table_matches)
    pk_fk_acc = calculate_pk_fk_accuracy(gold_ir, pred_ir, table_matches)
    datatype_acc = calculate_datatype_accuracy(gold_ir, pred_ir, table_matches)
    
    results = {
        'id': entry_id,
        'table_precision': table_metrics['precision'],
        'table_recall': table_metrics['recall'],
        'table_f1': table_metrics['f1'],
        'table_accuracy': table_metrics['accuracy'],
        'column_f1': column_metrics['f1'],
        'column_accuracy': column_metrics['accuracy'],
        'pk_fk_accuracy': pk_fk_acc,
        'datatype_accuracy': datatype_acc
    }
    
    # Calculate average
    scores = [
        table_metrics['f1'],
        table_metrics['accuracy'],
        column_metrics['f1'],
        column_metrics['accuracy'],
        pk_fk_acc,
        datatype_acc
    ]
    results['average'] = sum(scores) / len(scores)
    
    return results


def main():
    """Main evaluation function."""
    script_dir = Path(__file__).parent
    annotation_file = script_dir / "annotation_ddl.jsonl"
    
    if not annotation_file.exists():
        print(f"Error: {annotation_file} not found")
        return
    
    print(f"Reading annotations from {annotation_file}...")
    entries = []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    print(f"Found {len(entries)} entries")
    
    # Evaluate each entry
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}]")
        result = evaluate_single_entry(entry)
        if result:
            results.append(result)
    
    # Calculate averages
    if results:
        num_entries = len(results)
        avg_table_f1 = sum(r['table_f1'] for r in results) / num_entries
        avg_table_acc = sum(r['table_accuracy'] for r in results) / num_entries
        avg_column_f1 = sum(r['column_f1'] for r in results) / num_entries
        avg_column_acc = sum(r['column_accuracy'] for r in results) / num_entries
        avg_pk_fk_acc = sum(r['pk_fk_accuracy'] for r in results) / num_entries
        avg_datatype_acc = sum(r['datatype_accuracy'] for r in results) / num_entries
        avg_overall = sum(r['average'] for r in results) / num_entries
        
        # Write results to markdown
        output_file = script_dir / "evaluation_results.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RSchema Evaluation Results\n\n")
            f.write(f"Total entries evaluated: {num_entries}\n\n")
            f.write("## Overall Averages\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Table F1 | {avg_table_f1:.4f} |\n")
            f.write(f"| Table Accuracy | {avg_table_acc:.4f} |\n")
            f.write(f"| Column F1 | {avg_column_f1:.4f} |\n")
            f.write(f"| Column Accuracy | {avg_column_acc:.4f} |\n")
            f.write(f"| PK/FK Accuracy | {avg_pk_fk_acc:.4f} |\n")
            f.write(f"| Datatype Accuracy | {avg_datatype_acc:.4f} |\n")
            f.write(f"| **Average** | **{avg_overall:.4f}** |\n\n")
            
            f.write("## Per-Entry Results\n\n")
            f.write("| ID | Table F1 | Table Acc | Column F1 | Column Acc | PK/FK Acc | Datatype Acc | Average |\n")
            f.write("|----|----------|-----------|-----------|------------|-----------|--------------|----------|\n")
            
            for r in results:
                f.write(f"| {r['id']} | {r['table_f1']:.4f} | {r['table_accuracy']:.4f} | "
                       f"{r['column_f1']:.4f} | {r['column_accuracy']:.4f} | "
                       f"{r['pk_fk_accuracy']:.4f} | {r['datatype_accuracy']:.4f} | "
                       f"{r['average']:.4f} |\n")
        
        print(f"\n\nResults written to {output_file}")
        print(f"\nOverall Average: {avg_overall:.4f}")
    else:
        print("\nNo successful evaluations")


if __name__ == "__main__":
    main()

