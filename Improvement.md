# NL2Data IR Enhancement Plan

> **Purpose**: Tactical engineering guide for making IRs richer and more aligned with natural language descriptions.

---

## 🗺️ Current Status

### ✅ Completed Features

All major features have been implemented:
- ✅ DSL extensions: `in`/`not in`, casts, `normal`/`lognormal`/`pareto` functions, `case_when`, `between`, helpers
- ✅ Window functions (`DistWindow`, `window_eval.py`, Phase 3 in generator)
- ✅ Event/Incident system (`EventSpec`, `EventEffect`, `event_eval.py`, Phase 1.5)
- ✅ Nuance coverage checker (`check_nuance_coverage()`)
- ✅ Agent repair loop system (integrated in Orchestrator)
- ✅ Quality metrics tracking (integrated in Orchestrator)
- ✅ Constraint enforcement (integrated in fact generator as Phase 1.7)
- ✅ All distributions: Lognormal, Pareto, Mixture, Poisson, Exponential (IR models, samplers, factory)
- ✅ Prompt fixes (inconsistencies fixed, lag/lead clarified, domain bias reduced)

### 🎯 Pending Tasks

1. **🔥 Multi-Table Relational Evaluation Framework** - **HIGH PRIORITY - IMPLEMENTING SOON**
   - Comprehensive evaluation system for benchmarking synthetic DBs against real DBs
   - Handles schema mismatches, multi-table relationships, and utility scoring
   - See detailed design in "Multi-Table Relational Evaluation Framework" section below
   
2. **Integration Testing** - Test with 2-3 queries to verify LLMs use functions correctly
3. **Optional Enhancements** - See below

---

## 📋 Future Enhancement Tasks

### Task 1: Integration Testing

**Goal**: Verify that LLMs correctly use all implemented features in practice.

**Impact**: HIGH | **Risk**: LOW | **Effort**: LOW

**Steps**:
1. Re-run NL-3 (fraud) and NL-1 (clickstream) queries
2. Verify new features appear in generated IRs
3. Check that LLMs use previously-forbidden functions correctly

---

### Task 2: Enhance Nuance Coverage (Optional)

**Goal**: Expand nuance coverage checker to catch more missing patterns.

**Impact**: MEDIUM | **Risk**: LOW | **Effort**: MEDIUM

**Current State**: Basic implementation exists with good coverage (rolling, incident, heavy_tail, log-normal, pareto, mixture, zipf, pay-day, surge, churn, fraud, seasonal)

**Potential Enhancements**:
- Add more keyword patterns (e.g., "readmission", "proration", "within_days_of")
- Improve semantic understanding (e.g., "pay day" → check for day_of_month logic)
- Integrate into repair loop (automatically fix missing nuances)
- More semantic understanding (beyond keyword matching)
- Automatic repair suggestions

**Note**: This is already well-implemented. Optional enhancements could improve coverage further.

---

### Task 3: Add Additional Distributions (Gamma, Beta, Binomial, Truncated Normal)

**Goal**: Expand distribution coverage for specialized use cases.

**Impact**: MEDIUM | **Risk**: LOW | **Effort**: LOW (~2-3 hours total)

**Distributions to Add**:

1. **Gamma** (`DistGamma`):
   - Use cases: Service times, waiting times, insurance claims (more flexible than Exponential)
   - Parameters: `shape` (k), `scale` (θ)
   - Implementation: `numpy.random.gamma(shape, scale, size)`

2. **Beta** (`DistBeta`):
   - Use cases: Proportions, probabilities, bounded values (0-1), conversion rates
   - Parameters: `alpha` (a), `beta` (b)
   - Implementation: `numpy.random.beta(alpha, beta, size)`

3. **Binomial** (`DistBinomial`):
   - Use cases: Success/failure counts, click-through rates, binary outcomes
   - Parameters: `n` (trials), `p` (success probability)
   - Implementation: `numpy.random.binomial(n, p, size)`

4. **Truncated Normal** (`DistTruncatedNormal`):
   - Use cases: Normal distribution with bounds (ages 0-120, percentages 0-100)
   - Parameters: `mean`, `std`, `low`, `high`
   - Implementation: Sample from normal, clip to bounds, resample if out of bounds

**Implementation**: Follow same pattern as Poisson/Exponential (see Implementation Patterns below)

**Priority**: Can be done if needed for specific queries

---

### Task 4: Add Conditional/Correlated Distributions

**Goal**: Enable region-specific categories, fraud rings, correlated metrics.

**Impact**: MEDIUM | **Risk**: MEDIUM | **Effort**: HIGH

**Note**: Likely deferred until after v1.0 - complex implementation, lower immediate impact.

**Files**:
- `ir/generation.py` - Add `DistConditionalCategorical`, `DistCorrelatedNormal`
- `generation/distributions/conditional.py` - New file with samplers
- `generation/distributions/correlated.py` - New file with samplers
- `generation/distributions/factory.py` - Add factory cases

**Steps**:
1. Add `DistConditionalCategorical` with `condition_on`, `rules`, `default`
2. Add `DistCorrelatedNormal` with `columns`, `means`, `cov_matrix`
3. Implement samplers (conditional needs DataFrame for conditions)
4. Wire into factory

**Gotchas**:
- Conditional categorical needs condition DataFrame during sampling
- Correlated normal requires scipy.stats.multivariate_normal
- Covariance matrix must be positive definite

---

### Task 5: Add State Machine Constraints

**Goal**: Enforce realistic order status flows, session state transitions.

**Impact**: MEDIUM | **Risk**: MEDIUM | **Effort**: MEDIUM

**Files**:
- `ir/constraint_ir.py` - Add `StateMachineConstraint`
- `generation/distributions/state_machine.py` - New file with sampler
- `generation/enforce.py` - Add enforcement logic

**Steps**:
1. Add IR model with `states`, `transitions`, `initial_states`, `terminal_states`
2. Implement `StateMachineSampler` that tracks state per entity
3. Add enforcement in `enforce_batch()`

**Gotchas**:
- Need entity ID column to track state per entity
- Terminal states should have probability of staying vs transitioning

---

### Task 6: Add `within_days_of()` Helper Function

**Goal**: Enable "within 30 days of upgrade" type logic.

**Impact**: MEDIUM | **Risk**: LOW | **Effort**: LOW

**Files**:
- `generation/derived_program.py` - Add to `ALLOWED_FUNCS`
- `generation/derived_eval.py` - Implement function

**Steps**:
1. Add `"within_days_of"` to `ALLOWED_FUNCS`
2. Implement function that checks if `event_ts` is within `k` days of `ref_ts`
3. Handle Series vs scalar cases

**Gotchas**:
- Must handle both Series and scalar inputs
- Days can be before or after reference timestamp

---

### Task 7: Integrate WorkloadIR into Distribution Engineering

**Goal**: Ensure generated data matches query workload patterns.

**Impact**: MEDIUM | **Risk**: LOW | **Effort**: MEDIUM

**Files**:
- `agents/roles/dist_engineer.py` - Read WorkloadIR from blackboard
- `prompts/roles/dist_user.txt` - Include workload context

**Steps**:
1. Modify `DistributionEngineer.run()` to include `WorkloadIR` in prompt
2. Update prompt template to include `{WORKLOAD_JSON}`
3. Add guidance: "If queries group by X, use Zipf for X"

**Gotchas**:
- WorkloadIR might be None (handle gracefully)
- Don't over-constrain based on workload

---

## 📊 Multi-Table Relational Evaluation Framework

> **🔥 PRIORITY IMPLEMENTATION TASK** - This comprehensive evaluation framework is the next major feature to implement. It provides end-to-end benchmarking of synthetic multi-table relational databases against real databases, handling schema mismatches and providing interpretable scores.

---

### 0. Big Picture

**Goal**: Benchmark the pipeline `NL description → generator → synthetic DB (1+ tables)` against a real DB, where:
- Table names may differ
- Column names/types may differ
- Number of tables may differ

**Requirements**:
- Reusable, seeded tests
- Scores in [0,1]
- Several high-level scores rather than dozens of raw metrics

**Three Evaluation Axes** (extended to multi-table):
1. **Schema / Marginal Quality** – per table & per column
2. **Structural Quality** – intra-table + inter-table relationships
3. **Utility** – can synthetic DB substitute real DB for tasks/queries?

**Schema Mismatch Handling**: Via a matching layer (tables + columns) inspired by schema-matching research/tools like Valentine.

**Integration**: Can leverage SDMetrics/SDV's **Multi-Table Quality Report** for connected tables (column shapes, column pair trends, and inter-table trends).

---

### 1. Core Scores (Single-Table → Multi-Table Generalization)

For a full relational DB, define:

- **S_schema** ∈ [0,1]: Schema & Marginals
- **S_structure,intra** ∈ [0,1]: Within-table structure
- **S_structure,inter** ∈ [0,1]: Between-table structure (relations, FKs)
- **S_utility** ∈ [0,1]: Downstream / query utility

**Optional Global Score**:

```
S_global = w₁·S_schema + w₂·S_structure,intra + w₃·S_structure,inter + w₄·S_utility
          (wᵢ ≥ 0, Σwᵢ = 1)
```

**Note**: No privacy score, since real rows are never seen at generation time.

---

### 2. Multi-Table Schema Matching and Alignment

Before scoring, align **real DB** and **synthetic DB**:
- Real: tables T₁ᴿ, ..., Tₘᴿ
- Synthetic: T₁ˢ, ..., Tₙˢ

**Required Mappings**:
- Table mapping: Π_T: Tᴿ → Tˢ ∪ {∅}
- Column mapping per table: Π_C^(k): cols(Tₖᴿ) → cols(Tˢ_Π_T(k)) ∪ {∅}

This is classic schema matching. Valentine and similar frameworks combine name similarity, datatype, and instance distributions to match attributes.

#### 2.1 Table-Level Matching

For each real table vs synthetic table:

**Signals**:
- Name similarity (tokenized names, edit distance, Jaccard over tokens)
- Row count scale (order of magnitude)
- Key cardinality patterns (does it have a primary key-like column?)
- Foreign key pattern: if a table in real references users, and a synthetic table references some "user"-like table, that's a hint they correspond

**Process**:
1. Score a similarity for each (real, synthetic) table pair
2. Solve a bipartite matching that maximizes total similarity (Hungarian / maximum weight matching style)
3. Drop pairs below a threshold (e.g. similarity < 0.6 → treat as unmatched)

**Result**:
- Matched tables (M_T)
- Real-only tables (T_miss)
- Synthetic-only tables (T_extra)

#### 2.2 Column-Level Matching Per Table

For each matched table pair:

**Signals** (similar to schema-matching literature):
- Name similarity
- Type compatibility (numeric vs cat vs datetime)
- Distribution similarity (range, cardinality, example values)
- Optional: semantic similarity using NL descriptions

**Result**:
- Matched columns (M_C^(k))
- Missing columns in that table (C_miss^(k))
- Extra synthetic columns (C_extra^(k))

#### 2.3 Impact of Unmatched Tables/Columns

**Penalize** missing and extra schema elements:

**Table Coverage Factor**:
```
C_T = |M_T| / m_T
```
where m_T = # real tables, |M_T| = matched tables

**Column Coverage Per Table**:
```
C_k = |M_C^(k)| / m_k
```
where m_k = # real columns in table k, |M_C^(k)| = matched columns

**Usage**: These factors are used as multiplicative penalties after computing per-table quality scores.

**Extra Tables/Columns**: Can get an additional simplicity penalty if needed (e.g. exponential in count of extras), but table/column coverage alone is usually enough.

---

### 3. Schema & Marginal Score (S_schema)

#### 3.1 What to Check (Multi-Table)

Per matched table, for each matched column:

**Numeric**:
- Distribution shape (histogram / CDF)
- Range, moments
- Missingness rate

**Categorical**:
- Frequency distribution
- Rare vs common categories
- Missingness

Essentially, the same as single-table column-shape metrics, done **per table** and then aggregated.

#### 3.2 How to Aggregate

**Process**:
1. For each matched table pair, build "aligned" real vs synthetic sub-tables containing only matched columns
2. Run a **single-table Quality Report** on each pair to get that table's column-shape score (S̃_schema^(k) ∈ [0,1])
3. Weight tables by something sensible (e.g. row count, or 1 per table)

**Table-Level Aggregation** (before penalties):
```
S_schema,aligned = (1/|M_T|) · Σ_{k ∈ M_T} S̃_schema^(k)
```

**Coverage Factor**:
```
C_schema = C_T · (1/|M_T| · Σ_{k ∈ M_T} C_k)
```

**Final Score**:
```
S_schema = S_schema,aligned · C_schema
```

**Interpretation**: If you nail shapes on all matched columns but only cover 50% of the real schema, you're capped at ≈0.5.

---

### 4. Structural Scores (S_structure,intra) and (S_structure,inter)

Relational data has **two levels of structure**:
1. Within tables (correlations, functional dependencies, etc.)
2. Across tables (foreign keys, referential integrity, cross-table patterns)

#### 4.1 Intra-Table Structure (S_structure,intra)

This is the multi-table generalization of "column pair trends":

**Process**:
- For each matched table pair:
  - Take the aligned columns
  - Compare pairwise relationships (correlations, contingency tables) between real and synthetic
- SDV/SDMetrics "Column Pair Trends" and related metrics already do this per table

**Aggregation**:
- Per table pair score: S̃_structure,intra^(k) ∈ [0,1]
- Aggregated:
  ```
  S_structure,intra,aligned = (1/|M_T|) · Σ_{k ∈ M_T} S̃_structure,intra^(k)
  ```
- Apply same coverage factor:
  ```
  S_structure,intra = S_structure,intra,aligned · C_schema
  ```

#### 4.2 Inter-Table Structure (S_structure,inter)

This is what really matters for relational DBs:

**Metrics**:

1. **Foreign Key Validity / Referential Integrity**:
   - For each relationship (real PK–FK pair), check fraction of synthetic FK values that exist in the corresponding PK table
   - Relational data quality literature treats referential integrity as a central quantitative quality dimension

2. **Relationship Cardinalities**:
   - 1-to-N distributions: how many children per parent? Are these distributions similar between real and synthetic (e.g. number of orders per customer, number of visits per patient)?
   - N-to-N via bridge tables (e.g. users-roles): distribution of degree on both sides

3. **Inter-Table Trends**:
   - Metrics like SDMetrics' proposed **InterTableTrends** look at relationships defined in metadata, join tables on PK/FK and treat joined pairs as columns to compare trends in denormalized space
   - Example: join orders with customers, check that relationships between customer_age and order_amount are preserved in synthetic join

**Scoring**:

For each real relationship (table A, table B, FK), compute:
- **r_RI**: referential integrity score in [0,1]
- **r_card**: similarity of child-per-parent count distribution
- **r_trend**: similarity of cross-table trends when joined (can reuse column-pair style metrics on the joined table)

**Relationship-Level Score**:
```
r_rel = α·r_RI + β·r_card + γ·r_trend  (α+β+γ=1)
```

**Average Over All Relationships**:
```
S_structure,inter = (1/|R|) · Σ_{relationships} r_rel
```

**Note**: "Missing relationships" (tables or keys that don't exist in synthetic schema) contribute 0 to the average.

Multi-table SDV/SDMetrics already bake some of this into their multi-table metrics and reports (e.g. multi-table quality report and upcoming InterTableTrends property).

---

### 5. Utility Score (S_utility) in a Relational Setting

Utility isn't just "train one classifier." For relational DBs you want:
- **Local tasks**: predictions using a single table
- **Relational tasks**: tasks that need joins
- **Workload / query behavior**: can synthetic DB answer representative analytical queries similarly to real DB?

You don't need all of these from day one, but they are good axes.

#### 5.1 Per-Table ML Tasks (Local Utility)

For tables with a known target (e.g. OpenML tasks; or if you define your own):

**Protocol**:
1. Work on aligned columns + target
2. **Real baseline**: Train model on real table, test on real held-out rows → M_real
3. **Synthetic**: Train on synthetic table, test on real held-out rows → M_syn
4. **Normalize**: u = max(0, min(1, M_syn/(M_real+ε)))

**Aggregate**: Per-table utilities (e.g. mean across all tables with targets) → S_utility,local

#### 5.2 Relational ML Tasks

Define tasks that **require joins**, e.g.:
- Predict customer churn from customers + transactions
- Predict fraud from users + devices + sessions

**Protocol**:
- Use a fixed join recipe (from real schema)
- **For real data**: Build a training dataset via joins, train model, evaluate → M_real,rel
- **Synthetic**: Join synthetic tables the same way, train model, test on **real** joined rows → M_syn,rel
- **Normalize** as before → relational utility score S_utility,rel

#### 5.3 Query-Level Utility

Inspired by relational synthetic data benchmarking work:

**Process**:
- A representative query workload (counts, aggregates, group-bys, join queries)
- For each query q:
  - Run on real DB → answer a_R
  - Run on synthetic DB → answer a_S
  - Compute error (e.g. relative error on aggregates)
- Convert into similarity score per query: s_q = 1 - normalized_error, clip to [0,1]
- Average across queries → S_utility,queries

Recent work on benchmarking synthetic relational data emphasizes this kind of combined fidelity/utility evaluation at the query level.

#### 5.4 Aggregate Utility

**Options**:
- Just pick one main utility dimension (e.g. relational ML tasks) and call that S_utility, or
- Combine:
  ```
  S_utility = λ·S_utility,local + (1-λ)·S_utility,rel
  ```
  optionally blending in query-level utility

---

### 6. Handling Serious Schema Differences Cleanly

#### 6.1 Table Names and Structures Don't Match

**If matching layer cannot find a plausible match** between a real table and any synthetic one (low similarity, missing key patterns, etc.):
- That table is considered **missing** from synthetic DB for scoring
- All metrics that would depend on that table (schema, structure, utility) get a 0 contribution from that table

**Conversely**: Synthetic tables with no real counterpart are ignored for quality but can incur a small penalty if you want to discourage hallucinated tables.

This is analogous to what schema-matching frameworks do when they fail to match attributes; unmapped attributes simply count against recall/precision.

#### 6.2 Column Names & Roles Differ

**For columns**:
- If matching algorithm gives a confident map (e.g. "customer_id" ↔ "cust_id" with high similarity), treat them as the same variable
- If confidence is low, either:
  - Drop the match and treat that real column as missing, or
  - Flag it and optionally require human inspection for the benchmark (you can keep a human-in-the-loop for tricky cases, just as Valentine suggests human involvement for hard matching scenarios)

**Coverage penalties** will then reflect how much schema your generator actually recovered from the description.

---

### 7. Final Shape of the Framework

**Per dataset / DB** (multi-table):

1. **Real DB → NL description → Synthetic DB** (your pipeline)

2. **Schema matching layer**:
   - Match tables (real ↔ synthetic)
   - For each matched table, match columns
   - Identify missing and extra tables/columns

3. **Schema & marginals**:
   - For each matched table pair, evaluate column shapes using SDV/SDMetrics-style metrics
   - Aggregate across tables; apply coverage penalties → S_schema ∈ [0,1]

4. **Structure**:
   - **Intra-table**: Evaluate column pair trends per table; aggregate with coverage → S_structure,intra
   - **Inter-table**: Evaluate referential integrity, relationship cardinalities, and inter-table trends for each real FK; average → S_structure,inter

5. **Utility**:
   - Define a set of per-table and relational tasks, plus optional query workload
   - Compute utility ratios (synthetic-trained vs real-trained models, synthetic vs real query answers)
   - Aggregate → S_utility

6. **Global score (optional)**:
   - Combine into S_global with chosen weights

7. **Repeat across many real DBs**:
   - Analyze distributions of scores vs schema size, #tables, mix of types, etc.

**Everything stays in [0,1]**, and you end up with:
- A **schema-aware** notion of success (did the generator even reconstruct the right tables/columns?)
- Clear separation between:
  - "Did it get column distributions right?"
  - "Did it get correlations and relationships right?"
  - "Is it actually useful for downstream work?"

**Implementation**: Can be implemented in Python with SDV/SDMetrics + a schema-matcher (could even wrap Valentine or similar ideas if you want to go full research-grade).

---

## 🛠️ Implementation Guide

### Feasibility Assessment

**✅ HIGHLY FEASIBLE** - The codebase already has strong evaluation infrastructure that can be extended:

**Existing Infrastructure**:
- ✅ `evaluation/schema_eval.py` - Schema coverage metrics (exact name matching)
- ✅ `evaluation/table_eval.py` - Marginal distributions (KS test, Wasserstein), correlations, mutual information
- ✅ `evaluation/relational_eval.py` - FK coverage, degree distributions, join selectivity
- ✅ `evaluation/integrity.py` - Referential integrity checks
- ✅ `evaluation/workload.py` - Query execution and metrics
- ✅ `evaluation/report_models.py` - Structured report models
- ✅ `evaluation/report_builder.py` - Report generation

**What Needs to Be Added**:
1. **Schema Matching Layer** - Fuzzy matching (name similarity, type compatibility, distribution similarity)
2. **Multi-Table Aggregation** - Coverage penalties, weighted aggregation across tables
3. **Inter-Table Structure Scoring** - Relationship cardinality comparison, inter-table trends
4. **Utility Scoring** - ML task evaluation, query-level utility comparison
5. **Global Score Computation** - Combine all scores into final metrics

### Implementation Plan

#### Phase 1: Schema Matching Layer (NEW)

**File**: `evaluation/schema_matching.py` (NEW)

**Key Functions**:

1. **`match_tables(real_ir: LogicalIR, synth_ir: LogicalIR, real_dfs: Dict, synth_dfs: Dict) -> Dict`**:
   - Compute similarity scores for all (real, synthetic) table pairs
   - Use signals: name similarity (tokenized, edit distance, Jaccard), row count scale, PK patterns, FK patterns
   - Solve bipartite matching (Hungarian algorithm or greedy)
   - Return: `{real_table: synth_table}` mapping, unmatched tables

2. **`match_columns(real_table: TableSpec, synth_table: TableSpec, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict`**:
   - For each matched table pair, match columns
   - Use signals: name similarity, type compatibility, distribution similarity (range, cardinality)
   - Return: `{real_col: synth_col}` mapping, unmatched columns

3. **`compute_table_similarity(real_name: str, synth_name: str, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float`**:
   - Name similarity: tokenized Jaccard, edit distance
   - Row count scale: `1 - abs(log10(real_rows) - log10(synth_rows)) / max_scale`
   - PK pattern: check if both have similar primary key structure
   - FK pattern: check if both reference similar tables
   - Combine with weights → similarity score [0,1]

4. **`compute_column_similarity(real_col: ColumnSpec, synth_col: ColumnSpec, real_series: pd.Series, synth_series: pd.Series) -> float`**:
   - Name similarity: tokenized Jaccard, edit distance
   - Type compatibility: numeric vs cat vs datetime (binary match)
   - Distribution similarity: for numeric (range overlap), for categorical (Jaccard of unique values)
   - Combine with weights → similarity score [0,1]

**Dependencies**: 
- `difflib` or `fuzzywuzzy` for string similarity
- `scipy.optimize.linear_sum_assignment` for Hungarian algorithm (or greedy matching)

**Integration Point**: Called before any scoring in the main evaluation function

---

#### Phase 2: Multi-Table Schema & Marginal Scoring (EXTEND EXISTING)

**File**: `evaluation/multi_table_eval.py` (NEW)

**Key Functions**:

1. **`compute_schema_score(real_ir: LogicalIR, synth_ir: LogicalIR, real_dfs: Dict, synth_dfs: Dict, table_mapping: Dict, column_mappings: Dict) -> float`**:
   - For each matched table pair:
     - Build aligned DataFrames (only matched columns)
     - Call existing `table_eval.numeric_marginals()` and `table_eval.categorical_marginals()` per column
     - Aggregate per-table score: `S̃_schema^(k) = mean(column_scores)`
   - Compute coverage factors: `C_T`, `C_k` (from schema matching)
   - Apply penalties: `S_schema = S_schema,aligned · C_schema`
   - Return: `S_schema ∈ [0,1]`

2. **`compute_intra_structure_score(real_ir: LogicalIR, synth_ir: LogicalIR, real_dfs: Dict, synth_dfs: Dict, table_mapping: Dict, column_mappings: Dict) -> float`**:
   - For each matched table pair:
     - Use existing `table_eval.correlation_metrics()` for all column pairs
     - Aggregate per-table score: `S̃_structure,intra^(k) = mean(1 - correlation_deltas)`
   - Apply coverage penalties
   - Return: `S_structure,intra ∈ [0,1]`

**Integration**: Extends `evaluation/table_eval.py` functions, adds aggregation logic

---

#### Phase 3: Inter-Table Structure Scoring (EXTEND EXISTING)

**File**: `evaluation/multi_table_eval.py` (extend)

**Key Functions**:

1. **`compute_inter_structure_score(real_ir: LogicalIR, synth_ir: LogicalIR, real_dfs: Dict, synth_dfs: Dict, table_mapping: Dict) -> float`**:
   - For each real FK relationship:
     - **Referential Integrity**: Use existing `integrity.fk_coverage()` or `relational_eval.fk_coverage_duckdb()` → `r_RI`
     - **Cardinality**: Use existing `relational_eval.degree_histogram()` for both real and synth, compare with `relational_eval.degree_distribution_divergence()` → `r_card`
     - **Inter-Table Trends**: Join tables, compute column-pair metrics on joined result → `r_trend`
   - Combine: `r_rel = α·r_RI + β·r_card + γ·r_trend`
   - Average over all relationships: `S_structure,inter = mean(r_rel)`
   - Return: `S_structure,inter ∈ [0,1]`

**Integration**: Extends `evaluation/relational_eval.py` functions

---

#### Phase 4: Utility Scoring (NEW)

**File**: `evaluation/utility_eval.py` (NEW)

**Key Functions**:

1. **`compute_local_utility(real_dfs: Dict, synth_dfs: Dict, table_mapping: Dict, column_mappings: Dict, target_columns: Dict[str, str]) -> float`**:
   - For each table with a target column:
     - Train model on real data, test on real held-out → `M_real`
     - Train model on synthetic data, test on real held-out → `M_syn`
     - Normalize: `u = max(0, min(1, M_syn/(M_real+ε)))`
   - Average across tables → `S_utility,local`
   - Return: `S_utility,local ∈ [0,1]`

2. **`compute_relational_utility(real_ir: LogicalIR, synth_ir: LogicalIR, real_dfs: Dict, synth_dfs: Dict, join_recipes: List[Dict]) -> float`**:
   - For each join recipe (from real schema):
     - Build training dataset via joins (real and synthetic)
     - Train models, evaluate → `M_real,rel`, `M_syn,rel`
     - Normalize: `u = max(0, min(1, M_syn,rel/(M_real,rel+ε)))`
   - Average → `S_utility,rel`
   - Return: `S_utility,rel ∈ [0,1]`

3. **`compute_query_utility(real_dfs: Dict, synth_dfs: Dict, queries: List[str]) -> float`**:
   - For each query:
     - Run on real DB → `a_R`
     - Run on synthetic DB → `a_S`
     - Compute relative error: `error = |a_R - a_S| / (|a_R| + ε)`
     - Convert to score: `s_q = 1 - min(error, 1.0)`
   - Average → `S_utility,queries`
   - Return: `S_utility,queries ∈ [0,1]`

**Dependencies**: 
- `sklearn` for ML models (already available)
- `duckdb` for query execution (already available)

---

#### Phase 5: Main Evaluation Function (NEW)

**File**: `evaluation/multi_table_eval.py` (main function)

**Key Function**:

```python
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
    
    Returns:
        MultiTableEvaluationReport with all scores
    """
```

**Integration**: New main entry point, calls all the above functions

---

### File Structure

```
nl2data/src/nl2data/evaluation/
├── schema_matching.py          # NEW - Fuzzy schema matching
├── multi_table_eval.py          # NEW - Main multi-table evaluation
├── utility_eval.py              # NEW - ML and query utility scoring
├── schema_eval.py               # EXISTING - Extend if needed
├── table_eval.py                # EXISTING - Use as-is
├── relational_eval.py            # EXISTING - Extend for inter-table trends
├── integrity.py                 # EXISTING - Use as-is
├── workload.py                  # EXISTING - Use as-is
├── report_models.py             # EXISTING - Add MultiTableEvaluationReport model
└── report_builder.py            # EXISTING - Extend to support multi-table
```

### Dependencies to Add

**New Python Packages** (if not already present):
- `scipy.optimize` - For Hungarian algorithm (bipartite matching)
- `difflib` or `fuzzywuzzy` - For string similarity (name matching)
- `sdmetrics` (optional) - For advanced quality metrics if needed

**Existing Dependencies** (already available):
- `scipy.stats` - For statistical tests
- `sklearn` - For ML utility tasks
- `duckdb` - For query execution
- `networkx` - For schema graphs
- `pandas`, `numpy` - Data manipulation

### Implementation Checklist

**Phase 1: Schema Matching**
- [ ] Implement `match_tables()` with name similarity, row count, PK/FK patterns
- [ ] Implement `match_columns()` with name, type, distribution similarity
- [ ] Add bipartite matching algorithm (Hungarian or greedy)
- [ ] Unit tests for matching logic

**Phase 2: Schema & Marginal Scoring**
- [ ] Implement `compute_schema_score()` using existing marginal metrics
- [ ] Add coverage penalty computation
- [ ] Aggregate across tables
- [ ] Unit tests

**Phase 3: Inter-Table Structure**
- [ ] Extend `compute_inter_structure_score()` using existing FK/degree metrics
- [ ] Add inter-table trend computation (join + column-pair metrics)
- [ ] Unit tests

**Phase 4: Utility Scoring**
- [ ] Implement `compute_local_utility()` with ML tasks
- [ ] Implement `compute_relational_utility()` with join-based tasks
- [ ] Implement `compute_query_utility()` with query workload
- [ ] Unit tests

**Phase 5: Integration**
- [ ] Implement main `evaluate_multi_table()` function
- [ ] Add `MultiTableEvaluationReport` model
- [ ] Update CLI to support multi-table evaluation
- [ ] Integration tests with real datasets

### Testing Strategy

1. **Unit Tests**: Each function independently
2. **Integration Tests**: Full pipeline on simple 2-table schemas
3. **Real Dataset Tests**: Test on OpenML datasets with known ground truth
4. **Schema Mismatch Tests**: Test with intentionally mismatched table/column names

### Performance Considerations

- **Schema Matching**: Can be slow for large schemas → use greedy matching for >20 tables
- **ML Utility Tasks**: Can be slow → make optional, use simple models (logistic regression)
- **Query Utility**: Use DuckDB for fast query execution (already integrated)

### Next Steps

1. Start with **Phase 1 (Schema Matching)** - This is the foundation
2. Then **Phase 2 (Schema Scoring)** - Builds on existing infrastructure
3. Then **Phase 3 (Inter-Table Structure)** - Extends existing relational metrics
4. Then **Phase 4 (Utility)** - New but straightforward
5. Finally **Phase 5 (Integration)** - Tie everything together

---

## 📁 Evaluation Folder Reorganization Plan

### Current Structure Analysis

**Existing Files** (11 files):
- `__init__.py` - Package exports
- `config.py` - Evaluation configuration and thresholds
- `report_models.py` - Pydantic models for reports
- `report_builder.py` - Main evaluation function (single-table focused)
- `schema_eval.py` - Schema coverage metrics (exact matching)
- `schema.py` - Schema validation checks (PK/FK validation)
- `table_eval.py` - Per-table metrics (marginals, correlations)
- `relational_eval.py` - Relational metrics (FK coverage, degree distributions)
- `integrity.py` - Referential integrity checks
- `workload.py` - Workload query execution
- `stats.py` - Statistical utility functions

**Issues with Current Structure**:
- Mixed concerns: validation vs evaluation vs metrics
- Single-table focus (no multi-table aggregation)
- No schema matching layer
- No utility scoring
- No clear separation between single-table and multi-table evaluation

### Proposed Reorganized Structure

```
nl2data/src/nl2data/evaluation/
│
├── __init__.py                          # Package exports (updated)
├── config.py                            # Configuration (extended)
│
├── models/                              # NEW - Report models
│   ├── __init__.py
│   ├── single_table.py                  # Single-table report models (moved from report_models.py)
│   └── multi_table.py                   # NEW - Multi-table report models
│
├── matching/                            # NEW - Schema matching layer
│   ├── __init__.py
│   ├── table_matcher.py                 # Table-level matching
│   ├── column_matcher.py                # Column-level matching
│   └── similarity.py                    # Similarity computation utilities
│
├── metrics/                             # REORGANIZED - All metric computations
│   ├── __init__.py
│   ├── schema/                          # Schema-level metrics
│   │   ├── __init__.py
│   │   ├── coverage.py                   # Schema coverage (from schema_eval.py)
│   │   └── validation.py                # Schema validation (from schema.py)
│   │
│   ├── table/                           # Table-level metrics
│   │   ├── __init__.py
│   │   ├── marginals.py                  # Marginal distributions (from table_eval.py)
│   │   ├── correlations.py              # Column pair correlations (from table_eval.py)
│   │   └── fidelity.py                   # Table fidelity scoring (from table_eval.py)
│   │
│   ├── relational/                      # Relational metrics
│   │   ├── __init__.py
│   │   ├── integrity.py                 # FK integrity (from integrity.py, relational_eval.py)
│   │   ├── degrees.py                   # Degree distributions (from relational_eval.py)
│   │   └── joins.py                     # Join selectivity (from relational_eval.py)
│   │
│   └── utility/                          # NEW - Utility metrics
│       ├── __init__.py
│       ├── ml_tasks.py                   # ML task evaluation
│       └── queries.py                    # Query-level utility
│
├── aggregation/                         # NEW - Multi-table aggregation
│   ├── __init__.py
│   ├── schema_score.py                  # S_schema computation
│   ├── structure_score.py               # S_structure,intra and S_structure,inter
│   └── utility_score.py                 # S_utility computation
│
├── evaluators/                          # REORGANIZED - Main evaluation functions
│   ├── __init__.py
│   ├── single_table.py                  # Single-table evaluator (refactored from report_builder.py)
│   └── multi_table.py                   # NEW - Multi-table evaluator
│
├── execution/                           # REORGANIZED - Execution and utilities
│   ├── __init__.py
│   ├── workload.py                      # Workload execution (moved from workload.py)
│   └── stats.py                         # Statistical utilities (moved from stats.py)
│
└── utils/                               # NEW - Shared utilities
    ├── __init__.py
    └── normalization.py                 # Score normalization utilities
```

### Detailed File Organization

#### 1. Root Level

**`__init__.py`** (Updated):
```python
# Export both single-table and multi-table evaluators
from .evaluators.single_table import evaluate as evaluate_single_table
from .evaluators.multi_table import evaluate_multi_table
from .config import EvaluationConfig, MultiTableEvalConfig
from .models.single_table import EvaluationReport, ...
from .models.multi_table import MultiTableEvaluationReport, ...
```

**`config.py`** (Extended):
- Keep existing `EvaluationConfig`, `EvalThresholds`
- Add `MultiTableEvalConfig` with:
  - Schema matching thresholds
  - Coverage penalty weights
  - Utility task configurations
  - Global score weights

#### 2. `models/` - Report Models

**`models/single_table.py`** (Moved from `report_models.py`):
- `MetricResult`
- `ColumnReport`
- `TableReport`
- `WorkloadReport`
- `EvaluationReport`

**`models/multi_table.py`** (NEW):
- `TableMatch` - Matched table pair
- `ColumnMatch` - Matched column pair
- `SchemaMatchResult` - Complete schema matching result
- `TableScore` - Per-table score breakdown
- `RelationshipScore` - Per-relationship score
- `MultiTableEvaluationReport` - Complete multi-table report

#### 3. `matching/` - Schema Matching

**`matching/table_matcher.py`** (NEW):
- `match_tables()` - Main table matching function
- `compute_table_similarity()` - Similarity scoring
- `solve_bipartite_matching()` - Hungarian/greedy algorithm

**`matching/column_matcher.py`** (NEW):
- `match_columns()` - Main column matching function
- `compute_column_similarity()` - Similarity scoring

**`matching/similarity.py`** (NEW):
- `name_similarity()` - String similarity (Jaccard, edit distance)
- `type_compatibility()` - Type matching
- `distribution_similarity()` - Distribution comparison

#### 4. `metrics/` - Metric Computations

**`metrics/schema/coverage.py`** (Refactored from `schema_eval.py`):
- `schema_coverage()` - Exact name matching (keep for backward compat)
- `compute_coverage_factors()` - Coverage penalty computation

**`metrics/schema/validation.py`** (Moved from `schema.py`):
- `check_pk_fk()` - Schema validation

**`metrics/table/marginals.py`** (Refactored from `table_eval.py`):
- `numeric_marginals()` - KS test, Wasserstein
- `categorical_marginals()` - Chi-square test

**`metrics/table/correlations.py`** (Refactored from `table_eval.py`):
- `correlation_metrics()` - Pearson, Spearman
- `mutual_information()` - MI for categorical pairs

**`metrics/table/fidelity.py`** (Refactored from `table_eval.py`):
- `table_fidelity_score()` - Aggregate table score

**`metrics/relational/integrity.py`** (Refactored from `integrity.py`, `relational_eval.py`):
- `fk_coverage()` - Referential integrity
- `fk_coverage_duckdb()` - DuckDB-based FK check

**`metrics/relational/degrees.py`** (Refactored from `relational_eval.py`):
- `degree_histogram()` - Children per parent
- `degree_distribution_divergence()` - Compare degree distributions

**`metrics/relational/joins.py`** (Refactored from `relational_eval.py`):
- `join_selectivity()` - Join selectivity metrics

**`metrics/utility/ml_tasks.py`** (NEW):
- `evaluate_local_ml_task()` - Single-table ML evaluation
- `evaluate_relational_ml_task()` - Multi-table ML evaluation

**`metrics/utility/queries.py`** (NEW):
- `evaluate_query_utility()` - Query-level utility scoring

#### 5. `aggregation/` - Multi-Table Aggregation

**`aggregation/schema_score.py`** (NEW):
- `compute_schema_score()` - S_schema computation
- `aggregate_table_schema_scores()` - Per-table aggregation

**`aggregation/structure_score.py`** (NEW):
- `compute_intra_structure_score()` - S_structure,intra
- `compute_inter_structure_score()` - S_structure,inter

**`aggregation/utility_score.py`** (NEW):
- `compute_utility_score()` - S_utility computation
- `aggregate_utility_scores()` - Combine local, relational, query utility

#### 6. `evaluators/` - Main Evaluation Functions

**`evaluators/single_table.py`** (Refactored from `report_builder.py`):
- `evaluate()` - Main single-table evaluation (backward compatible)
- Uses metrics from `metrics/` modules

**`evaluators/multi_table.py`** (NEW):
- `evaluate_multi_table()` - Main multi-table evaluation
- Orchestrates: matching → metrics → aggregation → report

#### 7. `execution/` - Execution and Utilities

**`execution/workload.py`** (Moved from `workload.py`):
- `run_workloads()` - Query execution
- `_generate_query()` - Query generation

**`execution/stats.py`** (Moved from `stats.py`):
- `zipf_fit()`, `chi_square_test()`, `ks_test()`, etc.
- `gini_coefficient()`, `top_k_share()`

#### 8. `utils/` - Shared Utilities

**`utils/normalization.py`** (NEW):
- `normalize_score()` - Score normalization to [0,1]
- `clip_score()` - Clip scores to valid range

### Migration Strategy

#### Phase 1: Create New Structure (No Breaking Changes)

1. **Create new directories**:
   ```
   mkdir -p evaluation/{models,matching,metrics/{schema,table,relational,utility},aggregation,evaluators,execution,utils}
   ```

2. **Move files** (keep originals for now):
   - `report_models.py` → `models/single_table.py`
   - `schema.py` → `metrics/schema/validation.py`
   - `stats.py` → `execution/stats.py`
   - `workload.py` → `execution/workload.py`

3. **Split existing files**:
   - `schema_eval.py` → `metrics/schema/coverage.py` (keep exact matching)
   - `table_eval.py` → `metrics/table/{marginals,correlations,fidelity}.py`
   - `relational_eval.py` → `metrics/relational/{integrity,degrees,joins}.py`
   - `integrity.py` → Merge into `metrics/relational/integrity.py`

4. **Update imports** in moved files

#### Phase 2: Implement New Features

1. **Schema Matching**:
   - Create `matching/` module
   - Implement table and column matchers

2. **Multi-Table Aggregation**:
   - Create `aggregation/` module
   - Implement score computation functions

3. **Utility Scoring**:
   - Create `metrics/utility/` module
   - Implement ML and query utility

4. **Multi-Table Evaluator**:
   - Create `evaluators/multi_table.py`
   - Integrate all components

#### Phase 3: Update Exports and Maintain Backward Compatibility

1. **Update `__init__.py`**:
   - Export both `evaluate()` (single-table) and `evaluate_multi_table()`
   - Re-export moved models for backward compatibility

2. **Update `report_builder.py`** (temporary wrapper):
   ```python
   # Keep for backward compatibility
   from .evaluators.single_table import evaluate
   ```

3. **Update all internal imports** across codebase

#### Phase 4: Cleanup

1. **Remove old files** after migration complete
2. **Update documentation**
3. **Update tests**

### File Size Estimates

**New Files** (to be created):
- `matching/` - ~500 lines total
- `aggregation/` - ~400 lines total
- `metrics/utility/` - ~300 lines total
- `evaluators/multi_table.py` - ~200 lines
- `models/multi_table.py` - ~150 lines
- `utils/normalization.py` - ~50 lines

**Refactored Files** (split from existing):
- Existing ~2000 lines → Reorganized into ~2500 lines (with new features)

**Total**: ~4000 lines (well-organized, modular)

### Benefits of This Structure

1. **Clear Separation of Concerns**:
   - Matching vs Metrics vs Aggregation vs Evaluation
   - Single-table vs Multi-table clearly separated

2. **Easy to Extend**:
   - Add new metrics in `metrics/`
   - Add new aggregation methods in `aggregation/`
   - Add new evaluators in `evaluators/`

3. **Backward Compatible**:
   - Single-table evaluation still works
   - Can migrate gradually

4. **Testable**:
   - Each module can be tested independently
   - Clear interfaces between modules

5. **Maintainable**:
   - Related code grouped together
   - Easy to find and modify specific functionality

### Implementation Order

1. **Week 1**: Create structure, move existing files (Phase 1)
2. **Week 2**: Implement schema matching (Phase 2, Part 1)
3. **Week 3**: Implement aggregation and utility (Phase 2, Parts 2-3)
4. **Week 4**: Integrate multi-table evaluator, update exports (Phase 2, Part 4 + Phase 3)
5. **Week 5**: Testing, cleanup, documentation (Phase 4)

---

## 🛠️ Implementation Patterns

### Adding a New Distribution Type

1. **IR Model** (`ir/generation.py`):
   - Add Pydantic model with `kind: Literal["..."]`
   - Add to `Distribution` union (line ~178-194)
   - Add validators if needed

2. **Sampler** (`generation/distributions/numeric.py` or new file):
   - Implement `BaseSampler` subclass
   - Implement `sample(n, **kwargs)` method
   - Use `rng` from kwargs

3. **Factory** (`generation/distributions/factory.py`):
   - Add `elif isinstance(dist, DistNewType):` case
   - Return sampler instance

4. **Validation** (`ir/validators.py`):
   - Usually no changes needed (Pydantic handles it)

5. **Prompts** (`prompts/roles/dist_system.txt`):
   - Add example and guidance

**Sanity Checklist**:
- [ ] IR model compiles
- [ ] Factory returns sampler
- [ ] Sampler passes unit test
- [ ] Validator accepts it
- [ ] Prompt mentions it

---

### Adding a New DSL Function

1. **Allowlist** (`derived_program.py`):
   - Add to `ALLOWED_FUNCS` (line ~8-54)

2. **Implementation** (`derived_eval.py`):
   - Add function to `build_env()` (line ~10-287)
   - Handle both `pd.Series` and scalar inputs
   - Return appropriate type

3. **Testing**:
   - Test with Series inputs
   - Test with scalar inputs
   - Test edge cases (empty, null, etc.)

**Sanity Checklist**:
- [ ] Added to allowlist
- [ ] Function implemented (vectorized)
- [ ] Handles Series and scalars
- [ ] Test passes

---

## 🐛 Common Pitfalls

1. **Forgetting Distribution union**: Always add new distribution types to `Distribution` union in `ir/generation.py` (line ~178-194)

2. **Series vs scalar**: Always check `isinstance(x, pd.Series)` in DSL functions

3. **Missing factory case**: Every new distribution needs a factory case in `get_sampler()`

4. **Not updating prompts**: New features won't be used unless prompts explicitly request them

5. **Missing validation**: Add Pydantic validators for parameter ranges (e.g., `alpha > 0` for Pareto)

6. **Not testing edge cases**: Test with empty DataFrames, single rows, null values

---

## 📝 Changelog / Completed Work

### Phase 1: DSL Extensions ✅
- Added `in`/`not in` operators (AST handling + evaluation)
- Added cast functions: `int()`, `float()`, `bool()`, `str()`
- Added distribution functions: `normal()`, `lognormal()`, `pareto()` (DSL)
- Added `case_when` macro
- Added helper functions: `between()`, `geo_distance()`, `ts_diff()`, `overlap_days()`

### Phase 2: Window Functions ✅
- Added `DistWindow` IR model
- Created `window_eval.py` with full window function support
- Integrated as Phase 3 in fact generator
- Supports `RANGE` and `ROWS` frames, partitioning, ordering

### Phase 3: Event/Incident System ✅
- Added `EventSpec` and `EventEffect` IR models
- Created `event_eval.py` with event application logic
- Integrated as Phase 1.5 in fact generator
- Supports `multiply_distribution`, `add_offset`, `set_value` effects

### Phase 4: Infrastructure ✅
- Integrated agent repair loop system into Orchestrator
- Integrated quality metrics tracking
- Integrated constraint enforcement (Phase 1.7)
- Added basic nuance coverage checker
- Removed deprecated `DerivedSampler`

### Phase 5: Additional Distributions ✅
- Added Lognormal and Pareto distributions (IR models, samplers, factory)
- Added Mixture distribution (IR model, sampler, factory)
- Added Poisson and Exponential distributions (IR models, samplers, factory)
- All distributions integrated into prompt system

### Phase 6: Prompt Fixes ✅
- Fixed prompt inconsistencies (ALLOWED_FUNCS verification, lag/lead clarification)
- Reduced domain bias (made fraud patterns conditional)
- Clarified window function usage in both prompt files

---

## 🎯 Next Steps

1. **🔥 Multi-Table Relational Evaluation Framework** - **TOP PRIORITY**
   - Implement comprehensive evaluation system for benchmarking synthetic DBs against real DBs
   - See detailed design in "Multi-Table Relational Evaluation Framework" section above
   - Includes schema matching, structural scoring, and utility evaluation
   
2. **Integration Testing** - Test with 2-3 queries to verify LLMs use functions correctly
3. **Optional Enhancements** - Enhance nuance coverage, add additional distributions if needed
4. **Future Tasks** - Conditional distributions, state machines, `within_days_of()` helper, WorkloadIR integration (lower priority)
