# NL2Data Project - Comprehensive Overview

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
   - [Intermediate Representations (IRs)](#1-intermediate-representations-irs)
   - [Multi-Agent System](#2-multi-agent-system)
   - [Data Generation Engine](#3-data-generation-engine)
   - [Evaluation Framework](#4-evaluation-framework)
5. [Data Flow](#data-flow)
6. [Key Design Decisions](#key-design-decisions)
7. [Implementation Details](#implementation-details)
8. [Usage Examples](#usage-examples)

---

## Executive Summary

**NL2Data** is a multi-agent system that converts natural language descriptions into synthetic relational datasets. The system uses Large Language Models (LLMs) to progressively refine specifications through multiple Intermediate Representations (IRs), ultimately generating realistic CSV data with proper schema, distributions, and referential integrity.

### Key Capabilities
- **Natural Language Processing**: Converts free-form text descriptions into structured database schemas
- **Multi-Agent Pipeline**: Specialized agents handle different aspects (conceptual design, logical schema, distributions, workloads)
- **Realistic Data Generation**: Supports Zipf, seasonal, categorical, and derived column distributions
- **Scalable Generation**: Handles millions of rows with streaming/chunked generation
- **Comprehensive Evaluation**: Validates schema correctness, statistical properties, and workload performance
- **Constraint System**: Functional dependencies, implications, and composite primary keys
- **Self-Healing Pipeline**: Automatic repair loops with QA feedback
- **Provider System**: Faker, Mimesis, and geo-lookup providers for realistic data
- **Advanced Evaluation**: Schema coverage, relational metrics, and table-level fidelity scores

### Technology Stack
- **Python 3.10+**: Core language
- **Pydantic 2.0+**: Type-safe IR models and validation
- **LLM APIs**: 
  - OpenAI (GPT-4, GPT-3.5)
  - Google Gemini
  - Local OpenAI-compatible APIs (Ollama, vLLM, etc.)
- **NumPy/Pandas**: Data generation and manipulation
- **DuckDB**: In-memory SQL engine for workload evaluation
- **Streamlit**: Web UI for interactive use
- **Typer**: CLI framework
- **Python-dotenv**: Environment variable management
- **SciPy**: Statistical tests (KS test, Wasserstein distance, chi-square)
- **scikit-learn**: Mutual information and correlation metrics
- **NetworkX**: Graph-based schema evaluation (optional)
- **Faker/Mimesis**: Realistic data providers

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NL2Data System Architecture                        │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    INPUT LAYER                                │   │
│  │                                                               │   │
│  │  Natural Language Description                                 │   │
│  │  "Generate a retail sales dataset with 5M rows,               │   │
│  │   product_id follows Zipf distribution..."                   │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              MULTI-AGENT IR GENERATION LAYER                    │   │
│  │                                                               │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│  │  │ Manager  │───▶│Conceptual│───▶│ Logical  │              │   │
│  │  │  Agent   │    │ Designer │    │ Designer │              │   │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘              │   │
│  │       │              │                │                     │   │
│  │       ▼              ▼                ▼                     │   │
│  │  RequirementIR  ConceptualIR    LogicalIR                  │   │
│  │                                                               │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│  │  │Distrib.  │───▶│Workload   │───▶│   QA     │              │   │
│  │  │ Engineer │    │ Designer  │    │Compiler  │              │   │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘              │   │
│  │       │              │                │                     │   │
│  │       ▼              ▼                ▼                     │   │
│  │  GenerationIR   WorkloadIR      DatasetIR                │   │
│  │                                                               │   │
│  │  Blackboard (Shared State)                                  │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              DATA GENERATION LAYER                            │   │
│  │                                                               │   │
│  │  ┌──────────────┐                                            │   │
│  │  │   Derived    │  Compile expressions, compute dependencies│   │
│  │  │   Registry   │                                            │   │
│  │  └──────┬───────┘                                            │   │
│  │         │                                                     │   │
│  │         ▼                                                     │   │
│  │  ┌──────────────┐    ┌──────────────┐                      │   │
│  │  │  Dimension    │    │   Fact Table  │                      │   │
│  │  │  Generator    │    │   Generator  │                      │   │
│  │  │  (In-Memory)  │    │  (Streaming)  │                      │   │
│  │  └──────┬───────┘    └──────┬───────┘                      │   │
│  │         │                   │                               │   │
│  │         ▼                   ▼                               │   │
│  │  ┌──────────────┐    ┌──────────────┐                      │   │
│  │  │ Distribution │    │ Distribution │                      │   │
│  │  │  Samplers    │    │  Samplers    │                      │   │
│  │  │  (Uniform,   │    │  (Zipf, FK   │                      │   │
│  │  │   Normal,     │    │   sampling)  │                      │   │
│  │  │   Categorical)│   │              │                      │   │
│  │  └──────┬───────┘    └──────┬───────┘                      │   │
│  │         │                   │                               │   │
│  │         ▼                   ▼                               │   │
│  │  ┌──────────────┐    ┌──────────────┐                      │   │
│  │  │ Constraint    │    │ Constraint    │                      │   │
│  │  │ Enforcement   │    │ Enforcement   │                      │   │
│  │  │ (FD, Impl,    │    │ (FD, Impl,    │                      │   │
│  │  │  Nullability) │    │  Nullability) │                      │   │
│  │  └──────┬───────┘    └──────┬───────┘                      │   │
│  │         │                   │                               │   │
│  │         ▼                   ▼                               │   │
│  │  CSV Files (Small)      CSV Files (Large, Streamed)          │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              EVALUATION LAYER                                  │   │
│  │                                                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │   │
│  │  │   Schema     │  │ Statistical  │  │  Workload     │    │   │
│  │  │  Validation  │  │  Validation   │  │  Execution    │    │   │
│  │  │  (PK/FK)     │  │  (Zipf, KS,   │  │  (DuckDB)     │    │   │
│  │  │              │  │   Chi-square) │  │               │    │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │   │
│  │         │                 │                  │            │   │
│  │         └─────────────────┴──────────────────┘            │   │
│  │                            │                                │   │
│  │                            ▼                                │   │
│  │                    EvaluationReport                         │   │
│  │                    - Schema metrics                        │   │
│  │                    - Statistical metrics                    │   │
│  │                    - Workload metrics                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

**Key Data Flow**:
1. NL → IR: Multi-agent pipeline transforms natural language to structured IRs
2. IR → Data: Generation engine converts IRs to CSV files
3. Data → Evaluation: Framework validates generated data against specifications
```

### Three Core Subsystems

1. **NL → IR (Multi-Agent System)**
   - Natural language → structured IR models
   - Progressive refinement through specialized agents
   - Prompt-file driven for easy iteration

2. **IR → Data (Generation Engine)**
   - Converts IR specifications to CSV files
   - Supports multiple distribution types
   - Streaming generation for large datasets

3. **Evaluation**
   - Schema validation (PK/FK integrity)
   - Statistical alignment (distribution fitting)
   - Workload performance testing

---

## Project Structure

### Directory Tree

```
Project v3/
├── nl2data/                          # Main package
│   ├── pyproject.toml               # Package configuration
│   ├── README.md                     # Package documentation
│   ├── QUICKSTART.md                 # Quick start guide
│   ├── example_description.txt      # Example NL input
│   │
│   ├── src/nl2data/                 # Source code
│   │   ├── __init__.py              # Package initialization
│   │   │
│   │   ├── config/                  # Configuration management
│   │   │   ├── __init__.py
│   │   │   ├── settings.py          # Pydantic settings (LLM, generation params)
│   │   │   └── logging.py           # Logging configuration
│   │   │
│   │   ├── prompts/                 # Prompt templates
│   │   │   ├── __init__.py
│   │   │   ├── loader.py            # Prompt file loader
│   │   │   └── roles/               # Agent-specific prompts
│   │   │       ├── manager_system.txt
│   │   │       ├── manager_user.txt
│   │   │       ├── conceptual_system.txt
│   │   │       ├── conceptual_user.txt
│   │   │       ├── logical_system.txt
│   │   │       ├── logical_user.txt
│   │   │       ├── dist_system.txt
│   │   │       ├── dist_user.txt
│   │   │       ├── workload_system.txt
│   │   │       ├── workload_user.txt
│   │   │       ├── qa_system.txt
│   │   │       └── qa_user.txt
│   │   │
│   │   ├── ir/                      # Intermediate Representations
│   │   │   ├── __init__.py          # IR exports
│   │   │   ├── requirement.py       # RequirementIR model
│   │   │   ├── conceptual.py        # ConceptualIR (ER model)
│   │   │   ├── logical.py           # LogicalIR (relational schema)
│   │   │   ├── generation.py        # GenerationIR (distributions)
│   │   │   ├── workload.py          # WorkloadIR (query specs)
│   │   │   ├── dataset.py           # DatasetIR (combined)
│   │   │   └── validators.py        # IR validation functions
│   │   │
│   │   ├── agents/                  # Multi-agent system
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # BaseAgent, Blackboard
│   │   │   ├── orchestrator.py     # Agent orchestration
│   │   │   ├── runner.py            # Agent runner with repair loop
│   │   │   ├── tools/               # Agent utilities
│   │   │   │   ├── __init__.py
│   │   │   │   ├── llm_client.py    # LLM API wrapper (OpenAI/Gemini/local)
│   │   │   │   ├── json_parser.py   # Robust JSON extraction
│   │   │   │   ├── retry.py         # Retry utilities with exponential backoff
│   │   │   │   ├── error_handling.py # Standardized error handling
│   │   │   │   └── agent_retry.py   # Common agent retry logic utility
│   │   │   └── roles/               # Specialized agents
│   │   │       ├── __init__.py
│   │   │       ├── manager.py       # ManagerAgent (NL → RequirementIR)
│   │   │       ├── conceptual_designer.py  # ConceptualDesigner
│   │   │       ├── logical_designer.py     # LogicalDesigner
│   │   │       ├── dist_engineer.py         # DistributionEngineer
│   │   │       ├── workload_designer.py     # WorkloadDesigner
│   │   │       └── qa_compiler.py           # QACompilerAgent
│   │   │
│   │   ├── generation/              # Data generation engine
│   │   │   ├── __init__.py
│   │   │   ├── derived_registry.py  # Derived column dependency tracking
│   │   │   ├── derived_program.py   # AST-based expression compiler
│   │   │   ├── derived_eval.py      # Expression evaluator
│   │   │   ├── type_enforcement.py  # Type enforcement utilities
│   │   │   ├── column_sampling.py   # Column sampling utilities
│   │   │   ├── ir_helpers.py        # IR extraction utilities
│   │   │   ├── uniqueness.py        # Uniqueness enforcement utilities
│   │   │   ├── constants.py          # Generation constants
│   │   │   ├── distributions/       # Distribution samplers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py          # BaseSampler interface
│   │   │   │   ├── factory.py       # Sampler factory
│   │   │   │   ├── numeric.py       # UniformSampler, NormalSampler
│   │   │   │   ├── categorical.py   # CategoricalSampler
│   │   │   │   ├── zipf.py          # ZipfSampler
│   │   │   │   ├── seasonal.py     # SeasonalDateSampler
│   │   │   │   └── derived.py       # DerivedSampler (legacy)
│   │   │   └── engine/              # Generation pipeline
│   │   │       ├── __init__.py
│   │   │       ├── pipeline.py      # Main generation pipeline
│   │   │       ├── dim_generator.py  # Dimension table generator
│   │   │       ├── fact_generator.py # Fact table generator (streaming)
│   │   │       └── writer.py        # CSV writer utilities
│   │   │
│   │   ├── evaluation/              # Evaluation framework
│   │   │   ├── __init__.py
│   │   │   ├── config.py            # EvaluationConfig
│   │   │   ├── report_builder.py    # Main evaluation function
│   │   │   ├── report_models.py     # EvaluationReport models
│   │   │   ├── schema.py            # Schema validation (PK/FK)
│   │   │   ├── integrity.py         # Referential integrity checks
│   │   │   ├── stats.py              # Statistical tests
│   │   │   └── workload.py          # Workload query execution
│   │   │
│   │   ├── utils/                   # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── agent_factory.py     # Agent sequence factory functions
│   │   │   ├── ir_io.py             # IR load/save utilities
│   │   │   └── data_loader.py       # Data loading utilities
│   │   │
│   │   └── cli/                     # Command-line interface
│   │       ├── __init__.py
│   │       └── app.py               # Typer CLI app
│   │
│   ├── scripts/                     # Entry point scripts
│   │   └── nl2data.py              # CLI entry point
│   │
│   └── tests/                       # Test suite
│       ├── __init__.py
│       └── test_ir_models.py        # IR model tests
│
├── ui_streamlit/                    # Streamlit web UI
│   ├── app.py                       # Main Streamlit app
│   ├── pipeline_runner.py          # Pipeline adapter for UI
│   ├── step_models.py               # Step logging models
│   ├── requirements.txt             # UI dependencies
│   └── generated_data/              # Generated CSV files
│       └── latest_run/              # Latest generation output
│
├── test_all_queries.py              # Batch testing script
├── test_utils/                      # Test utilities module
│   ├── __init__.py
│   ├── query_parser.py              # Query parsing utilities
│   ├── cache_manager.py            # Data caching/existence checks
│   ├── report_formatter.py         # Evaluation report formatting
│   └── test_helpers.py             # Test helper functions
├── example queries.txt              # Example NL descriptions
├── Instructions.md                  # Detailed instructions
├── UI.md                            # UI documentation
└── PROJECT_OVERVIEW.md              # This file
```

---

## Core Components

### 1. Intermediate Representations (IRs)

The system uses a progressive refinement approach with 6 IR models:

**IR Transformation Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    IR Transformation Pipeline                      │
│                                                                  │
│  Natural Language Input                                          │
│  "Generate a retail sales dataset with 5M rows..."               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. RequirementIR                                          │  │
│  │    - domain: "retail"                                    │  │
│  │    - narrative: "retail sales dataset..."                │  │
│  │    - scale: [{"table": "fact_sales", "row_count": 5000000}]│  │
│  │    - distributions: [{"target": "product_id", "family": "zipf"}]│  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. ConceptualIR (ER Model)                               │  │
│  │    - entities: [                                          │  │
│  │        {name: "Sale", attributes: [...]},                 │  │
│  │        {name: "Product", attributes: [...]}               │  │
│  │      ]                                                     │  │
│  │    - relationships: [                                      │  │
│  │        {name: "has", participants: ["Sale", "Product"],  │  │
│  │         cardinality: "many_to_one"}                       │  │
│  │      ]                                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. LogicalIR (Relational Schema)                         │  │
│  │    - tables: {                                            │  │
│  │        "fact_sales": {                                    │  │
│  │          columns: [                                       │  │
│  │            {name: "sale_id", sql_type: "INT64",           │  │
│  │             role: "primary_key"},                         │  │
│  │            {name: "product_id", sql_type: "INT64",       │  │
│  │             role: "foreign_key",                          │  │
│  │             references: "dim_product.product_id"}         │  │
│  │          ],                                               │  │
│  │          primary_key: ["sale_id"],                       │  │
│  │          foreign_keys: [...]                             │  │
│  │        }                                                  │  │
│  │      }                                                    │  │
│  │    - constraints: {                                      │  │
│  │        fds: [...],                                       │  │
│  │        implications: [...],                              │  │
│  │        composite_pks: [...]                             │  │
│  │      }                                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. GenerationIR (Distribution Specs)                     │  │
│  │    - columns: [                                          │  │
│  │        {                                                  │  │
│  │          table: "fact_sales",                            │  │
│  │          column: "product_id",                            │  │
│  │          distribution: {                                 │  │
│  │            kind: "zipf",                                 │  │
│  │            s: 1.2,                                       │  │
│  │            n: 10000                                      │  │
│  │          }                                                │  │
│  │        },                                                 │  │
│  │        {                                                    │  │
│  │          table: "fact_sales",                            │  │
│  │          column: "total_price",                          │  │
│  │          distribution: {                                 │  │
│  │            kind: "derived",                              │  │
│  │            expression: "price * quantity"               │  │
│  │          }                                                │  │
│  │        }                                                  │  │
│  │      ]                                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. WorkloadIR (Query Specs)                             │  │
│  │    - targets: [                                          │  │
│  │        {                                                  │  │
│  │          type: "group_by",                               │  │
│  │          query_hint: "GROUP BY product_id",              │  │
│  │          expected_skew: "high"                          │  │
│  │        },                                                 │  │
│  │        {                                                  │  │
│  │          type: "join",                                   │  │
│  │          join_graph: ["fact_sales", "dim_product"]      │  │
│  │        }                                                  │  │
│  │      ]                                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. DatasetIR (Combined)                                   │  │
│  │    - logical: LogicalIR                                   │  │
│  │    - generation: GenerationIR                             │  │
│  │    - workload: WorkloadIR (optional)                      │  │
│  │    - description: string (optional)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  Ready for Data Generation                                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.1 RequirementIR (`ir/requirement.py`)

**Purpose**: Captures structured requirements from natural language.

```python
class RequirementIR(BaseModel):
    domain: Optional[str] = None              # Domain name (e.g., "retail", "healthcare")
    narrative: str                           # Main requirement text
    tables_hint: Optional[str] = None         # Hint about expected tables
    scale: List[ScaleHint] = []               # Table size hints
    distributions: List[DistributionHint] = []  # Distribution hints
    nonfunctional_goals: List[str] = []       # Non-functional requirements
```

**Key Features**:
- Extracts domain, scale, and distribution hints from NL
- Provides foundation for subsequent IRs
- Used by ManagerAgent to structure initial requirements

#### 1.2 ConceptualIR (`ir/conceptual.py`)

**Purpose**: ER-style conceptual model (entities, attributes, relationships).

```python
class ConceptualIR(BaseModel):
    entities: List[Entity]                    # List of entities
    relationships: List[Relationship] = []   # Entity relationships

class Entity(BaseModel):
    name: str                                 # Entity name
    attributes: List[Attribute] = []          # Entity attributes

class Attribute(BaseModel):
    name: str                                 # Attribute name
    kind: Literal["identifier", "numeric",    # Attribute type
                  "categorical", "text",
                  "datetime", "boolean"] = "text"

class Relationship(BaseModel):
    name: str                                 # Relationship name
    participants: List[str]                   # Entity names
    cardinality: str = "many_to_one"          # Relationship cardinality
```

**Key Features**:
- High-level conceptual design
- Independent of implementation details
- Used by ConceptualDesigner agent

#### 1.3 LogicalIR (`ir/logical.py`)

**Purpose**: Relational schema specification (tables, columns, PK/FK).

```python
class LogicalIR(BaseModel):
    tables: Dict[str, TableSpec]              # Table name → specification
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)  # Advanced constraints
    schema_mode: Literal["oltp", "star", "snowflake"] = "star"  # Schema design pattern

class TableSpec(BaseModel):
    name: str                                 # Table name
    kind: Optional[Literal["fact", "dimension"]]  # Table type
    row_count: Optional[int] = None          # Expected row count
    columns: List[ColumnSpec]                 # Column specifications
    primary_key: List[str] = []              # PK column names
    foreign_keys: List[ForeignKeySpec] = []  # FK specifications

class ColumnSpec(BaseModel):
    name: str                                 # Column name
    sql_type: SQLType                         # SQL data type
    nullable: bool = True                     # Nullable flag
    unique: bool = False                      # Unique constraint (NOTE: not enforced)
    role: Optional[Literal["primary_key",    # Column role
                           "foreign_key",
                           "measure", "attribute"]] = None
    references: Optional[str] = None          # FK reference (e.g., "table.column")

class ForeignKeySpec(BaseModel):
    column: str                               # FK column name
    ref_table: str                            # Referenced table
    ref_column: str                           # Referenced column
```

**Key Features**:
- Complete relational schema definition
- Supports star schema (fact/dimension)
- Primary and foreign key constraints
- **Advanced Constraints**: Functional dependencies, implications, composite PKs
- **Schema Mode**: OLTP, star, or snowflake schema patterns
- **Note**: `unique` flag exists but is not enforced during generation

#### 1.4 GenerationIR (`ir/generation.py`)

**Purpose**: Data generation specifications (distributions, domains).

```python
class GenerationIR(BaseModel):
    columns: List[ColumnGenSpec] = []         # Generation specs per column

class ColumnGenSpec(BaseModel):
    table: str                                # Target table
    column: str                               # Target column
    distribution: Distribution                # Distribution specification

# Distribution types (discriminated union)
Distribution = Union[
    DistUniform,      # Uniform distribution (low, high)
    DistNormal,       # Normal distribution (mean, std)
    DistZipf,         # Zipf distribution (s exponent, n domain size)
    DistSeasonal,     # Seasonal dates (weights, granularity)
    DistCategorical,  # Categorical (values, probabilities)
    DistDerived,      # Derived column (expression, dtype, depends_on)
]
```

**Key Features**:
- Specifies how each column should be generated
- Supports multiple distribution families
- Derived columns with dependency tracking
- **Distribution Preprocessing**: Automatic fixes for common LLM errors:
  - Categorical values converted to strings
  - Date strings in uniform distributions converted to timestamps
  - Missing parameters defaulted appropriately

#### 1.5 WorkloadIR (`ir/workload.py`)

**Purpose**: Query workload specifications for evaluation.

```python
class WorkloadIR(BaseModel):
    targets: List[WorkloadSpec] = []          # Workload specifications

class WorkloadSpec(BaseModel):
    type: Literal["group_by", "join", "filter"]  # Query type
    query_hint: Optional[str] = None          # SQL hint
    expected_skew: Optional[Literal["low", "medium", "high"]] = None
    join_graph: Optional[List[str]] = None    # Tables to join
    selectivity_hint: Optional[Literal["low", "medium", "high"]] = None
```

**Key Features**:
- Defines expected query patterns
- Used for workload evaluation
- Optional component

#### 1.6 DatasetIR (`ir/dataset.py`)

**Purpose**: Final combined IR containing all components.

```python
class DatasetIR(BaseModel):
    logical: LogicalIR                        # Required: schema
    generation: GenerationIR                  # Required: generation specs
    workload: WorkloadIR | None = None        # Optional: workload specs
    description: str | None = None           # Optional: description
```

**Key Features**:
- Combines all IRs into single model
- Used as input to generation engine
- Validated by QACompilerAgent

#### 1.7 ConstraintIR (`ir/constraint_ir.py`)

**Purpose**: Defines advanced constraints beyond basic PK/FK (functional dependencies, implications, composite PKs).

```python
class AtomicCondition(BaseModel):
    """Atomic condition in a structured expression."""
    col: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "is_null", "not_null"]
    value: Any | None = None  # list for "in", single value for others

class ConditionExpr(BaseModel):
    """Structured condition expression (tree of atomic conditions)."""
    kind: Literal["atom", "and", "or", "not"]
    atom: AtomicCondition | None = None
    children: List["ConditionExpr"] = Field(default_factory=list)

class FDConstraint(BaseModel):
    """Functional dependency constraint."""
    table: str
    lhs: List[str]  # determinant columns
    rhs: List[str]  # dependent columns
    mode: Literal["intra_row", "lookup"] = "intra_row"

class ImplicationConstraint(BaseModel):
    """Implication constraint: if condition, then effect."""
    table: str
    condition: ConditionExpr
    effect: ConditionExpr

class CompositePKConstraint(BaseModel):
    """Composite primary key constraint."""
    table: str
    cols: List[str]

class ConstraintSpec(BaseModel):
    """Collection of all constraints for a schema."""
    fds: List[FDConstraint] = Field(default_factory=list)
    implications: List[ImplicationConstraint] = Field(default_factory=list)
    composite_pks: List[CompositePKConstraint] = Field(default_factory=list)
```

**Key Features**:
- **Functional Dependencies**: Expresses relationships like "product_id → product_name"
- **Implications**: Conditional logic (e.g., "if status='cancelled', then shipped_at IS NULL")
- **Composite PKs**: Multi-column primary keys (e.g., (order_id, line_no))
- **Structured Conditions**: Tree-based condition expressions for complex logic
- **Integration**: Embedded in LogicalIR via `constraints` field

**Example Usage**:
```python
# Functional dependency: product_id determines product_name
fd = FDConstraint(
    table="order_items",
    lhs=["product_id"],
    rhs=["product_name"],
    mode="intra_row"
)

# Implication: cancelled orders have no shipment date
impl = ImplicationConstraint(
    table="orders",
    condition=ConditionExpr(
        kind="atom",
        atom=AtomicCondition(col="status", op="eq", value="cancelled")
    ),
    effect=ConditionExpr(
        kind="atom",
        atom=AtomicCondition(col="shipped_at", op="is_null")
    )
)

# Composite PK
cpk = CompositePKConstraint(
    table="order_items",
    cols=["order_id", "line_no"]
)
```

#### 1.8 Schema Mode

**Purpose**: Specifies the database design pattern (OLTP, star schema, snowflake schema).

**Location**: `RequirementIR.schema_mode` and `LogicalIR.schema_mode`

```python
schema_mode: Literal["oltp", "star", "snowflake"] = "star"
```

**Modes**:
- **`"oltp"`**: Normalized OLTP schema (3NF/BCNF)
- **`"star"`**: Star schema (fact table + denormalized dimensions)
- **`"snowflake"`**: Snowflake schema (fact table + normalized dimensions)

**Usage**: Guides LogicalDesigner agent in schema design decisions.

---

### 2. Multi-Agent System

#### 2.1 Blackboard Pattern (`agents/base.py`)

**Purpose**: Shared state between agents.

```python
class Blackboard(BaseModel):
    """Shared blackboard for multi-agent communication."""
    requirement_ir: Optional[RequirementIR] = None
    conceptual_ir: Optional[ConceptualIR] = None
    logical_ir: Optional[LogicalIR] = None
    generation_ir: Optional[GenerationIR] = None
    workload_ir: Optional[WorkloadIR] = None
    dataset_ir: Optional[DatasetIR] = None
```

**Design Rationale**:
- Agents read from and write to blackboard
- Enables sequential pipeline execution
- Type-safe with Pydantic validation

#### 2.2 Base Agent (`agents/base.py`)

```python
class BaseAgent:
    """Base class for all agents."""
    name: str = "base_agent"
    
    def run(self, board: Blackboard) -> Blackboard:
        """Execute agent's task. Must be implemented by subclasses."""
        raise NotImplementedError
```

#### 2.3 Agent Sequence

**ManagerAgent** (`agents/roles/manager.py`)
- **Input**: Natural language description
- **Output**: RequirementIR
- **Process**: 
  1. Loads manager prompts (system + user)
  2. Sends to LLM with NL input
  3. Extracts JSON from response
  4. Validates as RequirementIR

```python
class ManagerAgent(BaseAgent):
    def __init__(self, nl_request: str):
        self.nl_request = nl_request
    
    def run(self, board: Blackboard) -> Blackboard:
        sys_tmpl = load_prompt("roles/manager_system.txt")
        usr_tmpl = load_prompt("roles/manager_user.txt")
        user_content = render_prompt(usr_tmpl, NARRATIVE=self.nl_request)
        
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        raw = chat(messages)
        data = extract_json(raw)
        board.requirement_ir = RequirementIR.model_validate(data)
        return board
```

**ConceptualDesigner** (`agents/roles/conceptual_designer.py`)
- **Input**: RequirementIR
- **Output**: ConceptualIR
- **Process**: Converts requirements to ER model

```python
class ConceptualDesigner(BaseAgent):
    """Designs conceptual ER model from RequirementIR."""
    
    name = "conceptual_designer"
    
    def run(self, board: Blackboard) -> Blackboard:
        if board.requirement_ir is None:
            logger.warning("ConceptualDesigner: RequirementIR not found, skipping")
            return board
        
        logger.info("ConceptualDesigner: Generating ConceptualIR")
        
        # Load prompts
        sys_tmpl = load_prompt("roles/conceptual_system.txt")
        usr_tmpl = load_prompt("roles/conceptual_user.txt")
        
        # Render user prompt with RequirementIR
        user_content = render_prompt(
            usr_tmpl,
            REQUIREMENT_JSON=board.requirement_ir.model_dump_json(indent=2),
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        raw = chat(messages)
        data = extract_json(raw)
        board.conceptual_ir = ConceptualIR.model_validate(data)
        
        logger.info(
            f"ConceptualDesigner: Generated ConceptualIR with "
            f"{len(board.conceptual_ir.entities)} entities and "
            f"{len(board.conceptual_ir.relationships)} relationships"
        )
        
        return board
```

**LogicalDesigner** (`agents/roles/logical_designer.py`)
- **Input**: ConceptualIR + RequirementIR
- **Output**: LogicalIR
- **Process**: Converts ER model to relational schema

```python
class LogicalDesigner(BaseAgent):
    """Designs logical relational schema from ConceptualIR."""
    
    name = "logical_designer"
    
    def run(self, board: Blackboard) -> Blackboard:
        if board.conceptual_ir is None:
            logger.warning("LogicalDesigner: ConceptualIR not found, skipping")
            return board
        
        logger.info("LogicalDesigner: Generating LogicalIR")
        
        # Load prompts
        sys_tmpl = load_prompt("roles/logical_system.txt")
        usr_tmpl = load_prompt("roles/logical_user.txt")
        
        # Render user prompt with both IRs
        conceptual_json = board.conceptual_ir.model_dump_json(indent=2)
        requirement_json = (
            board.requirement_ir.model_dump_json(indent=2)
            if board.requirement_ir
            else "null"
        )
        
        user_content = render_prompt(
            usr_tmpl,
            CONCEPTUAL_JSON=conceptual_json,
            REQUIREMENT_JSON=requirement_json,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        raw = chat(messages)
        data = extract_json(raw)
        board.logical_ir = LogicalIR.model_validate(data)
        
        logger.info(
            f"LogicalDesigner: Generated LogicalIR with "
            f"{len(board.logical_ir.tables)} tables"
        )
        
        return board
```

**DistributionEngineer** (`agents/roles/dist_engineer.py`)
- **Input**: LogicalIR + RequirementIR
- **Output**: GenerationIR
- **Process**: Designs distribution specifications for each column

```python
class DistributionEngineer(BaseAgent):
    """Designs generation specifications from RequirementIR and LogicalIR."""
    
    name = "dist_engineer"
    
    def run(self, board: Blackboard) -> Blackboard:
        if board.logical_ir is None:
            logger.warning("DistributionEngineer: LogicalIR not found, skipping")
            return board
        
        logger.info("DistributionEngineer: Generating GenerationIR")
        
        # Load prompts
        sys_tmpl = load_prompt("roles/dist_system.txt")
        usr_tmpl = load_prompt("roles/dist_user.txt")
        
        # Render user prompt with IR data
        logical_json = board.logical_ir.model_dump_json(indent=2)
        requirement_json = (
            board.requirement_ir.model_dump_json(indent=2)
            if board.requirement_ir
            else "null"
        )
        
        user_content = render_prompt(
            usr_tmpl,
            LOGICAL_JSON=logical_json,
            REQUIREMENT_JSON=requirement_json,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        raw = chat(messages)
        data = extract_json(raw)
        
        # Validate and store
        board.generation_ir = GenerationIR.model_validate(data)
        
        logger.info(
            f"DistributionEngineer: Generated GenerationIR with "
            f"{len(board.generation_ir.columns)} column specifications"
        )
        
        return board
```

**WorkloadDesigner** (`agents/roles/workload_designer.py`)
- **Input**: LogicalIR + RequirementIR
- **Output**: WorkloadIR
- **Process**: Designs query workload specifications

```python
class WorkloadDesigner(BaseAgent):
    """Designs workload specifications from LogicalIR and RequirementIR."""
    
    name = "workload_designer"
    
    def run(self, board: Blackboard) -> Blackboard:
        if board.logical_ir is None:
            logger.warning("WorkloadDesigner: LogicalIR not found, skipping")
            return board
        
        logger.info("WorkloadDesigner: Generating WorkloadIR")
        
        # Load prompts
        sys_tmpl = load_prompt("roles/workload_system.txt")
        usr_tmpl = load_prompt("roles/workload_user.txt")
        
        # Render user prompt
        logical_json = board.logical_ir.model_dump_json(indent=2)
        requirement_json = (
            board.requirement_ir.model_dump_json(indent=2)
            if board.requirement_ir
            else "null"
        )
        
        user_content = render_prompt(
            usr_tmpl,
            LOGICAL_JSON=logical_json,
            REQUIREMENT_JSON=requirement_json,
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        raw = chat(messages)
        data = extract_json(raw)
        
        # Handle case where LLM returns a list instead of dict with "targets" key
        if isinstance(data, list):
            logger.warning(
                "WorkloadDesigner: LLM returned a list, wrapping in {'targets': [...]}"
            )
            data = {"targets": data}
        
        # Validate and store
        board.workload_ir = WorkloadIR.model_validate(data)
        
        logger.info(
            f"WorkloadDesigner: Generated WorkloadIR with "
            f"{len(board.workload_ir.targets)} workload targets"
        )
        
        return board
```

**QACompilerAgent** (`agents/roles/qa_compiler.py`)
- **Input**: All previous IRs
- **Output**: DatasetIR
- **Process**: 
  1. Combines IRs into DatasetIR
  2. Validates logical schema (PK/FK)
  3. Validates generation specs

```python
class QACompilerAgent(BaseAgent):
    """Validates and compiles final DatasetIR from all IR components."""
    
    name = "qa_compiler"
    
    def run(self, board: Blackboard) -> Blackboard:
        if not (board.logical_ir and board.generation_ir):
            logger.warning(
                "QACompilerAgent: Missing required IR components "
                "(logical_ir or generation_ir), skipping"
            )
            return board
        
        logger.info("QACompilerAgent: Compiling DatasetIR")
        
        try:
            # Combine IRs into DatasetIR
            dataset = DatasetIR(
                logical=board.logical_ir,
                generation=board.generation_ir,
                workload=board.workload_ir,
                description=(
                    board.requirement_ir.narrative
                    if board.requirement_ir
                    else None
                ),
            )
            
            # Validate
            validate_logical(dataset)  # PK/FK checks
            validate_generation(dataset)  # Generation spec checks
            
            board.dataset_ir = dataset
            
            logger.info(
                f"QACompilerAgent: Successfully compiled DatasetIR with "
                f"{len(dataset.logical.tables)} tables"
            )
        
        except Exception as e:
            logger.error(f"QACompilerAgent: Validation failed: {e}", exc_info=True)
            raise
        
        return board
```

#### 2.4 Agent Runner with Repair Loop (`agents/runner.py`)

**Purpose**: Executes agents with automatic repair loops based on validation feedback.

**Repair Loop Flowchart**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Repair Loop Flow                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: Initial Production                                  │  │
│  │                                                           │  │
│  │  board = agent._produce(board)                           │  │
│  │    ├─ Load prompts                                       │  │
│  │    ├─ Call LLM                                           │  │
│  │    ├─ Extract JSON                                       │  │
│  │    └─ Validate & store IR in blackboard                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: Validation                                        │  │
│  │                                                           │  │
│  │  issues = collect_issues(validators, board)              │  │
│  │    ├─ Run validate_logical_ir(board)                     │  │
│  │    ├─ Run validate_generation_ir(board)                  │  │
│  │    └─ Collect all QaIssue objects                       │  │
│  │                                                           │  │
│  │  QaIssue:                                                 │  │
│  │    - stage: "LogicalIR" | "GenerationIR"               │  │
│  │    - code: "MISSING_PK" | "FK_REF_INVALID" | ...       │  │
│  │    - location: "table_name" | "table.column"            │  │
│  │    - message: "Human-readable error message"            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│                    ┌───────────────┐                           │
│                    │  Any issues?   │                           │
│                    └───────┬───────┘                           │
│                            │                                     │
│              ┌─────────────┴─────────────┐                     │
│              │                           │                     │
│             NO                          YES                     │
│              │                           │                     │
│              ▼                           ▼                     │
│  ┌──────────────────────┐   ┌──────────────────────────────┐  │
│  │ Return board          │   │ Step 3: Check Retries        │  │
│  │ (Validation passed)  │   │                              │  │
│  └──────────────────────┘   │  if attempt < max_retries:   │  │
│                              │    └─ Proceed to repair      │  │
│                              │  else:                        │  │
│                              │    └─ Raise RuntimeError      │  │
│                              └──────────────────────────────┘  │
│                                      │                          │
│                                      ▼                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 4: Repair                                            │  │
│  │                                                           │  │
│  │  board = agent._repair(board, issues)                    │  │
│  │                                                           │  │
│  │  Repair Process:                                         │  │
│  │    1. Build repair prompt:                              │  │
│  │       ├─ Format issues as JSON                          │  │
│  │       ├─ Include current IR state                      │  │
│  │       └─ Request fixes                                  │  │
│  │                                                           │  │
│  │    2. Call LLM with repair prompt:                      │  │
│  │       ├─ System: Original agent system prompt           │  │
│  │       └─ User: Repair prompt with issues                │  │
│  │                                                           │  │
│  │    3. Extract & validate fixed IR:                      │  │
│  │       ├─ Extract JSON from response                     │  │
│  │       └─ Validate as IR model                          │  │
│  │                                                           │  │
│  │    4. Update blackboard:                                 │  │
│  │       └─ Store fixed IR                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│                    ┌───────────────┐                           │
│                    │  attempt++    │                           │
│                    └───────┬───────┘                           │
│                            │                                     │
│                            ▼                                     │
│                    Loop back to Step 2                         │
│                    (Re-validate)                               │
└─────────────────────────────────────────────────────────────────┘

**Repair Prompt Example**:

```
The following issues were found during validation:

[
  {
    "stage": "LogicalIR",
    "code": "MISSING_PK",
    "location": "fact_sales",
    "message": "Table 'fact_sales' is missing a primary key"
  },
  {
    "stage": "LogicalIR",
    "code": "FK_REF_TABLE_MISSING",
    "location": "fact_sales.product_id",
    "message": "Foreign key 'product_id' references missing table 'dim_product'"
  }
]

Current IR state:

{
  "tables": {
    "fact_sales": {
      "name": "fact_sales",
      "columns": [...],
      "primary_key": [],
      "foreign_keys": [
        {"column": "product_id", "ref_table": "dim_product", ...}
      ]
    }
  }
}

Please fix these issues and return the corrected IR as JSON.
Focus on addressing each issue systematically.
```
```

**Key Function**:

```python
def run_with_repair(
    agent: BaseAgent,
    board: Blackboard,
    validators: List[Callable[[Blackboard], List[QaIssue]]],
    max_retries: int = 2,
) -> Blackboard:
    """
    Run agent with repair loop.
    
    The agent's _produce() method is called first. Then validators are run.
    If issues are found, _repair() is called up to max_retries times.
    
    Process:
    1. Call agent._produce(board) to generate initial IR
    2. Run validators to collect QaIssues
    3. If issues found and retries remaining:
       - Call agent._repair(board, issues)
       - Repeat validation
    4. If max_retries exceeded, raise RuntimeError
    """
    logger.info(f"Running {agent.name} with repair loop (max_retries={max_retries})")
    
    # Initial production
    b = agent._produce(board)
    
    # Repair loop
    for attempt in range(max_retries + 1):
        issues = collect_issues(validators, b)
        if not issues:
            logger.info(f"{agent.name}: Validation passed after {attempt} repair attempt(s)")
            return b
        
        if attempt < max_retries:
            b = agent._repair(b, issues)
        else:
            raise RuntimeError(f"Agent {agent.name} failed to repair after {max_retries} retries")
    
    return b
```

**Repair Prompt Builder**:

```python
def build_repair_prompt(qa_items: List[QaIssue], current_ir_json: str) -> str:
    """
    Build a repair prompt from QA issues.
    
    Formats the issues and current IR state into a prompt for LLM-based repair.
    """
    issues_json = json.dumps([asdict(i) for i in qa_items], indent=2)
    
    prompt = f"""The following issues were found during validation:

{issues_json}

Current IR state:

{current_ir_json}

Please fix these issues and return the corrected IR as JSON.
Focus on addressing each issue systematically.
"""
    return prompt
```

**Usage Example**:

```python
from nl2data.agents.runner import run_with_repair
from nl2data.ir.validators import validate_logical_ir

# Define validators
validators = [validate_logical_ir]

# Run agent with repair loop
board = run_with_repair(
    agent=LogicalDesigner(),
    board=board,
    validators=validators,
    max_retries=2
)
```

**Base Agent Interface**:

Agents that support repair should implement:

```python
class BaseAgent:
    def _produce(self, board: Blackboard) -> Blackboard:
        """Generate initial IR. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _repair(self, board: Blackboard, issues: List[QaIssue]) -> Blackboard:
        """
        Repair IR based on validation issues.
        
        Default implementation uses LLM-based repair with build_repair_prompt().
        Subclasses can override for custom repair logic.
        """
        # Default: LLM-based repair
        prompt = build_repair_prompt(issues, current_ir_json)
        # ... call LLM and update board
        return board
    
    def run(self, board: Blackboard) -> Blackboard:
        """Legacy interface - calls _produce() for backward compatibility."""
        return self._produce(board)
```

#### 2.5 Orchestrator (`agents/orchestrator.py`)

**Purpose**: Executes agents in sequence (with optional repair loops).

```python
class Orchestrator:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
    
    def execute(self, board: Blackboard) -> Blackboard:
        current_board = board
        for agent in self.agents:
            current_board = agent.run(current_board)
        return current_board
    
    def execute_with_repair(
        self, 
        board: Blackboard,
        validators_map: Dict[str, List[Callable]]
    ) -> Blackboard:
        """
        Execute agents with repair loops.
        
        Args:
            board: Initial blackboard
            validators_map: Dict mapping agent name -> list of validators
        """
        current_board = board
        for agent in self.agents:
            validators = validators_map.get(agent.name, [])
            if validators:
                current_board = run_with_repair(
                    agent, current_board, validators, max_retries=2
                )
            else:
                current_board = agent.run(current_board)
        return current_board
```

---

### 3. Data Generation Engine

#### 3.1 Generation Pipeline (`generation/engine/pipeline.py`)

**Main Entry Point**:

```python
def generate_from_ir(
    ir: DatasetIR, 
    out_dir: Path, 
    seed: int, 
    chunk_rows: int
) -> None:
    """
    Generate data from DatasetIR.
    
    Process:
    1. Build derived registry (compile expressions, compute dependencies)
    2. Separate dimension and fact tables
    3. Generate dimension tables (in-memory)
    4. Generate fact tables (streaming/chunked)
    """
```

**Generation Pipeline Flowchart**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Generation Pipeline                       │
│                                                                  │
│  Input: DatasetIR                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - logical: LogicalIR (tables, columns, PK/FK)            │  │
│  │ - generation: GenerationIR (distributions)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: Build Derived Registry                            │  │
│  │                                                           │  │
│  │  For each derived column:                                │  │
│  │    ├─ Compile expression to AST                          │  │
│  │    ├─ Extract dependencies (column names)                │  │
│  │    └─ Store compiled program                             │  │
│  │                                                           │  │
│  │  Per table:                                              │  │
│  │    └─ Topological sort columns by dependencies          │  │
│  │                                                           │  │
│  │  Output: DerivedRegistry                                 │  │
│  │    - programs: {(table, col) → DerivedProgram}          │  │
│  │    - order: {table → [cols in dependency order]}        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: Separate Tables                                   │  │
│  │                                                           │  │
│  │  dimensions = {name: table for table.kind == "dimension"}│  │
│  │  facts = {name: table for table.kind == "fact"}          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 3: Generate Dimension Tables (In-Memory)            │  │
│  │                                                           │  │
│  │  For each dimension table:                               │  │
│  │    ┌─────────────────────────────────────────────────┐  │  │
│  │    │ Phase 1: Generate Base Columns                  │  │  │
│  │    │   For each non-derived column:                  │  │  │
│  │    │     ├─ Get distribution spec                    │  │  │
│  │    │     ├─ Create sampler (Uniform, Zipf, etc.)     │  │  │
│  │    │     └─ Sample n values                          │  │  │
│  │    └─────────────────────────────────────────────────┘  │  │
│  │                            │                             │  │
│  │                            ▼                             │  │
│  │    ┌─────────────────────────────────────────────────┐  │  │
│  │    │ Phase 2: Compute Derived Columns                │  │  │
│  │    │   For each derived column (in dependency order):│  │  │
│  │    │     ├─ Get compiled program                     │  │  │
│  │    │     ├─ Evaluate expression on DataFrame         │  │  │
│  │    │     └─ Add column to DataFrame                 │  │  │
│  │    └─────────────────────────────────────────────────┘  │  │
│  │                            │                             │  │
│  │                            ▼                             │  │
│  │    ┌─────────────────────────────────────────────────┐  │  │
│  │    │ Phase 3: Enforce Constraints                   │  │  │
│  │    │   ├─ Functional dependencies (intra-row)       │  │  │
│  │    │   ├─ Implications (if-then rules)             │  │  │
│  │    │   └─ Nullability constraints                    │  │  │
│  │    └─────────────────────────────────────────────────┘  │  │
│  │                            │                             │  │
│  │                            ▼                             │  │
│  │    Write CSV file                                        │  │
│  │    Store DataFrame in dim_dfs (for FK references)      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 4: Generate Fact Tables (Streaming)                 │  │
│  │                                                           │  │
│  │  For each fact table:                                    │  │
│  │    ┌─────────────────────────────────────────────────┐  │  │
│  │    │ While produced < total_rows:                     │  │  │
│  │    │   chunk_size = min(chunk_rows, remaining)       │  │  │
│  │    │                                                  │  │  │
│  │    │   ┌──────────────────────────────────────────┐ │  │  │
│  │    │   │ Phase 1: Generate Base Columns (Chunk)   │ │  │  │
│  │    │   │   For each non-derived column:           │ │  │  │
│  │    │   │     ├─ If FK: sample from dim_dfs        │ │  │  │
│  │    │   │     │   (with Zipf skew if specified)   │ │  │  │
│  │    │   │     └─ Else: sample from distribution    │ │  │  │
│  │    │   └──────────────────────────────────────────┘ │  │  │
│  │    │                            │                   │  │  │
│  │    │                            ▼                   │  │  │
│  │    │   ┌──────────────────────────────────────────┐ │  │  │
│  │    │   │ Phase 2: Compute Derived Columns        │ │  │  │
│  │    │   │   (same as dimension tables)            │ │  │  │
│  │    │   └──────────────────────────────────────────┘ │  │  │
│  │    │                            │                   │  │  │
│  │    │                            ▼                   │  │  │
│  │    │   ┌──────────────────────────────────────────┐ │  │  │
│  │    │   │ Phase 3: Enforce Constraints            │ │  │  │
│  │    │   │   (same as dimension tables)             │ │  │  │
│  │    │   └──────────────────────────────────────────┘ │  │  │
│  │    │                            │                   │  │  │
│  │    │                            ▼                   │  │  │
│  │    │   Yield DataFrame chunk                        │  │  │
│  │    │   Write chunk to CSV (append mode)            │  │  │
│  │    │   produced += chunk_size                      │  │  │
│  │    └─────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Output: CSV Files in out_dir/                            │  │
│  │   - dim_table1.csv                                       │  │
│  │   - dim_table2.csv                                       │  │
│  │   - fact_table1.csv (streamed, large)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Steps**:

1. **Derived Registry Building**:
   ```python
   derived_reg = build_derived_registry(ir)
   ```
   - Compiles all derived expressions
   - Extracts column dependencies
   - Computes topological ordering per table

2. **Table Separation**:
   ```python
   dims = {name: t for name, t in tables.items() if t.kind == "dimension"}
   facts = {name: t for name, t in tables.items() if t.kind == "fact"}
   ```

3. **Dimension Generation** (with error handling):
   ```python
   for name, table_spec in dims.items():
       try:
           df = generate_dimension(table_spec, ir, rng, derived_reg)
           write_csv(df, output_path)
           dim_dfs[name] = df  # Store for FK references
       except Exception as e:
           logger.error(f"Failed to generate dimension table '{name}': {e}")
           # Continue with remaining tables instead of stopping
   ```
   - **Error Resilience**: If one dimension table fails, generation continues with remaining tables
   - **Failure Tracking**: Failed tables are logged and reported at the end

4. **Fact Generation** (Streaming, with error handling):
   ```python
   for name, table_spec in facts.items():
       try:
           stream = generate_fact_stream(
               table_spec, ir, dim_dfs, rng, chunk_rows, derived_reg
           )
           write_csv_stream(stream, output_path)
       except Exception as e:
           logger.error(f"Failed to generate fact table '{name}': {e}")
           # Continue with remaining tables instead of stopping
   ```
   - **Error Resilience**: If one fact table fails, generation continues with remaining tables
   - **Partial Success**: Pipeline completes successfully even if some tables fail

#### 3.2 Dimension Generator (`generation/engine/dim_generator.py`)

**Two-Phase Generation**:

```python
def generate_dimension(
    table: TableSpec,
    ir: DatasetIR,
    rng: np.random.Generator,
    derived_reg: DerivedRegistry,
) -> pd.DataFrame:
    """
    Phase 1: Generate base columns (non-derived)
    Phase 2: Compute derived columns in dependency order
    """
    # Phase 1: Base columns
    base_data = {}
    for col in table.columns:
        dist = gen_map.get((table.name, col.name))
        if isinstance(dist, DistDerived):
            continue  # Skip derived columns
        base_data[col.name] = sample_column(table, col, dist, n, rng)
    
    df = pd.DataFrame(base_data)
    
    # Phase 2: Derived columns
    derived_cols = derived_reg.order.get(table.name, [])
    for col_name in derived_cols:
        prog = derived_reg.programs[(table.name, col_name)]
        df[col_name] = eval_derived(prog, df)
    
    return df
```

**Column Sampling**:

```python
def sample_column(
    table: TableSpec,
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample values for a column.
    
    If dist is None, uses fallback based on SQL type.
    Otherwise, uses the specified distribution sampler.
    """
    if dist is None:
        # Fallback by SQL type
        if col.sql_type in ("INT32", "INT64"):
            return rng.integers(1, 1_000_000, size=n, dtype=np.int64)
        if col.sql_type in ("FLOAT32", "FLOAT64"):
            return rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
        if col.sql_type == "TEXT":
            return np.array([f"{col.name}_{i}" for i in range(n)])
        if col.sql_type == "BOOL":
            return rng.choice([True, False], size=n)
        if col.sql_type in ("DATE", "DATETIME"):
            base = np.datetime64("2020-01-01")
            days = rng.integers(0, 365 * 3, size=n)
            return base + days.astype("timedelta64[D]")
        # Default fallback
        logger.warning(
            f"No fallback for type {col.sql_type}, using text fallback"
        )
        return np.array([f"{col.name}_{i}" for i in range(n)])
    
    # Use specified distribution
    sampler = get_sampler(dist, rng=rng)
    return sampler.sample(n, rng=rng)
```

#### 3.3 Fact Generator (`generation/engine/fact_generator.py`)

**Streaming Generation**:

```python
def generate_fact_stream(
    table: TableSpec,
    ir: DatasetIR,
    dims: Dict[str, pd.DataFrame],
    rng: np.random.Generator,
    chunk_rows: int,
    derived_reg: DerivedRegistry,
) -> Iterator[pd.DataFrame]:
    """
    Generates fact table in chunks (streaming).
    
    Yields DataFrame chunks to avoid memory issues with large tables.
    """
    n_total = table.row_count or 1_000_000
    
    while produced < n_total:
        m = min(chunk_rows, n_total - produced)
        
        # Phase 1: Base columns
        base_block = {}
        for col in table.columns:
            if isinstance(dist, DistDerived):
                continue
            base_block[col.name] = sample_fact_column(
                table, col, dist, m, rng, dims
            )
        
        df_chunk = pd.DataFrame(base_block)
        
        # Phase 2: Derived columns
        for col_name in derived_cols:
            prog = derived_reg.programs[(table.name, col_name)]
            df_chunk[col_name] = eval_derived(prog, df_chunk)
        
        yield df_chunk
        produced += m
```

**Foreign Key Handling**:

```python
def sample_fact_column(
    table: TableSpec,
    col: ColumnSpec,
    dist,
    n: int,
    rng: np.random.Generator,
    dims: Dict[str, pd.DataFrame],
) -> np.ndarray:
    """
    Handles foreign keys specially to maintain referential integrity.
    """
    if col.role == "foreign_key" and col.references:
        ref_table, ref_col = col.references.split(".")
        support = dims[ref_table][ref_col].to_numpy(copy=False)
        
        if isinstance(dist, DistZipf):
            # Use Zipf distribution over FK values
            sampler = get_sampler(dist, rng=rng, support=support, n_items=len(support))
            return sampler.sample(n, rng=rng, support=support)
        else:
            # Uniform sampling from FK values
            idx = rng.integers(0, len(support), size=n)
            return support[idx]
    
    # Non-FK: reuse dimension sampling logic
    return sample_column(table, col, dist, n, rng)
```

#### 3.4 Derived Column System

The system supports derived columns that are computed from other columns using a Domain-Specific Language (DSL). Derived columns enable complex calculations, temporal extractions, and conditional logic without requiring pre-computed values.

**Key Features**:
- **Dependency Tracking**: Automatic detection and topological sorting of column dependencies
- **Vectorized Evaluation**: Efficient pandas-based computation on entire DataFrames
- **Type Safety**: Optional dtype hints for expression result types
- **Dimension Lookups**: Support for referencing columns from joined dimension tables
- **Comprehensive Validation**: Compile-time and runtime checks for expression validity

**DSL Functions Reference**:

**Date/Time Extraction**:
- `hour(datetime_col)`: Extract hour (0-23) as integer
- `date(datetime_col)`: Extract date part (returns Timestamp with time at midnight)
- `day_of_week(datetime_col)`: Extract day of week (0=Monday, 6=Sunday) as integer
- `day_of_month(datetime_col)`: Extract day of month (1-31) as integer
- `month(datetime_col)`: Extract month (1-12) as integer
- `year(datetime_col)`: Extract year as integer

**Arithmetic Operations**:
- `+`, `-`, `*`, `/`, `//` (floor division), `%` (modulo), `**` (exponentiation)

**Boolean Operations**:
- `and`, `or`, `not`

**Comparison Operators**:
- `<`, `<=`, `>`, `>=`, `==`, `!=`

**Conditional Logic**:
- `where(condition, value_if_true, value_if_false)`: Vectorized conditional
- `value_if_true if condition else value_if_false`: Ternary expression

**Math Functions**:
- `abs(x)`: Absolute value
- `sqrt(x)`: Square root
- `log(x)`: Natural logarithm
- `exp(x)`: Exponential
- `clip(x, lower, upper)`: Clip value to range

**Time Arithmetic** (for intervals):
- `seconds(n)`: Convert to timedelta in seconds
- `minutes(n)`: Convert to timedelta in minutes
- `hours(n)`: Convert to timedelta in hours
- `days(n)`: Convert to timedelta in days

**Expression Examples**:

1. **Simple Arithmetic**:
   ```python
   "unit_price * quantity"  # Calculate line subtotal
   ```

2. **Date Extraction**:
   ```python
   "date(timestamp)"  # Extract date part
   "hour(timestamp)"  # Extract hour of day
   ```

3. **Conditional Boolean**:
   ```python
   "where((hour(timestamp) >= 7 and hour(timestamp) <= 9) or (hour(timestamp) >= 16 and hour(timestamp) <= 18), 1, 0)"  # Peak hour flag
   ```

4. **Weekend Detection**:
   ```python
   "where(day_of_week(timestamp) >= 5, 1, 0)"  # Weekend flag (Saturday=5, Sunday=6)
   ```

5. **Chained Arithmetic**:
   ```python
   "gross_fare - discount_amount + tax_amount"  # Net fare calculation
   ```

6. **Conditional with Threshold**:
   ```python
   "where(consumption_kwh > threshold, cost_before_rebate * rebate_rate, 0)"  # Rebate amount
   ```

7. **Dimension Lookup** (after join):
   ```python
   "dynamic_price_per_kwh / baseline_price_per_kwh"  # Price multiplier (baseline from dimension)
   ```

8. **Time Arithmetic**:
   ```python
   "start_time + minutes(duration_minutes)"  # End time calculation
   ```

**Dimension Lookup Mechanism**:

Derived expressions can reference columns from dimension tables through automatic joins. The fact generator performs left joins on foreign keys before computing derived columns, making dimension attributes available in the expression environment.

Example:
- Fact table has `tariff_plan_id` foreign key
- Dimension table `dim_tariff_plan` has `baseline_price_per_kwh` column
- Derived expression: `"dynamic_price_per_kwh / baseline_price_per_kwh"`
- The generator automatically joins `dim_tariff_plan` before evaluating the expression

**Validation**:

The system validates derived columns at multiple stages:

1. **Compile-time**: Expression syntax and allowed functions are checked
2. **IR Validation**: Dependencies are verified to exist in the schema
3. **Runtime**: Missing columns or evaluation errors provide helpful error messages

**Implementation Details**:

**Dependency Tracking** (`generation/derived_registry.py`):

```python
class DerivedRegistry:
    """Registry of compiled derived expressions."""
    programs: Dict[DerivedKey, DerivedProgram]  # (table, col) → compiled program
    order: Dict[str, List[str]]                 # table → [cols in topological order]

def build_derived_registry(ir: DatasetIR) -> DerivedRegistry:
    """
    1. Collect derived specs and compile them
    2. Extract dependencies from AST
    3. Topologically sort columns per table
    """
    reg = DerivedRegistry()
    per_table_deps: Dict[str, Dict[str, Set[str]]] = {}
    
    for cg in ir.generation.columns:
        if isinstance(cg.distribution, DistDerived):
            prog = compile_derived(dist.expression, dist.dtype)
            reg.programs[(table, col)] = prog
            dist.depends_on = list(prog.dependencies)
            per_table_deps.setdefault(table, {})[col] = prog.dependencies
    
    # Topological sort
    for table, dep_map in per_table_deps.items():
        reg.order[table] = topo_sort_columns(dep_map)
    
    return reg
```

**Expression Compilation** (`generation/derived_program.py`):

```python
def compile_derived(expr: str, dtype: Optional[str] = None) -> DerivedProgram:
    """
    Compile a derived expression string into a DerivedProgram.
    
    Process:
    1. Normalize expression (convert SQL INTERVAL syntax)
    2. Parse to AST
    3. Validate AST (whitelist of allowed nodes)
    4. Extract dependencies (column names)
    """
    # Normalize expression
    src = _rewrite_expr(expr)
    
    # Parse to AST
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {expr}") from e
    
    # Track dependencies (column names)
    deps: Set[str] = set()
    
    def visit(node: ast.AST) -> None:
        """Recursively visit AST nodes to extract dependencies and validate."""
        if isinstance(node, ast.BinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp):
            visit(node.operand)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                visit(v)
        elif isinstance(node, ast.Compare):
            visit(node.left)
            for comp in node.comparators:
                visit(comp)
        elif isinstance(node, ast.IfExp):
            visit(node.test)
            visit(node.body)
            visit(node.orelse)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"Only simple function names allowed, not {type(node.func).__name__}"
                )
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCS:
                raise ValueError(
                    f"Function '{func_name}' not allowed. "
                    f"Allowed functions: {sorted(ALLOWED_FUNCS)}"
                )
            for arg in node.args:
                visit(arg)
            if node.keywords:
                raise ValueError("Keyword arguments not supported in expressions")
        elif isinstance(node, ast.Name):
            # This is a column reference
            deps.add(node.id)
        elif isinstance(node, ast.Constant):
            # Literal value (int, float, str, bool, None)
            pass
        elif isinstance(node, (ast.Num, ast.Str)):  # Python < 3.8 compatibility
            # Legacy constant nodes
            pass
        else:
            raise ValueError(
                f"Node type {type(node).__name__} not allowed in expression: {expr}"
            )
    
    # Visit the AST
    visit(tree)
    
    # Remove function names from dependencies (they're not columns)
    deps = {d for d in deps if d not in ALLOWED_FUNCS}
    
    return DerivedProgram(
        expr=src,
        ast_root=tree.body,  # Expression.body is the actual expression node
        dependencies=deps,
        dtype=dtype,
    )
```

**Allowed Functions**:
- Math: `abs`, `log`, `exp`, `sqrt`, `clip`, `where`
- Time intervals: `seconds`, `minutes`, `hours`, `days` (for datetime arithmetic)
- Date/time extraction: `hour`, `date`, `day_of_week`, `day_of_month`, `month`, `year`

**Expression Evaluation** (`generation/derived_eval.py`):

The system uses **AST-based evaluation** for safety and performance:

```python
def eval_derived(prog: DerivedProgram, df: pd.DataFrame) -> pd.Series:
    """
    Evaluates derived expression using AST traversal.
    
    Process:
    1. Build evaluation environment (columns + functions)
    2. Recursively evaluate AST nodes
    3. Ensure result is a Series with correct length
    """
    env = build_env(df)  # Column refs + allowed functions
    result = eval_node(prog.ast_root, env)
    
    # Ensure result is Series with correct length
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)
    
    return result

def build_env(df: pd.DataFrame) -> Dict[str, Any]:
    """Build evaluation environment with column references and functions."""
    env = {}
    
    # Column references
    for col in df.columns:
        env[col] = df[col]
    
    # Vectorized math functions
    env["log"] = np.log
    env["exp"] = np.exp
    env["sqrt"] = np.sqrt
    env["abs"] = np.abs
    env["where"] = np.where
    
    # Time helpers (convert numeric to timedelta)
    env["seconds"] = lambda x: pd.to_timedelta(x, unit="s")
    env["minutes"] = lambda x: pd.to_timedelta(x, unit="m")
    env["hours"] = lambda x: pd.to_timedelta(x, unit="h")
    env["days"] = lambda x: pd.to_timedelta(x, unit="D")
    
    # Clip function (handles 1 or 2 args)
    env["clip"] = clip_func
    
    return env

def eval_node(node: ast.AST, env: Dict[str, Any]) -> Any:
    """
    Recursively evaluate AST node.
    
    Supports:
    - Binary operations (+, -, *, /, //, %, **)
    - Unary operations (+, -, ~)
    - Boolean operations (and, or)
    - Comparisons (<, <=, >, >=, ==, !=)
    - Conditional expressions (x if cond else y)
    - Function calls (only whitelisted functions)
    - Column references (ast.Name)
    - Constants (numbers, strings)
    """
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, env)
        right = eval_node(node.right, env)
        op = node.op
        # Handle +, -, *, /, //, %, **
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            return left / right
        elif isinstance(op, ast.FloorDiv):
            return left // right
        elif isinstance(op, ast.Mod):
            return left % right
        elif isinstance(op, ast.Pow):
            return left ** right
        else:
            raise ValueError(f"Unexpected BinOp {type(op)}")
    
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        func = env[func_name]
        args = [eval_node(arg, env) for arg in node.args]
        return func(*args)
    
    elif isinstance(node, ast.Name):
        return env[node.id]  # Column reference
    
    # ... handle other node types
```

**Key Features**:
- **AST-based**: More secure than `eval()` - only whitelisted operations
- **Vectorized**: Operations work on entire Series/arrays
- **Type-safe**: Validates node types during compilation
- **Error handling**: Clear error messages for unsupported operations

#### 3.5 Constraint Enforcement (`generation/enforce.py`)

**Purpose**: Enforces advanced constraints (FDs, implications, nullability) during data generation.

**Key Functions**:

1. **`eval_condition(df: pd.DataFrame, expr: ConditionExpr) -> pd.Series`**:
   Evaluates structured condition expressions on DataFrames.

```python
def eval_condition(df: pd.DataFrame, expr: ConditionExpr) -> pd.Series:
    """
    Evaluate structured condition expression on DataFrame.
    
    Supports:
    - Atomic conditions: eq, ne, lt, le, gt, ge, in, is_null, not_null
    - Boolean operators: and, or, not
    - Tree-based expressions
    """
    if expr.kind == "atom":
        c = expr.atom
        if c.op == "eq":
            return df[c.col] == c.value
        elif c.op == "in":
            return df[c.col].isin(c.value)
        # ... other operators
    elif expr.kind == "and":
        mask = pd.Series(True, index=df.index)
        for child in expr.children:
            mask &= eval_condition(df, child)
        return mask
    # ... other operators
```

2. **`enforce_intra_fd(df: pd.DataFrame, fd: FDConstraint) -> pd.DataFrame`**:
   Enforces intra-row functional dependencies.

```python
def enforce_intra_fd(df: pd.DataFrame, fd: FDConstraint) -> pd.DataFrame:
    """
    Enforce intra-row functional dependency.
    
    Groups by LHS columns and ensures RHS values are unique per group.
    If violations exist, picks first value and overwrites others.
    """
    key = fd.lhs
    val = fd.rhs[0]
    
    # Create mapping from lhs -> canonical rhs
    canon_df = df[key + fd.rhs].drop_duplicates(subset=key, keep="first")
    canon = canon_df.set_index(key)[val]
    
    # Map and overwrite
    df[val] = df[key[0]].map(canon)
    return df
```

3. **`enforce_implication(df: pd.DataFrame, constraint: ImplicationConstraint) -> pd.DataFrame`**:
   Enforces implication constraints (if condition, then effect).

```python
def enforce_implication(df: pd.DataFrame, constraint: ImplicationConstraint) -> pd.DataFrame:
    """
    Enforce implication: if condition, then effect.
    
    When condition matches, enforce effect (set to NULL, set to value, etc.).
    """
    mask_if = eval_condition(df, constraint.condition)
    
    if constraint.effect.kind == "atom" and constraint.effect.atom:
        atom = constraint.effect.atom
        col = atom.col
        
        if atom.op == "is_null":
            df.loc[mask_if, col] = None
        elif atom.op == "eq":
            df.loc[mask_if, col] = atom.value
    
    return df
```

4. **`enforce_nullability(df: pd.DataFrame, table_spec) -> pd.DataFrame`**:
   Enforces nullability constraints from table specification.

5. **`enforce_batch(df: pd.DataFrame, constraints: ConstraintSpec, table_spec=None) -> pd.DataFrame`**:
   Applies all constraints to a batch of rows.

**Usage in Generation Pipeline**:
```python
from nl2data.generation.enforce import enforce_batch

# After generating base columns
df = generate_base_columns(...)

# Enforce constraints
df = enforce_batch(df, ir.logical.constraints, table_spec)
```

#### 3.6 Foreign Key Allocation (`generation/allocator.py`)

**Purpose**: Memory-safe FK allocation with guaranteed coverage and Zipf skew.

**Key Functions**:

1. **`zipf_probs(K: int, alpha: float) -> np.ndarray`**:
   Computes normalized Zipf probabilities.

```python
def zipf_probs(K: int, alpha: float) -> np.ndarray:
    """
    Compute normalized Zipf probabilities for K items.
    
    Args:
        K: Number of items
        alpha: Zipf exponent (higher = more skew)
    
    Returns:
        Normalized probability array of length K
    """
    ranks = np.arange(1, K + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    probs = weights / weights.sum()
    return probs
```

2. **`fk_assignments(pk_ids, n_rows, probs, rng, batch=5_000_000) -> Iterator[Tuple[np.ndarray, int]]`**:
   Generates FK assignments with guaranteed coverage and target skew.

```python
def fk_assignments(
    pk_ids: np.ndarray,
    n_rows: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    batch: int = 5_000_000,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generate FK assignments with guaranteed coverage and target skew.
    
    Guarantees that each PK gets at least one child (coverage), then allocates
    remaining rows by Zipf probabilities. Streams (pk_id, count) pairs without
    materializing a giant fk_pool.
    
    Yields:
        Tuples of (pk_id, count) where count is the number of fact rows for this PK
    """
    K = len(pk_ids)
    
    # Guarantee coverage: each PK gets at least 1
    base = np.ones(K, dtype=np.int64)
    leftover = n_rows - K
    
    # Allocate remaining rows by Zipf probabilities
    if leftover > 0:
        alloc = rng.multinomial(leftover, probs, size=1).ravel()
    else:
        alloc = np.zeros(K, dtype=np.int64)
    
    counts = base + alloc
    
    # Stream out (pk_id, count) pairs
    for i in range(K):
        c = counts[i]
        if c > 0:
            yield pk_ids[i], int(c)
```

3. **`generate_fk_array(...) -> np.ndarray`**:
   Convenience function that materializes the full FK array (use with caution for large datasets).

**Key Features**:
- **Guaranteed Coverage**: Every parent PK gets at least one child
- **Zipf Skew**: Remaining rows allocated by Zipf distribution
- **Memory Efficient**: Streams assignments without materializing full array
- **Batch Processing**: Handles very large dimension tables efficiently

#### 3.7 Value Providers (`generation/providers/`)

**Purpose**: Realistic data providers for generating human-readable values (names, addresses, etc.).

**Provider Types**:

1. **Faker Provider** (`faker_provider.py`):
   Uses Faker library for realistic data.

```python
from nl2data.generation.providers import get_provider

provider = get_provider("faker.email")
values = provider.generate(n=1000)
```

2. **Mimesis Provider** (`mimesis_provider.py`):
   Uses Mimesis library for realistic data.

3. **Geo Lookup Provider** (`lookup_geo.py`):
   Uses GeoNames/Natural Earth datasets for geographic data.

**Provider Registry** (`registry.py`):

```python
from nl2data.generation.providers import get_provider, list_providers

# List available providers
providers = list_providers()
# ['faker.email', 'faker.name', 'mimesis.full_name', 'lookup.city', ...]

# Get a provider
provider = get_provider("faker.email", config={"locale": "en_US"})
values = provider.generate(n=1000)
```

**Available Providers**:
- `faker.name`, `faker.email`, `faker.phone_number`, `faker.address`, `faker.city`, `faker.country`, `faker.company`, `faker.job`, `faker.date`, `faker.date_time`
- `mimesis.full_name`, `mimesis.email`, `mimesis.telephone`, `mimesis.address`
- `lookup.city`, `lookup.country`

**Custom Providers**:
```python
from nl2data.generation.providers import register_provider
from nl2data.generation.providers.base import ValueProvider

class MyProvider(ValueProvider):
    def generate(self, n: int) -> np.ndarray:
        # Custom generation logic
        pass

register_provider("my.provider", lambda cfg: MyProvider(**cfg))
```

#### 3.8 Fact Generation Utilities (`generation/facts.py`)

**Purpose**: Utilities for generating fact tables with composite PKs and parent-child relationships.

**Key Functions**:

1. **`spawn_children(parent_ids, child_count_dist, parent_fk_name, child_seq_name="line_no", rng=None) -> pd.DataFrame`**:
   Generates child rows with composite PK (parent_fk, sequence).

```python
def spawn_children(
    parent_ids: pd.Series,
    child_count_dist: Callable[[int], np.ndarray],
    parent_fk_name: str,
    child_seq_name: str = "line_no",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate child rows with composite PK (parent_fk, sequence).
    
    Useful for generating order_items from orders, where each order
    has multiple line items with a composite PK (order_id, line_no).
    
    Example:
        orders = pd.DataFrame({"order_id": [1, 2, 3]})
        order_items = spawn_children(
            orders["order_id"],
            lambda n: poisson_child_counts(n, mean=3.0),
            parent_fk_name="order_id",
            child_seq_name="line_no"
        )
        # Result: DataFrame with columns [order_id, line_no]
    """
```

2. **`poisson_child_counts(n_parents, mean=3.0, min_count=1, max_count=None, rng=None) -> np.ndarray`**:
   Generates child counts using Poisson distribution.

3. **`uniform_child_counts(n_parents, min_count=1, max_count=10, rng=None) -> np.ndarray`**:
   Generates child counts using uniform distribution.

#### 3.9 Distribution Samplers

**Distribution Sampler System Architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│              Distribution Sampler System                         │
│                                                                  │
│  GenerationIR Column Spec                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ {                                                         │  │
│  │   "table": "fact_sales",                                 │  │
│  │   "column": "product_id",                                │  │
│  │   "distribution": {                                      │  │
│  │     "kind": "zipf",                                      │  │
│  │     "s": 1.2,                                            │  │
│  │     "n": 10000                                           │  │
│  │   }                                                       │  │
│  │ }                                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Factory: get_sampler(distribution, **context)            │  │
│  │                                                           │  │
│  │  if isinstance(dist, DistUniform):                       │  │
│  │    return UniformSampler(dist.low, dist.high)           │  │
│  │  elif isinstance(dist, DistNormal):                      │  │
│  │    return NormalSampler(dist.mean, dist.std)             │  │
│  │  elif isinstance(dist, DistZipf):                       │  │
│  │    return ZipfSampler(dist.s, dist.n or ctx["n_items"])│  │
│  │  elif isinstance(dist, DistSeasonal):                    │  │
│  │    return SeasonalDateSampler(dist.weights, ...)         │  │
│  │  elif isinstance(dist, DistCategorical):                 │  │
│  │    return CategoricalSampler(dist.domain.values, ...)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ BaseSampler Interface                                    │  │
│  │                                                           │  │
│  │  class BaseSampler(ABC):                                 │  │
│  │    @abstractmethod                                       │  │
│  │    def sample(self, n: int, **kwargs) -> np.ndarray:   │  │
│  │      """                                                 │  │
│  │      Generate n samples from the distribution.          │  │
│  │      Args:                                               │  │
│  │        n: Number of samples                             │  │
│  │        **kwargs: Context (rng, support, n_items, etc.) │  │
│  │      """                                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Concrete Samplers                                        │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ UniformSampler                                     │  │  │
│  │  │   sample(n, rng) → uniform(low, high, size=n)     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ NormalSampler                                       │  │  │
│  │  │   sample(n, rng) → normal(mean, std, size=n)      │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ ZipfSampler                                         │  │  │
│  │  │   sample(n, rng, support, n_items) →               │  │  │
│  │  │     - If support: sample from FK values with Zipf   │  │  │
│  │  │     - Else: generate integer IDs 1..n_items        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ SeasonalDateSampler                               │  │  │
│  │  │   sample(n, rng, year_range) →                    │  │  │
│  │  │     - Sample periods by weights                   │  │  │
│  │  │     - Convert to dates                           │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ CategoricalSampler                                │  │  │
│  │  │   sample(n, rng) →                                │  │  │
│  │  │     - Sample from values with probabilities       │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  Generated Column Values (np.ndarray)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ array([1234, 5678, 1234, 9012, ...])                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

**Context Parameters**:
- `rng`: Random number generator (np.random.Generator)
- `support`: Array of values to sample from (for FK sampling with Zipf)
- `n_items`: Number of distinct items (for Zipf without support)
- `year_range`: Tuple of (start_year, end_year) for seasonal dates
```

**Factory Pattern** (`generation/distributions/factory.py`):

```python
def get_sampler(dist: Distribution, **ctx: Any) -> Any:
    """Creates appropriate sampler for distribution type."""
    if isinstance(dist, DistUniform):
        return UniformSampler(dist.low, dist.high)
    elif isinstance(dist, DistNormal):
        return NormalSampler(dist.mean, dist.std)
    elif isinstance(dist, DistZipf):
        return ZipfSampler(dist.s, dist.n or ctx.get("n_items"))
    elif isinstance(dist, DistSeasonal):
        return SeasonalDateSampler(dist.weights, dist.granularity)
    elif isinstance(dist, DistCategorical):
        return CategoricalSampler(dist.domain.values, dist.domain.probs)
```

**Base Sampler Interface** (`generation/distributions/base.py`):

```python
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseSampler(ABC):
    """
    Base class for all distribution samplers.
    
    All samplers must implement the sample() method which generates
    n values using the provided random number generator.
    """
    
    @abstractmethod
    def sample(self, n: int, **kwargs: Any) -> np.ndarray:
        """
        Generate n samples from the distribution.
        
        Args:
            n: Number of samples to generate
            **kwargs: Additional parameters specific to sampler
                - rng: Random number generator (np.random.Generator)
                - support: Array of values to sample from (for FK sampling)
                - n_items: Number of distinct items (for Zipf)
        
        Returns:
            Array of samples
        """
        pass
```

**Key Design**:
- All samplers inherit from `BaseSampler`
- `sample()` method signature is consistent across all samplers
- Context parameters passed via `**kwargs` (e.g., `rng`, `support` for FK sampling)
- Random number generator extracted from kwargs: `rng = kwargs.get("rng", np.random.default_rng())`

**Example: Uniform Sampler** (`generation/distributions/numeric.py`):

```python
class UniformSampler(BaseSampler):
    """Uniform distribution sampler."""
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        """
        Initialize uniform sampler.
        
        Args:
            low: Lower bound
            high: Upper bound
        """
        self.low = low
        self.high = high
        logger.debug(f"Initialized UniformSampler: [{low}, {high})")
    
    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n uniform samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.uniform(self.low, self.high, size=n)
```

**Example: Normal Sampler** (`generation/distributions/numeric.py`):

```python
class NormalSampler(BaseSampler):
    """Normal (Gaussian) distribution sampler."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize normal sampler.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation
        """
        self.mean = mean
        self.std = std
        logger.debug(f"Initialized NormalSampler: μ={mean}, σ={std}")
    
    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n normal samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        return rng.normal(self.mean, self.std, size=n)
```

**Example: Categorical Sampler** (`generation/distributions/categorical.py`):

```python
class CategoricalSampler(BaseSampler):
    """Categorical distribution sampler."""
    
    def __init__(self, values: List[str], probs: Optional[List[float]] = None):
        """
        Initialize categorical sampler.
        
        Args:
            values: List of categorical values
            probs: Optional probability distribution (must sum to 1)
        """
        self.values = values
        if probs is not None:
            if abs(sum(probs) - 1.0) > 1e-6:
                raise ValueError("Probabilities must sum to 1.0")
            if len(probs) != len(values):
                raise ValueError("Probabilities length must match values length")
            self.probs = probs
        else:
            # Uniform distribution
            self.probs = [1.0 / len(values)] * len(values)
    
    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Generate n categorical samples."""
        rng = kwargs.get("rng", np.random.default_rng())
        indices = rng.choice(len(self.values), size=n, p=self.probs)
        return np.array([self.values[i] for i in indices])
```

**Example: Zipf Sampler** (`generation/distributions/zipf.py`):

```python
class ZipfSampler(BaseSampler):
    """Zipf distribution sampler for generating skewed distributions."""
    
    def __init__(self, s: float = 1.2, n: Optional[int] = None):
        """
        Initialize Zipf sampler.
        
        Args:
            s: Zipf exponent (higher = more skewed, typical range 1.0-2.0)
            n: Domain size (number of distinct values)
        """
        self.s = s
        self.n = n
    
    def sample(self, n: int, rng: np.random.Generator, support=None, n_items=None, **kwargs):
        """
        Generate n samples from Zipf distribution.
        
        Two modes:
        1. With support (FK case): Samples from provided array of values
        2. Without support: Generates integer IDs 1..n_items
        
        Zipf probability: P(rank=k) = (1/k^s) / sum(1/i^s for i in 1..n)
        """
        n_items = self.n or n_items or len(support) if support is not None else None
        
        if n_items is None:
            raise ValueError("ZipfSampler requires either 'support' or 'n_items'")
        
        # Generate Zipf probabilities: P(k) = 1/k^s
        ranks = np.arange(1, n_items + 1)
        probs = 1.0 / (ranks ** self.s)
        probs = probs / probs.sum()  # Normalize
        
        # Sample indices according to Zipf distribution
        indices = rng.choice(n_items, size=n, p=probs)
        
        if support is not None:
            # Map indices to actual FK values
            return support[indices]
        else:
            # Return 1-indexed integer IDs
            return indices + 1
```

**Example: Seasonal Date Sampler** (`generation/distributions/seasonal.py`):

```python
class SeasonalDateSampler(BaseSampler):
    """Seasonal distribution sampler for temporal data."""
    
    def __init__(self, weights: Dict[str, float], granularity: str = "month"):
        """
        Initialize seasonal date sampler.
        
        Args:
            weights: Dict mapping period names to weights
                    (e.g., {"January": 0.1, "December": 0.15, ...})
            granularity: "month" or "week"
        """
        self.weights = weights
        self.granularity = granularity
        self._normalize_weights()  # Ensure weights sum to 1.0
    
    def sample(self, n: int, rng: np.random.Generator, year_range=(2020, 2023), **kwargs):
        """
        Generate n seasonal date samples.
        
        Process:
        1. Sample periods (months/weeks) according to weights
        2. Convert periods to actual dates
        3. Handle edge cases (leap years, week boundaries)
        """
        # Sample periods based on weights
        periods = list(self.weights.keys())
        probs = [self.weights[p] for p in periods]
        sampled_periods = rng.choice(periods, size=n, p=probs)
        
        # Convert periods to dates
        dates = []
        for period in sampled_periods:
            if self.granularity == "month":
                month = month_map[period]  # Map "January" -> 1
                year = rng.integers(year_range[0], year_range[1] + 1)
                max_day = calendar.monthrange(year, month)[1]  # Handle leap years
                day = rng.integers(1, max_day + 1)
                dates.append(np.datetime64(f"{year}-{month:02d}-{day:02d}"))
            else:  # week granularity
                # Parse week number and convert to ISO week date
                ...
        
        return np.array(dates)
```

---

### 4. Evaluation Framework

#### 4.1 Evaluation Configuration (`evaluation/config.py`)

**Threshold Configuration**:

```python
class ZipfConfig(BaseModel):
    """Configuration for Zipf distribution evaluation."""
    min_r2: float = 0.92  # Minimum R² for Zipf fit
    s_tolerance: float = 0.15  # Tolerance for Zipf exponent

class CategoricalConfig(BaseModel):
    """Configuration for categorical distribution evaluation."""
    min_chi2_pvalue: float = 0.01  # Minimum chi-square p-value

class NumericConfig(BaseModel):
    """Configuration for numeric distribution evaluation."""
    min_ks_pvalue: float = 0.01  # Minimum Kolmogorov-Smirnov p-value
    max_wasserstein: float = 0.10  # Maximum Wasserstein distance

class WorkloadConfig(BaseModel):
    """Configuration for workload evaluation."""
    max_runtime_sec: float = 5.0  # Maximum query runtime

class EvalThresholds(BaseModel):
    """All evaluation thresholds."""
    zipf: ZipfConfig = ZipfConfig()
    categorical: CategoricalConfig = CategoricalConfig()
    numeric: NumericConfig = NumericConfig()
    workload: WorkloadConfig = WorkloadConfig()

class EvaluationConfig(BaseModel):
    """Complete evaluation configuration."""
    thresholds: EvalThresholds = EvalThresholds()
```

#### 4.2 Evaluation Report Models (`evaluation/report_models.py`)

```python
class MetricResult(BaseModel):
    """Result of a single metric evaluation."""
    name: str                           # Metric name (e.g., "r2", "chi2_pvalue")
    value: float                        # Computed value
    threshold: Optional[float] = None   # Threshold for pass/fail
    passed: Optional[bool] = None      # Pass/fail status

class ColumnReport(BaseModel):
    """Evaluation report for a column."""
    table: str
    column: str
    family: str                         # Distribution family
    metrics: List[MetricResult] = Field(default_factory=list)

class TableReport(BaseModel):
    """Evaluation report for a table."""
    name: str
    row_count: int
    pk_ok: bool                         # Primary key valid
    fk_ok: bool                         # Foreign keys valid

class WorkloadReport(BaseModel):
    """Evaluation report for a workload query."""
    sql: str
    type: str
    elapsed_sec: float
    rows: int
    group_gini: Optional[float] = None   # Gini coefficient (for group_by queries)
    top1_share: Optional[float] = None   # Top group share (for group_by queries)
    passed: Optional[bool] = None

class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    schema: List[TableReport] = Field(default_factory=list)
    columns: List[ColumnReport] = Field(default_factory=list)
    workloads: List[WorkloadReport] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False
```

#### 4.3 Schema Validation (`evaluation/schema.py`)

**Primary Key and Foreign Key Checking**:

```python
def check_pk_fk(ir: DatasetIR) -> List[str]:
    """
    Check primary key and foreign key constraints.
    
    Returns:
        List of issue messages (empty if all checks pass)
    """
    issues = []
    tables = ir.logical.tables
    
    for table_name, table in tables.items():
        # Check primary key exists
        if not table.primary_key:
            issues.append(f"{table_name}: missing primary key")
            continue
        
        # Check primary key columns exist
        column_names = {c.name for c in table.columns}
        for pk_col in table.primary_key:
            if pk_col not in column_names:
                issues.append(
                    f"{table_name}: primary key column '{pk_col}' does not exist"
                )
        
        # Check foreign keys
        for fk in table.foreign_keys:
            # Check FK column exists
            if fk.column not in column_names:
                issues.append(
                    f"{table_name}: foreign key column '{fk.column}' does not exist"
                )
                continue
            
            # Check referenced table exists
            if fk.ref_table not in tables:
                issues.append(
                    f"{table_name}: foreign key '{fk.column}' references "
                    f"missing table '{fk.ref_table}'"
                )
                continue
            
            # Check referenced column exists
            ref_table = tables[fk.ref_table]
            ref_column_names = {c.name for c in ref_table.columns}
            if fk.ref_column not in ref_column_names:
                issues.append(
                    f"{table_name}: foreign key '{fk.column}' references "
                    f"'{fk.ref_table}.{fk.ref_column}' which does not exist"
                )
    
    return issues
```

**Referential Integrity Checking** (`evaluation/integrity.py`):

```python
def fk_coverage(
    ir: DatasetIR, dfs: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Check foreign key referential integrity coverage.
    
    Returns the fraction of FK values that exist in referenced tables.
    
    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame
    
    Returns:
        Dictionary mapping "table.column" -> coverage fraction (0.0 to 1.0)
    """
    coverage = {}
    
    for table_name, table in ir.logical.tables.items():
        if table_name not in dfs:
            logger.warning(f"Table '{table_name}' not found in dataframes")
            continue
        
        if not table.foreign_keys:
            continue
        
        fact_df = dfs[table_name]
        
        for fk in table.foreign_keys:
            if fk.column not in fact_df.columns:
                logger.warning(
                    f"FK column '{fk.column}' not found in table '{table_name}'"
                )
                continue
            
            if fk.ref_table not in dfs:
                logger.warning(
                    f"Referenced table '{fk.ref_table}' not found in dataframes"
                )
                continue
            
            dim_df = dfs[fk.ref_table]
            if fk.ref_column not in dim_df.columns:
                logger.warning(
                    f"Referenced column '{fk.ref_column}' not found in "
                    f"table '{fk.ref_table}'"
                )
                continue
            
            # Calculate coverage
            fk_values = fact_df[fk.column]
            ref_values = set(dim_df[fk.ref_column].unique())
            coverage_frac = fk_values.isin(ref_values).mean()
            key = f"{table_name}.{fk.column}"
            coverage[key] = float(coverage_frac)
            
            if coverage_frac < 0.999:
                logger.warning(
                    f"FK coverage for {key}: {coverage_frac:.4f} "
                    f"(expected >= 0.999)"
                )
    
    return coverage
```

#### 4.4 Workload Evaluation (`evaluation/workload.py`)

**DuckDB Integration**:

```python
def run_workloads(
    ir: DatasetIR,
    dfs: Dict[str, pd.DataFrame]
) -> List[Dict[str, Any]]:
    """
    Run workload queries using DuckDB.
    
    Process:
    1. Create DuckDB connection
    2. Register DataFrames as tables
    3. Generate SQL queries from WorkloadSpec
    4. Execute queries and measure performance
    5. Calculate skew metrics (Gini coefficient, top-k share)
    """
    conn = duckdb.connect()
    
    # Register DataFrames as tables
    for table_name, df in dfs.items():
        conn.register(table_name, df)
    
    results = []
    for spec in ir.workload.targets:
        sql = _generate_query(spec, ir)
        
        start_time = time.time()
        result_df = conn.execute(sql).df()
        elapsed_sec = time.time() - start_time
        
        # Calculate metrics for group_by queries only
        group_gini = None
        top1_share = None
        
        if spec.type == "group_by" and len(result_df) > 0:
            # Assume first numeric column is the count/aggregate
            numeric_cols = result_df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                counts = result_df[numeric_cols[0]].values
                group_gini = gini_coefficient(counts)  # Measure skew
                top1_share = top_k_share(counts, k=1)  # Top group's share
        
        results.append({
            "sql": sql,
            "type": spec.type,
            "elapsed_sec": elapsed_sec,
            "rows": len(result_df),
            "group_gini": group_gini,
            "top1_share": top1_share,
        })
    
    return results
```

**Query Generation**:
- Uses `spec.query_hint` if provided
- Otherwise generates queries based on `spec.type`:
  - `group_by`: Groups by FK column, counts rows
  - `join`: Joins fact table with dimension tables
  - `filter`: Filters on numeric columns

#### 4.5 Evaluation Process (`evaluation/report_builder.py`)

```python
def evaluate(
    ir: DatasetIR,
    dfs: Dict[str, pd.DataFrame],
    cfg: EvaluationConfig
) -> EvaluationReport:
    """
    Evaluate generated data against IR specifications.
    
    Args:
        ir: Dataset IR
        dfs: Dictionary of table name -> DataFrame
        cfg: Evaluation configuration
    
    Returns:
        Evaluation report
    """
    logger.info("Starting evaluation")
    
    # Schema validation
    issues = check_pk_fk(ir)
    fk_cov = fk_coverage(ir, dfs)
    
    # Build table reports
    table_reports: List[TableReport] = []
    for table_name, table in ir.logical.tables.items():
        if table_name not in dfs:
            logger.warning(f"Table '{table_name}' not found in dataframes")
            continue
        
        df = dfs[table_name]
        pk_ok = len(table.primary_key) > 0
        fk_ok = all(
            fk_cov.get(f"{table_name}.{fk.column}", 0.0) > 0.999
            for fk in table.foreign_keys
        )
        
        table_reports.append(
            TableReport(
                name=table_name,
                row_count=len(df),
                pk_ok=pk_ok,
                fk_ok=fk_ok,
            )
        )
    
    # Build column reports
    column_reports: List[ColumnReport] = []
    for cg in ir.generation.columns:
        if cg.table not in dfs:
            continue
        
        df = dfs[cg.table]
        if cg.column not in df.columns:
            continue
        
        values = df[cg.column].values
        metrics: List[MetricResult] = []
        
        # Evaluate based on distribution type
        dist = cg.distribution
        
        if dist.kind == "zipf":
            r2, s = zipf_fit(values)
            metrics.append(
                MetricResult(
                    name="r2",
                    value=r2,
                    threshold=cfg.thresholds.zipf.min_r2,
                    passed=r2 >= cfg.thresholds.zipf.min_r2,
                )
            )
            metrics.append(
                MetricResult(
                    name="exponent",
                    value=s,
                    threshold=dist.s,
                    passed=abs(s - dist.s) <= cfg.thresholds.zipf.s_tolerance,
                )
            )
        
        elif dist.kind == "categorical":
            # Count frequencies
            value_counts = pd.Series(values).value_counts()
            if dist.domain.probs:
                # Build observed counts for each domain value
                observed = np.array([
                    value_counts.get(v, 0) for v in dist.domain.values
                ])
                expected = np.array(dist.domain.probs) * len(values)
                chi2, p_value = chi_square_test(observed, expected)
            else:
                # Uniform distribution
                observed = value_counts.values
                chi2, p_value = chi_square_test(observed)
            
            metrics.append(
                MetricResult(
                    name="chi2_pvalue",
                    value=p_value,
                    threshold=cfg.thresholds.categorical.min_chi2_pvalue,
                    passed=p_value >= cfg.thresholds.categorical.min_chi2_pvalue,
                )
            )
        
        elif dist.kind in ("uniform", "normal"):
            # For numeric distributions, use KS test
            ks_stat, ks_pvalue = ks_test(values)
            metrics.append(
                MetricResult(
                    name="ks_pvalue",
                    value=ks_pvalue,
                    threshold=cfg.thresholds.numeric.min_ks_pvalue,
                    passed=ks_pvalue >= cfg.thresholds.numeric.min_ks_pvalue,
                )
            )
        
        elif dist.kind == "seasonal":
            # Simplified seasonal check
            if hasattr(values[0], "month"):
                months = [v.month for v in values]
                month_counts = pd.Series(months).value_counts().sort_index()
                # Compare with expected weights (simplified)
                metrics.append(
                    MetricResult(
                        name="seasonal_check",
                        value=1.0,  # Placeholder
                        passed=True,
                    )
                )
        
        column_reports.append(
            ColumnReport(
                table=cg.table,
                column=cg.column,
                family=dist.kind,
                metrics=metrics,
            )
        )
    
    # Run workloads
    wl_raw = run_workloads(ir, dfs)
    wl_reports: List[WorkloadReport] = []
    for w in wl_raw:
        passed = None
        if "error" not in w:
            passed = w["elapsed_sec"] <= cfg.thresholds.workload.max_runtime_sec
        
        wl_reports.append(
            WorkloadReport(
                sql=w["sql"],
                type=w["type"],
                elapsed_sec=w["elapsed_sec"],
                rows=w["rows"],
                group_gini=w.get("group_gini"),
                top1_share=w.get("top1_share"),
                passed=passed,
            )
        )
    
    # Calculate summary
    failures = len(issues)
    for cr in column_reports:
        failures += sum(1 for m in cr.metrics if m.passed is False)
    
    failures += sum(1 for wr in wl_reports if wr.passed is False)
    
    passed = failures == 0
    
    logger.info(f"Evaluation completed: {failures} failures, passed={passed}")
    
    return EvaluationReport(
        schema=table_reports,
        columns=column_reports,
        workloads=wl_reports,
        summary={
            "failures": failures,
            "total_checks": len(issues) + len(column_reports) + len(wl_reports)
        },
        passed=passed,
    )
```

#### 4.6 Relational Evaluation (`evaluation/relational_eval.py`)

**Purpose**: Evaluates relational properties (FK coverage, degree distributions, join selectivity).

**Key Functions**:

1. **`fk_coverage_duckdb(con, child_table, fk_column, parent_table, pk_column="id") -> float`**:
   Computes fraction of FK values that have valid references using DuckDB.

```python
def fk_coverage_duckdb(
    con,
    child_table: str,
    fk_column: str,
    parent_table: str,
    pk_column: str = "id",
) -> float:
    """
    Compute fraction of FK values that have valid references using DuckDB.
    
    Returns:
        Coverage fraction (0.0 to 1.0, where 1.0 = perfect coverage)
    """
    # Total rows in child table
    tot = con.sql(f"SELECT COUNT(*) as cnt FROM {child_table}").fetchone()[0]
    
    # Valid FK references
    val = con.sql(f"""
        SELECT COUNT(*) as cnt
        FROM {child_table} c
        WHERE EXISTS (
            SELECT 1 FROM {parent_table} p
            WHERE p.{pk_column} = c.{fk_column}
        )
    """).fetchone()[0]
    
    return float(val / tot)
```

2. **`degree_histogram(con, child_table, fk_column, parent_table, pk_column="id") -> Tuple[np.ndarray, np.ndarray]`**:
   Computes histogram of children per parent (degree distribution).

```python
def degree_histogram(
    con,
    child_table: str,
    fk_column: str,
    parent_table: str,
    pk_column: str = "id",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of children per parent (degree distribution).
    
    Returns:
        Tuple of (histogram counts, bin edges)
    """
    query = f"""
        SELECT p.{pk_column} as pid, COUNT(c.{fk_column}) as deg
        FROM {parent_table} p
        LEFT JOIN {child_table} c ON p.{pk_column} = c.{fk_column}
        GROUP BY p.{pk_column}
    """
    df = con.sql(query).df()
    degrees = df["deg"].values
    hist, bins = np.histogram(degrees, bins="auto")
    return hist, bins
```

3. **`degree_distribution_divergence(real_hist, synth_hist) -> float`**:
   Computes Wasserstein distance between degree distributions.

4. **`join_selectivity(con, query, base_rowcount) -> float`**:
   Computes join selectivity (fraction of base rows that match).

5. **`evaluate_relational_metrics(ir, dfs, con=None) -> Dict[str, float]`**:
   Evaluates all relational metrics for a dataset.

```python
def evaluate_relational_metrics(
    ir: DatasetIR,
    dfs: Dict[str, pd.DataFrame],
    con: Optional[any] = None,
) -> Dict[str, float]:
    """
    Evaluate all relational metrics for a dataset.
    
    Returns:
        Dictionary of metric names -> values:
        - {table}.{fk_column}_coverage: FK coverage for each FK
        - avg_fk_coverage: Average FK coverage
        - min_fk_coverage: Minimum FK coverage
        - {table}.{fk_column}_avg_degree: Average degree for each FK
        - {table}.{fk_column}_max_degree: Maximum degree for each FK
    """
    metrics = {}
    
    # FK coverage for all foreign keys
    fk_coverages = []
    for table_name, table in ir.logical.tables.items():
        for fk in table.foreign_keys:
            coverage = fk_coverage_duckdb(
                con, table_name, fk.column, fk.ref_table, pk_col
            )
            fk_coverages.append(coverage)
            metrics[f"{table_name}.{fk.column}_coverage"] = coverage
    
    if fk_coverages:
        metrics["avg_fk_coverage"] = float(np.mean(fk_coverages))
        metrics["min_fk_coverage"] = float(np.min(fk_coverages))
    
    # Degree distributions for fact->dimension joins
    for table_name, table in ir.logical.tables.items():
        if table.kind != "fact":
            continue
        for fk in table.foreign_keys:
            hist = degree_histogram(con, table_name, fk.column, fk.ref_table, pk_col)
            # Store histogram statistics
            # ...
    
    return metrics
```

#### 4.7 Schema Evaluation (`evaluation/schema_eval.py`)

**Purpose**: Evaluates schema-level properties (coverage metrics, graph edit distance).

**Key Functions**:

1. **`schema_coverage(gold_ir, synth_ir) -> Dict[str, float]`**:
   Computes coverage metrics comparing synthetic schema to gold schema.

```python
def schema_coverage(
    gold_ir: LogicalIR,
    synth_ir: LogicalIR,
) -> Dict[str, float]:
    """
    Compute coverage metrics comparing synthetic schema to gold schema.
    
    Returns:
        Dictionary with precision, recall, F1 for:
        - tables: Table-level metrics
        - columns: Column-level metrics (for common tables)
        - pk: Primary key metrics
        - fk: Foreign key metrics
    """
    gold_tables = set(gold_ir.tables.keys())
    synth_tables = set(synth_ir.tables.keys())
    
    # Table-level metrics
    table_intersection = gold_tables & synth_tables
    table_precision = len(table_intersection) / len(synth_tables) if synth_tables else 0.0
    table_recall = len(table_intersection) / len(gold_tables) if gold_tables else 0.0
    table_f1 = 2 * table_precision * table_recall / (table_precision + table_recall) if (table_precision + table_recall) > 0 else 0.0
    
    # Column-level metrics (for common tables)
    # ... compute column precision, recall, F1
    
    # PK/FK metrics
    # ... compute PK and FK precision, recall, F1
    
    return {
        "table_precision": table_precision,
        "table_recall": table_recall,
        "table_f1": table_f1,
        "column_precision": col_precision,
        "column_recall": col_recall,
        "column_f1": col_f1,
        "pk_precision": pk_precision,
        "pk_recall": pk_recall,
        "pk_f1": pk_f1,
        "fk_precision": fk_precision,
        "fk_recall": fk_recall,
        "fk_f1": fk_f1,
    }
```

2. **`schema_graph(ir) -> NetworkX.DiGraph`**:
   Builds schema graph from LogicalIR (nodes = tables, edges = foreign keys).

3. **`graph_edit_distance(gold_ir, synth_ir, timeout=5) -> Optional[float]`**:
   Computes graph edit distance between gold and synthetic schemas.

```python
def graph_edit_distance(
    gold_ir: LogicalIR,
    synth_ir: LogicalIR,
    timeout: int = 5,
) -> Optional[float]:
    """
    Compute graph edit distance between gold and synthetic schemas.
    
    Uses NetworkX's optimized graph edit distance algorithm.
    
    Returns:
        Graph edit distance (lower is better), or None if timeout/error
    """
    G_gold = schema_graph(gold_ir)
    G_synth = schema_graph(synth_ir)
    
    # Use optimized graph edit distance
    ged = nx.optimize_graph_edit_distance(
        G_gold, G_synth, timeout=timeout
    )
    return float(next(ged))
```

#### 4.8 Table Evaluation (`evaluation/table_eval.py`)

**Purpose**: Evaluates data-level properties (marginal distributions, correlations, mutual information).

**Key Functions**:

1. **`numeric_marginals(real, synth) -> Dict[str, float]`**:
   Computes marginal distribution metrics for numeric columns.

```python
def numeric_marginals(
    real: np.ndarray,
    synth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute marginal distribution metrics for numeric columns.
    
    Returns:
        Dictionary with:
        - ks_statistic: Kolmogorov-Smirnov statistic
        - ks_pvalue: KS test p-value
        - wasserstein_distance: Earth Mover's Distance
    """
    # Kolmogorov-Smirnov test
    ks_result = ks_2samp(real, synth)
    ks_stat = float(ks_result.statistic)
    
    # Wasserstein distance (Earth Mover's Distance)
    w1_dist = float(wasserstein_distance(real, synth))
    
    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": float(ks_result.pvalue),
        "wasserstein_distance": w1_dist,
    }
```

2. **`categorical_marginals(real, synth) -> Dict[str, float]`**:
   Computes marginal distribution metrics for categorical columns.

```python
def categorical_marginals(
    real: np.ndarray,
    synth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute marginal distribution metrics for categorical columns.
    
    Returns:
        Dictionary with:
        - chi2_statistic: Chi-square statistic
        - chi2_pvalue: Chi-square test p-value
    """
    # Get unique values from both
    all_values = np.unique(np.concatenate([real, synth]))
    
    # Count frequencies
    real_counts = np.array([np.sum(real == val) for val in all_values])
    synth_counts = np.array([np.sum(synth == val) for val in all_values])
    
    # Expected frequencies based on real distribution
    expected = (real_counts / real_counts.sum()) * synth_counts.sum()
    
    # Chi-square test
    chi2_result = chisquare(synth_counts, f_exp=expected)
    
    return {
        "chi2_statistic": float(chi2_result.statistic),
        "chi2_pvalue": float(chi2_result.pvalue),
    }
```

3. **`correlation_metrics(real_col1, real_col2, synth_col1, synth_col2) -> Dict[str, float]`**:
   Computes correlation metrics between two columns.

```python
def correlation_metrics(
    real_col1: np.ndarray,
    real_col2: np.ndarray,
    synth_col1: np.ndarray,
    synth_col2: np.ndarray,
) -> Dict[str, float]:
    """
    Compute correlation metrics between two columns.
    
    Returns:
        Dictionary with:
        - pearson_delta: Absolute difference in Pearson correlation
        - spearman_delta: Absolute difference in Spearman correlation
    """
    # Pearson correlation
    real_pearson, _ = pearsonr(real_col1, real_col2)
    synth_pearson, _ = pearsonr(synth_col1, synth_col2)
    pearson_delta = abs(float(real_pearson - synth_pearson))
    
    # Spearman correlation
    real_spearman, _ = spearmanr(real_col1, real_col2)
    synth_spearman, _ = spearmanr(synth_col1, synth_col2)
    spearman_delta = abs(float(real_spearman - synth_spearman))
    
    return {
        "pearson_delta": pearson_delta,
        "spearman_delta": spearman_delta,
    }
```

4. **`mutual_information(cat_col1, cat_col2) -> float`**:
   Computes mutual information between two categorical columns.

5. **`table_fidelity_score(marginal_scores, pairwise_scores, marginal_weight=0.7) -> float`**:
   Computes aggregate table fidelity score.

```python
def table_fidelity_score(
    marginal_scores: Dict[str, float],
    pairwise_scores: Dict[str, float],
    marginal_weight: float = 0.7,
) -> float:
    """
    Compute aggregate table fidelity score.
    
    Args:
        marginal_scores: Dictionary of marginal metric scores (normalized 0-1)
        pairwise_scores: Dictionary of pairwise metric scores (normalized 0-1)
        marginal_weight: Weight for marginal scores (default: 0.7)
    
    Returns:
        Fidelity score (0-1, higher is better)
    """
    marginal_avg = np.mean(list(marginal_scores.values())) if marginal_scores else 0.0
    pairwise_avg = np.mean(list(pairwise_scores.values())) if pairwise_scores else 0.0
    
    fidelity = marginal_weight * marginal_avg + (1 - marginal_weight) * pairwise_avg
    return float(fidelity)
```

#### 4.9 Statistical Tests (`evaluation/stats.py`)

**Zipf Fitting**:
```python
def zipf_fit(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Zipf distribution to values and return R² and exponent.
    
    Returns:
        (R², exponent s)
    """
    # Count frequencies
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) < 2:
        return 0.0, 0.0
    
    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    frequencies = counts[sorted_indices]
    ranks = np.arange(1, len(frequencies) + 1)
    
    # Fit log-log linear regression: log(freq) ~ log(rank)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    
    # Remove zeros/infs
    valid = np.isfinite(log_ranks) & np.isfinite(log_freqs)
    if np.sum(valid) < 2:
        return 0.0, 0.0
    
    # Use scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_ranks[valid], log_freqs[valid]
    )
    s = -slope  # Zipf exponent (negate because slope is negative)
    
    return r_value ** 2, s  # R² is correlation coefficient squared
```

**Chi-Square Test**:
```python
def chi_square_test(observed: np.ndarray, expected: Optional[np.ndarray] = None):
    """
    Perform chi-square goodness-of-fit test.
    
    If expected is None, tests against uniform distribution.
    Uses scipy.stats.chisquare for computation.
    """
    if expected is None:
        expected = np.full_like(observed, observed.sum() / len(observed))
    
    # Remove zeros
    mask = (observed > 0) | (expected > 0)
    obs = observed[mask]
    exp = expected[mask]
    
    chi2, p_value = stats.chisquare(obs, exp)
    return float(chi2), float(p_value)
```

**Gini Coefficient** (for measuring inequality/skew in group_by queries):

```python
def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient for measuring inequality.
    
    Args:
        values: Array of values
    
    Returns:
        Gini coefficient (0 to 1, higher = more unequal)
    """
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (
        n + 1
    ) / n
```

**Top-K Share** (for measuring concentration in group_by queries):

```python
def top_k_share(values: np.ndarray, k: int = 1) -> float:
    """
    Calculate the share of top-k values.
    
    Args:
        values: Array of values (e.g., group counts)
        k: Number of top values to consider
    
    Returns:
        Share of top-k values (0 to 1)
    """
    if len(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)[::-1]
    top_k_sum = np.sum(sorted_values[:k])
    total_sum = np.sum(sorted_values)
    
    if total_sum == 0:
        return 0.0
    
    return float(top_k_sum / total_sum)
```

---

## Data Flow

### Complete Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 1. Schema Validation                                         │  │
│  │    - Primary Key Validation (PK exists, columns exist)       │  │
│  │    - Foreign Key Validation (FK refs valid)                  │  │
│  │    - Constraint Validation (Composite PK, FD, Implications)  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 2. Referential Integrity (DuckDB)                           │  │
│  │    - FK Coverage (target: >= 0.999)                          │  │
│  │    - Degree Distributions (children per parent)              │  │
│  │    - Join Selectivity                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 3. Statistical Validation                                   │  │
│  │    - Zipf: R² >= 0.92, |s - expected| <= 0.15               │  │
│  │    - Categorical: Chi-square p-value >= 0.01                 │  │
│  │    - Numeric: KS p-value >= 0.01, W1 <= 0.10                │  │
│  │    - Seasonal: Month/week distribution alignment            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 4. Schema-Level Evaluation (Optional)                        │  │
│  │    - Schema Coverage (precision/recall/F1)                   │  │
│  │    - Graph Edit Distance (NetworkX)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 5. Table-Level Evaluation (Optional)                         │  │
│  │    - Marginal Distributions (KS, Wasserstein, Chi-square)   │  │
│  │    - Pairwise Metrics (Pearson, Spearman, MI)               │  │
│  │    - Table Fidelity Score (weighted average)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ 6. Workload Evaluation                                        │  │
│  │    - Query Execution (runtime < 5.0 seconds)                 │  │
│  │    - Skew Metrics (Gini coefficient, top-1 share)           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│                    EvaluationReport                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ - schema: List[TableReport]                                  │  │
│  │ - columns: List[ColumnReport]                                │  │
│  │ - workloads: List[WorkloadReport]                          │  │
│  │ - summary: Dict[str, float]                                 │  │
│  │ - passed: bool                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                  │
│  Natural Language: "Generate a retail sales dataset..."             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MANAGER AGENT                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 1. Load prompts (manager_system.txt, manager_user.txt)     │   │
│  │ 2. Render template with NL input                           │   │
│  │ 3. Call LLM API                                            │   │
│  │ 4. Extract JSON from response                              │   │
│  │ 5. Validate as RequirementIR                               │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.requirement_ir                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 CONCEPTUAL DESIGNER AGENT                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Input: RequirementIR                                       │   │
│  │ Process: Convert requirements to ER model                 │   │
│  │ Output: ConceptualIR (entities, relationships)            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.conceptual_ir                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LOGICAL DESIGNER AGENT                             │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Input: ConceptualIR + RequirementIR                       │   │
│  │ Process: Convert ER model to relational schema           │   │
│  │ Output: LogicalIR (tables, columns, PK/FK)                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.logical_ir                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               DISTRIBUTION ENGINEER AGENT                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Input: LogicalIR + RequirementIR                          │   │
│  │ Process: Design distribution specs for each column       │   │
│  │ Output: GenerationIR (distributions, domains)           │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.generation_ir                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                WORKLOAD DESIGNER AGENT                              │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Input: LogicalIR + RequirementIR                          │   │
│  │ Process: Design query workload specifications             │   │
│  │ Output: WorkloadIR (query specs)                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.workload_ir                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   QA COMPILER AGENT                                 │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Input: All IRs                                             │   │
│  │ Process:                                                  │   │
│  │   1. Combine into DatasetIR                               │   │
│  │   2. Validate logical schema (PK/FK)                      │   │
│  │   3. Validate generation specs                            │   │
│  │ Output: DatasetIR                                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    Blackboard.dataset_ir                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION ENGINE                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Build Derived Registry                            │   │
│  │   - Compile derived expressions                           │   │
│  │   - Extract dependencies                                  │   │
│  │   - Topological sort                                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Generate Dimension Tables                        │   │
│  │   For each dimension table:                               │   │
│  │     - Phase 1: Generate base columns                     │   │
│  │     - Phase 2: Compute derived columns                   │   │
│  │     - Write CSV                                           │   │
│  │     - Store in dim_dfs for FK references                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Step 3: Generate Fact Tables (Streaming)                  │   │
│  │   For each fact table:                                    │   │
│  │     - Generate chunks (chunk_rows rows each)              │   │
│  │     - Phase 1: Generate base columns                      │   │
│  │     - Phase 2: Compute derived columns                   │   │
│  │     - Stream CSV chunks                                   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    CSV Files in out_dir/                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                             │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 1. Schema Validation                                       │   │
│  │    - Check PK/FK integrity                                 │   │
│  │    - Check FK coverage                                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 2. Statistical Validation                                  │   │
│  │    - Zipf fitting (R², exponent)                         │   │
│  │    - Chi-square test (categorical)                       │   │
│  │    - KS test (numeric)                                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 3. Workload Testing                                        │   │
│  │    - Execute queries                                       │   │
│  │    - Measure runtime                                       │   │
│  │    - Check skew metrics                                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│                    EvaluationReport                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Execution Flow                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Manager    │                                               │
│  │   Agent      │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Load prompts                                       │
│         │ 2. Call LLM with NL input                              │
│         │ 3. Extract & validate JSON                            │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Shared State)                    │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ requirement_ir: RequirementIR                      │  │  │
│  │  │   - domain, narrative, scale, distributions        │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         │ Reads requirement_ir                                  │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ Conceptual   │                                               │
│  │   Designer    │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Load prompts                                       │
│         │ 2. Call LLM with RequirementIR                       │
│         │ 3. Extract & validate JSON                            │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Updated)                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ conceptual_ir: ConceptualIR                        │  │  │
│  │  │   - entities, relationships, attributes            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         │ Reads requirement_ir + conceptual_ir                  │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Logical    │                                               │
│  │   Designer   │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Load prompts                                       │
│         │ 2. Call LLM with ConceptualIR + RequirementIR      │
│         │ 3. Extract & validate JSON                            │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Updated)                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ logical_ir: LogicalIR                              │  │  │
│  │  │   - tables, columns, PK/FK, constraints            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         │ Reads logical_ir + requirement_ir                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ Distribution │                                               │
│  │   Engineer   │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Load prompts                                       │
│         │ 2. Call LLM with LogicalIR + RequirementIR          │
│         │ 3. Extract & validate JSON                            │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Updated)                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ generation_ir: GenerationIR                         │  │  │
│  │  │   - column distributions (uniform, zipf, etc.)     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         │ Reads logical_ir + requirement_ir                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  Workload    │                                               │
│  │   Designer   │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Load prompts                                       │
│         │ 2. Call LLM with LogicalIR + RequirementIR          │
│         │ 3. Extract & validate JSON                            │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Updated)                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ workload_ir: WorkloadIR                            │  │  │
│  │  │   - query specifications (group_by, join, filter) │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         │ Reads all IRs                                         │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   QA         │                                               │
│  │  Compiler    │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         │ 1. Combine IRs into DatasetIR                        │
│         │ 2. Validate schema (PK/FK)                           │
│         │ 3. Validate generation specs                          │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Blackboard (Final)                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ dataset_ir: DatasetIR                              │  │  │
│  │  │   - logical: LogicalIR                              │  │  │
│  │  │   - generation: GenerationIR                        │  │  │
│  │  │   - workload: WorkloadIR (optional)                │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Agent Execution Pattern:
┌─────────────────────────────────────────────────────────────┐
│ For each agent:                                              │
│                                                              │
│  1. Read from Blackboard                                     │
│     ├─ Previous IRs (input)                                 │
│     └─ RequirementIR (context)                             │
│                                                              │
│  2. Load Prompts                                            │
│     ├─ {agent}_system.txt (role definition)                │
│     └─ {agent}_user.txt (template with placeholders)       │
│                                                              │
│  3. Render User Prompt                                      │
│     └─ Fill placeholders with IR JSON                       │
│                                                              │
│  4. Call LLM API                                            │
│     ├─ System prompt                                        │
│     └─ User prompt (with IR data)                          │
│                                                              │
│  5. Extract JSON from Response                              │
│     ├─ Handle markdown code blocks                         │
│     ├─ Handle plain JSON                                   │
│     └─ Multiple fallback strategies                        │
│                                                              │
│  6. Validate as IR Model                                    │
│     └─ Pydantic validation                                  │
│                                                              │
│  7. Write to Blackboard                                     │
│     └─ Store new IR                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. IR-Centric Architecture

**Decision**: All transformations operate on Pydantic IR models.

**Rationale**:
- Type safety and validation at each stage
- Easy to serialize/deserialize (JSON)
- Clear contracts between components
- Enables debugging and inspection

**Implementation**:
- All IRs inherit from `pydantic.BaseModel`
- Validation happens automatically on creation
- JSON serialization for persistence

### 2. Prompt-File Driven

**Decision**: Prompts stored as text files, not hardcoded.

**Rationale**:
- Easy to iterate on prompts without code changes
- Version control friendly
- Can be edited by non-developers
- Supports A/B testing of prompts

**Implementation**:
```python
def load_prompt(path: str) -> str:
    """Load prompt file from prompts/ directory."""
    full_path = PROMPTS_DIR / path
    return full_path.read_text(encoding="utf-8")

def render_prompt(template: str, **kwargs) -> str:
    """Render template with placeholders."""
    return template.format(**kwargs)
```

### 3. Multi-Agent System

**Decision**: Decompose NL→IR into specialized agents.

**Rationale**:
- Each agent has focused responsibility
- Easier to debug and improve individual stages
- Can swap agents independently
- Matches human design process (requirements → conceptual → logical → implementation)

**Agent Responsibilities**:
- **Manager**: Extract structured requirements
- **ConceptualDesigner**: High-level ER design
- **LogicalDesigner**: Relational schema design
- **DistributionEngineer**: Data generation specs
- **WorkloadDesigner**: Query workload specs
- **QACompiler**: Validation and compilation

### 4. Streaming Generation

**Decision**: Generate fact tables in chunks, not all at once.

**Rationale**:
- Handles very large tables (50M+ rows)
- Memory efficient
- Can start writing CSV before generation completes

**Implementation**:
```python
def generate_fact_stream(...) -> Iterator[pd.DataFrame]:
    """Yields DataFrame chunks."""
    while produced < n_total:
        m = min(chunk_rows, n_total - produced)
        df_chunk = generate_chunk(...)
        yield df_chunk
        produced += m
```

### 5. Derived Column Dependency Tracking

**Decision**: Use topological sort for derived column ordering.

**Rationale**:
- Ensures dependencies are computed before dependents
- Detects circular dependencies
- Enables parallel computation (future)

**Implementation**:
- AST-based dependency extraction
- Kahn's algorithm for topological sort
- Per-table ordering

### 6. Distribution Sampler Factory

**Decision**: Use factory pattern for distribution samplers.

**Rationale**:
- Easy to add new distribution types
- Consistent interface
- Context-aware (e.g., FK support for Zipf)

**Implementation**:
```python
def get_sampler(dist: Distribution, **ctx) -> BaseSampler:
    """Factory function creates appropriate sampler."""
    if isinstance(dist, DistZipf):
        return ZipfSampler(dist.s, dist.n or ctx.get("n_items"))
    ...
```

---

## Implementation Details

### Configuration Management (`config/settings.py`)

**Environment Variables**:
```python
class Settings(BaseSettings):
    # LLM Configuration
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    model_name: Optional[str] = None
    llm_url: Optional[str] = None  # Local OpenAI-compatible API
    model: Optional[str] = None
    temperature: float = 0.2
    
    # Generation Configuration
    seed: int = 7
    output_dir: Path = Path("output")
    chunk_rows: int = 1_000_000
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
```

**LLM Selection Priority**:
1. OpenAI (if `OPENAI_API_KEY` and `MODEL_NAME` set)
2. Local API (if `LLM_URL` and `MODEL` set)
3. Gemini (if `GEMINI_API_KEY` and `GEMINI_MODEL` set)

**Example `.env` file**:
```bash
# LLM Configuration (choose one)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-pro

# OR use OpenAI
# OPENAI_API_KEY=your_openai_key_here
# MODEL_NAME=gpt-4

# OR use local LLM server
# LLM_URL=http://localhost:11434
# MODEL=llama2

# Generation Configuration
TEMPERATURE=0.2
SEED=7
OUTPUT_DIR=output
CHUNK_ROWS=1000000

# Logging
LOG_LEVEL=INFO
```

### LLM Client (`agents/tools/llm_client.py`)

**Multi-Provider Support**:

The system supports three LLM providers with automatic fallback:

```python
def chat(messages: List[Dict[str, str]]) -> str:
    """
    Unified interface for multiple LLM providers.
    
    Priority order:
    1. OpenAI (if OPENAI_API_KEY and MODEL_NAME set)
    2. Local API (if LLM_URL and MODEL set) - OpenAI-compatible
    3. Gemini (if GEMINI_API_KEY and GEMINI_MODEL set)
    """
    if use_openai:
        return _chat_openai(messages)
    elif use_local:
        return _chat_local(messages)
    else:
        return _chat_gemini(messages)
```

**OpenAI Provider**:
- Uses official OpenAI Python SDK
- Standard chat completion API
- Temperature control

**Local Provider** (OpenAI-compatible):
- Supports local LLM servers (e.g., Ollama, vLLM)
- Custom base URL configuration
- **Retry logic** for transient errors:
  ```python
  max_retries = 3
  retry_delay = 2  # seconds
  for attempt in range(max_retries):
      try:
          response = client.chat.completions.create(...)
          return response.choices[0].message.content
      except Exception as e:
          # Retry on "Model reloaded" or 503 errors
          if is_transient_error(e) and attempt < max_retries - 1:
              time.sleep(retry_delay)
              continue
          raise
  ```
- 10-minute timeout for slow local models

**Gemini Provider**:
- Uses Google Generative AI SDK
- Combines system + user prompts (Gemini format)
- Handles response parts (text or parts array)

**Key Features**:
- **Provider abstraction**: Same interface for all providers
- **Error handling**: Retry logic for transient failures
- **Logging**: Debug information for API calls
- **Timeout handling**: Long timeouts for local models

### JSON Extraction (`agents/tools/json_parser.py`)

**Robust Parsing with Multiple Strategies**:

```python
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM output using multiple strategies.
    
    LLMs often wrap JSON in markdown code blocks or add explanatory text.
    This function tries multiple extraction methods:
    
    1. Markdown code blocks: ```json {...} ```
    2. Plain JSON object: {...}
    3. Entire text as JSON
    """
    # Strategy 1: Markdown code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Plain JSON object
    json_obj_pattern = r"\{.*\}"
    match = re.search(json_obj_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    raise JSONParseError("Could not extract valid JSON")
```

**Key Features**:
- **Multiple strategies**: Handles various LLM output formats
- **Regex-based**: Efficient extraction without full parsing
- **Error handling**: Clear error messages with context
- **Logging**: Debug information for troubleshooting

### Validation (`ir/validators.py`)

**Enhanced Validation System with QaIssue**:

The validation system uses a structured `QaIssue` model to track validation problems:

```python
@dataclass
class QaIssue:
    """QA issue found during validation."""
    stage: Literal["LogicalIR", "GenerationIR", "PostGen"]
    code: str  # e.g., "MISSING_PK", "FK_REF_INVALID"
    location: str  # e.g., "table_name" or "table_name.column_name"
    message: str
    details: dict = field(default_factory=dict)
```

**Validation Functions**:

1. **`validate_logical(ir: DatasetIR) -> List[QaIssue]`**:
   - Primary keys exist and reference valid columns
   - Foreign keys reference existing tables/columns
   - Returns list of QaIssue objects

2. **`validate_generation(ir: DatasetIR) -> List[QaIssue]`**:
   - All generation specs reference existing tables/columns
   - No orphaned specs

3. **`validate_dataset(ir: DatasetIR) -> List[QaIssue]`**:
   - Combines logical and generation validation

4. **`validate_logical_ir(board: Blackboard) -> List[QaIssue]`**:
   - Schema mode consistency checks
   - Composite PK validation
   - FD constraint validation
   - Implication constraint validation

**Issue Codes**:
- `MISSING_PK`: Table missing primary key
- `PK_COL_MISSING`: Primary key column doesn't exist
- `FK_REF_TABLE_MISSING`: Foreign key references missing table
- `FK_COL_MISSING`: Foreign key column doesn't exist
- `FK_REF_COL_MISSING`: Foreign key references missing column
- `GEN_TABLE_MISSING`: Generation spec references unknown table
- `GEN_COL_MISSING`: Generation spec references unknown column
- `SCHEMA_MODE_MISMATCH`: Schema mode mismatch between IRs
- `COMPOSITE_PK_TABLE_MISSING`: Composite PK references unknown table
- `COMPOSITE_PK_COL_MISSING`: Composite PK column doesn't exist
- `COMPOSITE_PK_MISMATCH`: Composite PK doesn't match table's primary_key
- `FD_TABLE_MISSING`: FD constraint references unknown table
- `FD_LHS_COL_MISSING`: FD LHS column doesn't exist
- `FD_RHS_COL_MISSING`: FD RHS column doesn't exist
- `IMPLICATION_TABLE_MISSING`: Implication constraint references unknown table

**Usage in Repair Loop**:
```python
from nl2data.ir.validators import collect_issues, validate_logical_ir

validators = [validate_logical_ir]
issues = collect_issues(validators, board)
if issues:
    # Pass to agent's _repair() method
    board = agent._repair(board, issues)
```

**Note**: 
- `unique` constraint is not validated (exists in schema but not enforced during generation)
- Functional dependencies are now tracked and validated via ConstraintIR

### CSV Writing (`generation/engine/writer.py`)

**Streaming Writer**:

```python
def write_csv_stream(df_iter: Iterator[pd.DataFrame], path: Path) -> None:
    """
    Write a stream of DataFrames to a CSV file.
    
    Handles:
    - Header writing (only on first chunk)
    - Append mode for subsequent chunks
    - Chunk counting for logging
    """
    header_written = False
    chunk_count = 0
    
    for df in df_iter:
        df.to_csv(
            path,
            index=False,
            header=not header_written,  # Header only on first chunk
            mode="w" if not header_written else "a",  # Append after first
        )
        header_written = True
        chunk_count += 1
        logger.debug(f"Wrote chunk {chunk_count} to {path}")
    
    logger.info(f"Completed writing {chunk_count} chunks to {path}")

def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a single DataFrame to CSV (for dimension tables)."""
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(df)} rows to {path}")
```

**Key Features**:
- **Streaming support**: Handles large fact tables efficiently
- **Header management**: Writes header only once
- **Logging**: Tracks chunk progress
- **Memory efficient**: Processes chunks one at a time

### Type Enforcement (`generation/type_enforcement.py`)

**Purpose**: Centralized type enforcement for column generation.

**Key Functions**:
- `enforce_column_type(arr, sql_type, col_name, rng) -> np.ndarray`: Enforces SQL type on sampled arrays
- `_extract_datetime_to_int(arr, col_name) -> np.ndarray`: Extracts integer values from datetime arrays
- `_extract_datetime_to_float(arr, col_name, rng) -> np.ndarray`: Extracts float values from datetime arrays with semantic transformations

**Benefits**:
- **Modularity**: Separates type enforcement logic from generation logic
- **Reusability**: Used by both dimension and fact generators
- **Maintainability**: Single place to update type conversion rules

### Column Sampling Utilities (`generation/column_sampling.py`)

**Purpose**: Common column sampling utilities.

**Key Functions**:
- `sample_fallback_column(col, n, rng) -> np.ndarray`: SQL type-based fallback sampling
- `sample_primary_key_column(col, dist, n, rng) -> np.ndarray`: Primary key sampling with uniqueness guarantee

**Benefits**:
- **Code Reuse**: Eliminates duplication between dimension and fact generators
- **Consistency**: Ensures uniform fallback behavior across generators

### IR Helpers (`generation/ir_helpers.py`)

**Purpose**: Utilities for extracting information from DatasetIR.

**Key Functions**:
- `build_distribution_map(ir) -> Dict[Tuple[str, str], Distribution]`: Builds distribution map from IR
- `build_provider_map(ir) -> Dict[Tuple[str, str], ProviderRef]`: Builds provider map from IR

**Benefits**:
- **Code Reuse**: Eliminates duplicate map-building logic
- **Consistency**: Ensures same map structure across generators

### Uniqueness Enforcement (`generation/uniqueness.py`)

**Purpose**: Utilities for enforcing uniqueness constraints on columns.

**Key Functions**:
- `is_category_column(col_name) -> bool`: Detects category/type columns
- `is_person_name_column(col_name) -> bool`: Detects person name columns
- `enforce_unique_categorical_column(df, col, dist, n, rng) -> Tuple[DataFrame, int]`: Enforces uniqueness on categorical columns
- `enforce_unique_non_text_column(df, col, n) -> Tuple[DataFrame, int]`: Enforces uniqueness on non-text columns

**Benefits**:
- **Modularity**: Separates uniqueness logic from generation logic
- **Reusability**: Can be used across different generation contexts
- **Maintainability**: Single place to update uniqueness patterns

### Generation Constants (`generation/constants.py`)

**Purpose**: Centralized constants for data generation and agent configuration.

**Key Constants**:

**Data Generation**:
- `DEFAULT_DIMENSION_ROWS = 100_000`: Default row count for dimension tables
- `DEFAULT_FACT_ROWS = 1_000_000`: Default row count for fact tables
- `DEFAULT_INT_RANGE = (1, 1_000_000)`: Default integer range for fallback sampling
- `DEFAULT_DATE_RANGE_DAYS = 365 * 3`: Default date range (3 years)
- `PROGRESS_LOG_INTERVAL_SECONDS = 30`: Progress logging interval
- `LARGE_TABLE_THRESHOLD = 1_000_000`: Threshold for "large" tables
- Rush hour and semantic transformation constants

**Agent Configuration**:
- `AGENT_MAX_RETRIES = 2`: Standard retry count for agent LLM calls
- `LARGE_SCHEMA_COLUMN_THRESHOLD = 30`: Threshold for "large" schemas (column count)
- `SCHEMA_COVERAGE_THRESHOLD = 0.8`: 80% coverage threshold for large schemas

**Error Message Truncation**:
- `ERROR_MESSAGE_TRUNCATE_LENGTH = 200`: Error message truncation length
- `DEBUG_DATA_TRUNCATE_LENGTH = 1000`: Debug log truncation length
- `VALIDATION_DATA_TRUNCATE_LENGTH = 2000`: Validation error truncation length

**Benefits**:
- **Maintainability**: Single place to update default values
- **Consistency**: Ensures same defaults across codebase
- **Configurability**: Easy to adjust thresholds and ranges
- **Code Quality**: Eliminates magic numbers throughout the codebase

### Retry Utilities (`agents/tools/retry.py`)

**Purpose**: Retry logic with exponential backoff for LLM API calls.

**Key Functions**:
- `is_transient_error(error) -> bool`: Checks if error is transient
- `retry_with_backoff(func, max_retries, base_delay, timeout_errors, operation_name) -> T`: Retries function with exponential backoff

**Benefits**:
- **Code Reuse**: Eliminates duplicate retry logic in LLM clients
- **Consistency**: Uniform retry behavior across providers
- **Maintainability**: Single place to update retry logic

### Error Handling Utilities (`agents/tools/error_handling.py`)

**Purpose**: Standardized error handling for agent operations.

**Key Functions**:
- `handle_agent_error(agent_name, operation, error, raise_on_json_error) -> None`: Standardized error handling
- `safe_extract_json(text, agent_name) -> dict`: Safely extract JSON with error handling

**Benefits**:
- **Code Reuse**: Eliminates duplicate error handling in agents
- **Consistency**: Uniform error messages across agents
- **Maintainability**: Single place to update error handling

### Agent Retry Utility (`agents/tools/agent_retry.py`)

**Purpose**: Common retry logic for agent LLM calls with JSON parsing and IR validation.

**Key Function**:
- `call_llm_with_retry(messages, ir_model, max_retries, pre_process, post_process, custom_validation) -> T`: Unified retry logic for all agents

**Features**:
- Generic retry logic for LLM calls with JSON parsing
- IR model validation with Pydantic
- Configurable pre/post-processing hooks
- Custom validation support
- Uses constants from `generation.constants` for configuration

**Benefits**:
- **Code Reduction**: Eliminates ~200-300 lines of duplicated code across 5 agent files
- **Consistency**: Uniform error handling and retry behavior across all agents
- **Maintainability**: Single place to update retry logic
- **Type Safety**: Generic type parameter ensures correct IR model types
- **Flexibility**: Supports agent-specific pre/post-processing and custom validation

**Usage Example**:
```python
from nl2data.agents.tools.agent_retry import call_llm_with_retry
from nl2data.ir.logical import LogicalIR

# Simple usage
board.logical_ir = call_llm_with_retry(messages, LogicalIR)

# With pre-processing (fix common LLM mistakes)
board.logical_ir = call_llm_with_retry(
    messages,
    LogicalIR,
    pre_process=_fix_common_llm_mistakes
)

# With custom validation
def validate_table_names(data: dict) -> None:
    # Custom validation logic
    if invalid_tables:
        raise ValueError(f"Invalid tables: {invalid_tables}")

board.generation_ir = call_llm_with_retry(
    messages,
    GenerationIR,
    custom_validation=validate_table_names
)
```

---

## Derived Columns and DSL

This section provides comprehensive documentation on the derived column system and Domain-Specific Language (DSL) for expression evaluation.

### Overview

Derived columns are computed from other columns using expressions written in a Python-like DSL. They enable:
- **Complex Calculations**: Arithmetic operations, conditional logic, and chained computations
- **Temporal Analysis**: Date/time extraction and time arithmetic
- **Dimension Lookups**: Reference attributes from joined dimension tables
- **Business Logic**: Implement domain-specific rules and constraints

### DSL Reference

See the [Derived Column System](#34-derived-column-system) section in Implementation Details for the complete DSL reference, including:
- All available functions
- Expression examples
- Dimension lookup mechanism
- Validation details

### Best Practices

1. **Use Descriptive Column Names**: Derived columns should have clear names indicating their purpose
2. **Document Complex Expressions**: For complex expressions, consider adding comments in the IR
3. **Test Edge Cases**: Validate expressions with boundary values (midnight, year boundaries, etc.)
4. **Leverage Dimension Joins**: Use dimension lookups for attributes that vary by entity type
5. **Type Safety**: Always specify `dtype` for derived columns to ensure correct type coercion

### Common Patterns

**Peak Hour Detection**:
```python
"where((hour(timestamp) >= 7 and hour(timestamp) <= 9) or (hour(timestamp) >= 16 and hour(timestamp) <= 18), 1, 0)"
```

**Weekend Flag**:
```python
"where(day_of_week(timestamp) >= 5, 1, 0)"
```

**Chained Monetary Calculations**:
```python
"gross_fare - discount_amount + tax_amount"
```

**Conditional Rebates**:
```python
"where(consumption_kwh > threshold, cost_before_rebate * rebate_rate, 0)"
```

**Dimension-Based Pricing**:
```python
"dynamic_price_per_kwh / baseline_price_per_kwh"  # baseline from dimension
```

For more examples and detailed documentation, see `Improvement.md`.

## Usage Examples

### CLI Usage

**End-to-End Pipeline**:
```bash
python scripts/nl2data.py end2end description.txt output/
```

**Step-by-Step**:
```bash
# NL → IR
python scripts/nl2data.py nl2ir description.txt dataset_ir.json

# IR → Data
python scripts/nl2data.py generate dataset_ir.json output/

# Evaluate
python scripts/nl2data.py evaluate-data dataset_ir.json output/ report.json
```

### Programmatic Usage

**Full Pipeline**:
```python
from nl2data.agents.base import Blackboard
from nl2data.agents.orchestrator import Orchestrator
from nl2data.utils.agent_factory import create_agent_list
from nl2data.utils.ir_io import save_ir_to_json
from nl2data.generation.engine.pipeline import generate_from_ir

# Create agents using utility function
nl_description = "Generate a retail sales dataset..."
agents = create_agent_list(nl_description)

# Execute pipeline
board = Orchestrator(agents).execute(Blackboard())
ir = board.dataset_ir

# Save IR (optional)
save_ir_to_json(ir, Path("output/dataset_ir.json"))

# Generate data
generate_from_ir(ir, Path("output"), seed=7, chunk_rows=1_000_000)
```

**Streamlit UI** (`ui_streamlit/app.py`):

The project includes a web-based UI built with Streamlit:

```python
def main():
    """Main Streamlit application."""
    st.title("NL → Synthetic Relational Data Generator")
    
    # NL input text area
    nl_text = st.text_area("Dataset description", value=example, height=220)
    
    if st.button("Run pipeline"):
        with st.spinner("Running multi-agent pipeline..."):
            ir, steps, out_dir, table_names = run_pipeline(
                nl_description=nl_text,
                output_root=OUTPUT_ROOT,
            )
            st.session_state["ir"] = ir
            st.session_state["steps"] = steps
    
    # Display pipeline steps with status
    if st.session_state["steps"]:
        show_step_log(st.session_state["steps"])
    
    # Display schema summary
    if st.session_state["ir"]:
        for table_name, table in ir.logical.tables.items():
            with st.expander(f"📊 {table_name}"):
                # Show table details
    
    # Download generated CSVs
    if st.session_state["tables"]:
        for table in st.session_state["tables"]:
            st.download_button(f"Download {table}.csv", ...)
```

**Features**:
- **Interactive Input**: Text area for natural language descriptions
- **Step Tracking**: Visual progress of pipeline execution
- **Schema Display**: Expandable view of generated schema
- **CSV Downloads**: Direct download of generated files
- **Error Display**: Error messages with traceback in expandable sections

**Pipeline Runner** (`ui_streamlit/pipeline_runner.py`):

```python
def run_pipeline(
    nl_description: str,
    output_root: Path,
) -> Tuple[DatasetIR, List[StepLog], Path, List[str]]:
    """
    Run the full NL -> IR -> Data pipeline and collect step logs.
    
    Returns:
        (DatasetIR, list of StepLog, output directory, list of table names)
    """
    steps: List[StepLog] = []
    
    # Execute agents sequentially
    for name, agent in agent_sequence:
        steps.append(StepLog(name=name, status="running"))
        try:
            board = agent.run(board)
            steps[-1].status = "done"
        except Exception as e:
            steps[-1].status = "error"
            steps[-1].message = str(e)
            raise
    
    # Generation step
    steps.append(StepLog(name="generation", status="running"))
    generate_from_ir(ir, out_dir, seed=settings.seed, chunk_rows=settings.chunk_rows)
    steps[-1].status = "done"
    
    return ir, steps, out_dir, table_names
```

**Step Logging** (`ui_streamlit/step_models.py`):

```python
StepName = Literal[
    "manager",
    "conceptual_designer",
    "logical_designer",
    "dist_engineer",
    "workload_designer",
    "qa_compiler",
    "generation",
]

StepStatus = Literal["pending", "running", "done", "error"]

@dataclass
class StepLog:
    """Log entry for a pipeline step."""
    name: StepName  # "manager", "conceptual_designer", etc.
    status: StepStatus  # "pending", "running", "done", "error"
    message: Optional[str] = None  # Error message if status="error"
    summary: Optional[str] = None  # Summary text for display
```

**Usage**:
```bash
cd ui_streamlit
streamlit run app.py
```

### Example Natural Language Input

```
Generate a retail sales dataset with one fact table and three dimension 
tables. The fact table should have 5 million rows. Product sales should 
follow a Zipf distribution. Customer purchases should be seasonal with 
peaks in December. The data should stress test group-by and join performance.
```

### Generated IR Structure

**DatasetIR JSON**:
```json
{
  "logical": {
    "tables": {
      "fact_sales": {
        "name": "fact_sales",
        "kind": "fact",
        "row_count": 5000000,
        "columns": [
          {"name": "sale_id", "sql_type": "INT64", "role": "primary_key"},
          {"name": "product_id", "sql_type": "INT64", "role": "foreign_key", 
           "references": "dim_product.product_id"},
          {"name": "customer_id", "sql_type": "INT64", "role": "foreign_key",
           "references": "dim_customer.customer_id"},
          {"name": "sale_date", "sql_type": "DATETIME"},
          {"name": "amount", "sql_type": "FLOAT64"}
        ],
        "primary_key": ["sale_id"],
        "foreign_keys": [
          {"column": "product_id", "ref_table": "dim_product", "ref_column": "product_id"},
          {"column": "customer_id", "ref_table": "dim_customer", "ref_column": "customer_id"}
        ]
      },
      "dim_product": {
        "name": "dim_product",
        "kind": "dimension",
        "row_count": 10000,
        "columns": [
          {"name": "product_id", "sql_type": "INT64", "role": "primary_key"},
          {"name": "product_name", "sql_type": "TEXT"}
        ],
        "primary_key": ["product_id"]
      }
    }
  },
  "generation": {
    "columns": [
      {
        "table": "fact_sales",
        "column": "product_id",
        "distribution": {
          "kind": "zipf",
          "s": 1.2,
          "n": 10000
        }
      },
      {
        "table": "fact_sales",
        "column": "sale_date",
        "distribution": {
          "kind": "seasonal",
          "granularity": "month",
          "weights": {
            "December": 0.15,
            "January": 0.08,
            ...
          }
        }
      }
    ]
  },
  "workload": {
    "targets": [
      {
        "type": "group_by",
        "query_hint": "GROUP BY product_id",
        "expected_skew": "high"
      },
      {
        "type": "join",
        "join_graph": ["fact_sales", "dim_product", "dim_customer"]
      }
    ]
  }
}
```

---

## Prompt System

### Prompt File Structure

Prompts are stored as text files in `prompts/roles/` with two files per agent:

1. **`{agent}_system.txt`**: System prompt (role definition, instructions)
2. **`{agent}_user.txt`**: User prompt template (with placeholders)

### Complete Agent Prompts

#### Manager Agent Prompts

**`manager_system.txt`**:
```
You are the Manager agent in a multi-agent system for generating synthetic relational datasets.

Your task is to extract a structured RequirementIR from the user's natural language description of a synthetic relational dataset.

You must return a JSON object matching this structure:

{
  "domain": string | null,
  "narrative": string,
  "tables_hint": string | null,
  "scale": [{"table": string | null, "row_count": int | null}, ...],
  "distributions": [{"target": "table.column", "family": "zipf" | "seasonal" | "categorical" | "numeric", "params": {...}}, ...],
  "nonfunctional_goals": [string, ...]
}

Key guidelines:
- Extract the domain/business area (e.g., "retail", "healthcare", "finance")
- Preserve the narrative description
- Identify hints about table names or structure
- Extract scale hints (row counts for specific tables)
- Identify distribution hints (e.g., "product_id follows Zipf distribution", "sales are seasonal")
- For distributions.params: use numeric values for zipf (e.g., {"s": 1.2}), use month names as strings for seasonal (e.g., {"peak_month": "December"}), use arrays for categorical
- Capture non-functional goals (e.g., "support real-time queries", "handle 50M rows")

IMPORTANT: The "params" field in distributions should contain appropriate types:
- For "zipf": use numbers like {"s": 1.2, "n": 1000}
- For "seasonal": use strings for month names like {"peak_month": "December"} or numeric weights
- For "categorical": use arrays like {"values": ["A", "B", "C"]}
- For "numeric": use numbers like {"mean": 100, "std": 10}

Return ONLY the JSON object, no explanations or markdown formatting.
```

**`manager_user.txt`**:
```
Here is the user's natural language description:

{NARRATIVE}

Extract the RequirementIR JSON as described in the system prompt.
```

#### Conceptual Designer Agent Prompts

**`conceptual_system.txt`**:
```
You are the Conceptual Designer agent in a multi-agent system for generating synthetic relational datasets.

Your task is to design a conceptual ER-style model from a RequirementIR specification.

You must return a JSON object matching this structure:

{
  "entities": [
    {
      "name": "EntityName",
      "attributes": [
        {
          "name": "attribute_name",
          "kind": "identifier" | "numeric" | "categorical" | "text" | "datetime" | "boolean"
        }
      ]
    }
  ],
  "relationships": [
    {
      "name": "relationship_name",
      "participants": ["EntityA", "EntityB", ...],
      "cardinality": "many_to_one" | "one_to_many" | "many_to_many"
    }
  ]
}

Key guidelines:
- Identify entities from the requirement narrative
- Assign appropriate attribute kinds (identifier for IDs, numeric for numbers, etc.)
- Identify relationships between entities
- Use appropriate cardinalities (many_to_one for FK relationships, many_to_many for junction tables)
- Focus on the conceptual model, not implementation details

Return ONLY the JSON object, no explanations or markdown formatting.
```

**`conceptual_user.txt`**:
```
RequirementIR:

{REQUIREMENT_JSON}

Design a conceptual ER model as JSON following the structure described in the system prompt.
```

#### Logical Designer Agent Prompts

**`logical_system.txt`**:
```
You are the Logical Designer agent in a multi-agent system for generating synthetic relational datasets.

Your task is to design a logical relational schema from a ConceptualIR model.

You must return a JSON object matching this structure:

{
  "tables": {
    "table_name": {
      "name": "table_name",
      "kind": "fact" | "dimension" | null,
      "row_count": int | null,
      "columns": [
        {
          "name": "column_name",
          "sql_type": "INT32" | "INT64" | "FLOAT32" | "FLOAT64" | "TEXT" | "DATE" | "DATETIME" | "BOOL",
          "nullable": bool,
          "unique": bool,
          "role": "primary_key" | "foreign_key" | "measure" | "attribute" | null,
          "references": "ref_table.ref_column" | null
        }
      ],
      "primary_key": ["column_name", ...],
      "foreign_keys": [
        {
          "column": "fk_column",
          "ref_table": "ref_table_name",
          "ref_column": "ref_column_name"
        }
      ]
    }
  }
}

Key guidelines:
- Convert entities to tables
- Convert attributes to columns with appropriate SQL types
- Identify fact tables (large, transactional) vs dimension tables (small, descriptive)
- Set primary keys (typically single-column IDs)
- Set foreign keys based on relationships
- Use appropriate SQL types (INT32/INT64 for integers, TEXT for strings, DATE/DATETIME for dates)
- Set row_count hints from RequirementIR scale hints
- IMPORTANT: Mark columns as "unique": true when they should have unique values:
  * Primary key columns (always unique)
  * Name/identifier columns in dimension tables (e.g., "type_name", "zone_name", "category_name")
  * Any column that logically represents a distinct identifier or category name

Return ONLY the JSON object, no explanations or markdown formatting.
```

**`logical_user.txt`**:
```
ConceptualIR:

{CONCEPTUAL_JSON}

RequirementIR (for hints):

{REQUIREMENT_JSON}

Design a logical relational schema as JSON following the structure described in the system prompt.
```

#### Distribution Engineer Agent Prompts

**`dist_system.txt`**:
```
You are the Distribution Engineer agent in a multi-agent system for generating synthetic relational datasets.

Your task is to design generation specifications for columns based on RequirementIR distribution hints and logical schema semantics.

You must return a JSON object matching this structure:

{
  "columns": [
    {
      "table": "table_name",
      "column": "column_name",
      "distribution": {
        "kind": "uniform" | "normal" | "zipf" | "seasonal" | "categorical" | "derived",
        ... (distribution-specific parameters)
      }
    }
  ]
}

Distribution types (EXACT STRUCTURE REQUIRED):

1. Uniform: {"kind": "uniform", "low": NUMBER, "high": NUMBER}
   - Use for numeric ranges. low and high MUST be numbers, not dates or strings.
   - Example: {"kind": "uniform", "low": 0.0, "high": 100.0}

2. Normal: {"kind": "normal", "mean": NUMBER, "std": NUMBER}
   - Example: {"kind": "normal", "mean": 50.0, "std": 10.0}

3. Zipf: {"kind": "zipf", "s": NUMBER, "n": INTEGER | null}
   - s = exponent (typically 1.2-2.0), n = domain size
   - Example: {"kind": "zipf", "s": 1.2, "n": 1000}

4. Seasonal: {"kind": "seasonal", "granularity": "month" | "week", "weights": {"January": 0.1, "February": 0.08, ...}}
   - Use for date columns with seasonal patterns
   - weights must be a dictionary mapping month/week names to numbers
   - Example: {"kind": "seasonal", "granularity": "month", "weights": {"December": 0.15, "November": 0.12, "January": 0.08, ...}}

5. Categorical: {"kind": "categorical", "domain": {"values": ["string1", "string2", ...], "probs": [0.1, 0.2, ...] | null}}
   - values MUST be an array of STRINGS (convert booleans/numbers to strings)
   - Example: {"kind": "categorical", "domain": {"values": ["true", "false"], "probs": [0.3, 0.7]}}
   - Example: {"kind": "categorical", "domain": {"values": ["2021", "2022", "2023"], "probs": null}}

6. Derived: {"kind": "derived", "expression": "expression_string"}
   - Example: {"kind": "derived", "expression": "price * quantity"}

CRITICAL RULES:
- For DATE columns: use "seasonal" distribution, NOT "uniform" with date strings
- For BOOLEAN columns: use "categorical" with values ["true", "false"] as STRINGS
- For INTEGER categoricals: convert to strings in categorical values (e.g., [2021, 2022] → ["2021", "2022"])
- uniform.low and uniform.high MUST be numbers, never dates or strings
- categorical.domain.values MUST be strings (convert bool/int to string)

Key guidelines:
- Use RequirementIR.distributions hints when available
- For foreign keys, typically use Zipf distribution (skewed)
- For date columns, use seasonal distribution if mentioned in requirements
- For categorical columns, use categorical distribution with string values
- For numeric measures, use normal or uniform
- For dimension IDs, use uniform or categorical
- Default to reasonable distributions if not specified

Return ONLY the JSON object, no explanations or markdown formatting.
```

**`dist_user.txt`**:
```
LogicalIR:

{LOGICAL_JSON}

RequirementIR (for distribution hints):

{REQUIREMENT_JSON}

Design generation specifications for all columns as JSON following the structure described in the system prompt.
```

#### Workload Designer Agent Prompts

**`workload_system.txt`**:
```
You are the Workload Designer agent in a multi-agent system for generating synthetic relational datasets.

Your task is to design workload specifications for evaluating the generated dataset.

You must return a JSON object matching this structure:

{
  "targets": [
    {
      "type": "group_by" | "join" | "filter",
      "query_hint": "SQL query string" | null,
      "expected_skew": "low" | "medium" | "high" | null,
      "join_graph": ["table1", "table2", ...] | null,
      "selectivity_hint": "low" | "medium" | "high" | null
    }
  ]
}

Key guidelines:
- Design queries that stress test the dataset
- Include group_by queries on fact tables (especially on FK columns for skew)
- Include join queries connecting fact and dimension tables
- Include filter queries with various selectivities
- Set expected_skew based on distribution hints (high for Zipf, low for uniform)
- Use join_graph to specify which tables should be joined
- Generate 3-5 workload targets covering different query types

Return ONLY the JSON object, no explanations or markdown formatting.
```

**`workload_user.txt`**:
```
LogicalIR:

{LOGICAL_JSON}

RequirementIR (for workload hints):

{REQUIREMENT_JSON}

Design workload specifications as JSON following the structure described in the system prompt.
```

#### QA Compiler Agent

**Note**: The QA Compiler agent does not use LLM prompts. It performs programmatic validation and compilation of the DatasetIR.

### Prompt Loading (`prompts/loader.py`)

```python
def load_prompt(path: str) -> str:
    """Load prompt file from prompts/ directory."""
    full_path = PROMPTS_DIR / path
    return full_path.read_text(encoding="utf-8")

def render_prompt(template: str, **kwargs) -> str:
    """
    Render template with placeholders.
    
    Uses str.format() for simple templating.
    Placeholders: {PLACEHOLDER_NAME}
    """
    return template.format(**kwargs)
```

### Prompt Usage in Agents

```python
# Load prompts
sys_tmpl = load_prompt("roles/manager_system.txt")
usr_tmpl = load_prompt("roles/manager_user.txt")

# Render user prompt with data
user_content = render_prompt(
    usr_tmpl,
    NARRATIVE=self.nl_request,
    REQUIREMENT_JSON=board.requirement_ir.model_dump_json(indent=2),
)

# Send to LLM
messages = [
    {"role": "system", "content": sys_tmpl},
    {"role": "user", "content": user_content},
]
```

**Benefits**:
- **Easy iteration**: Edit prompts without code changes
- **Version control**: Track prompt changes in git
- **A/B testing**: Try different prompt variations
- **Non-developer friendly**: Domain experts can edit prompts

---

## Error Handling and Robustness

### LLM Response Handling

1. **JSON Extraction**: Multiple strategies to extract JSON from LLM output
2. **Validation**: Pydantic validation catches malformed IRs
3. **Error Messages**: Clear error messages with context
4. **Logging**: Debug information for troubleshooting

### Data Generation Robustness

1. **Fallback Distributions**: If no distribution specified, uses SQL type-based fallbacks
2. **Dependency Validation**: Detects circular dependencies in derived columns
3. **Type Coercion**: Handles type mismatches (e.g., categorical values to strings)
4. **Memory Management**: Streaming generation prevents OOM errors
5. **Error Resilience**: Individual table generation failures don't stop the entire pipeline
   - Dimension and fact table generation wrapped in try-except blocks
   - Failed tables are logged and tracked separately
   - Pipeline completes successfully even if some tables fail
   - Summary report includes both successful and failed tables

### Evaluation Robustness

1. **Missing Data Handling**: Gracefully handles missing tables/columns
2. **Statistical Test Failures**: Continues evaluation even if some tests fail
3. **Query Error Handling**: Catches SQL errors and reports them

---

## Known Limitations and Future Work

### Current Limitations

1. **Unique Constraints**: The `unique` flag in `ColumnSpec` is not enforced during generation. This is a known gap.

2. **Functional Dependencies**: The system does not track functional dependencies (FDs) beyond those implied by primary keys.

3. **Derived Column Expressions**: Limited to a whitelist of functions. Complex expressions may not be supported.

4. **Error Handling**: Some edge cases in LLM responses may not be handled gracefully.

5. **Evaluation**: Some statistical tests are simplified (e.g., seasonal validation).

### Future Enhancements

1. **Uniqueness Enforcement**: Add uniqueness constraint checking and enforcement during generation.

2. **Functional Dependencies**: Add FD tracking to LogicalIR and enforce during generation.

3. **More Distribution Types**: Add support for more complex distributions (e.g., multivariate).

4. **Parallel Generation**: Parallelize dimension table generation and derived column computation.

5. **Incremental Generation**: Support adding rows to existing datasets.

6. **Better Evaluation**: More sophisticated statistical tests and workload analysis.

7. **UI Improvements**: Better visualization of IRs, generation progress, and evaluation results.

---

## Conclusion

NL2Data is a comprehensive system for generating synthetic relational datasets from natural language descriptions. Its multi-agent architecture, IR-centric design, and streaming generation capabilities make it suitable for both small-scale prototyping and large-scale data generation. The prompt-file driven approach enables easy iteration, and the comprehensive evaluation framework ensures data quality.

### Key Strengths

1. **Modularity**: Clear separation of concerns with specialized agents
2. **Type Safety**: Pydantic models ensure correctness at each stage
3. **Scalability**: Streaming generation handles millions of rows
4. **Flexibility**: Multiple LLM providers, easy prompt iteration
5. **Robustness**: Comprehensive error handling and validation
6. **Extensibility**: Easy to add new distributions, agents, or evaluation metrics
7. **Code Quality**: Centralized utilities reduce duplication, constants eliminate magic numbers, and error resilience ensures partial success scenarios

### Design Philosophy

The system demonstrates how LLMs can be effectively used in a structured pipeline, with each agent handling a specific transformation stage. The use of Pydantic for IR models ensures type safety and validation throughout the pipeline. The prompt-file driven approach allows non-developers to iterate on prompts, making the system accessible to domain experts.

### Use Cases

- **Database Testing**: Generate test data with realistic distributions
- **Performance Benchmarking**: Create datasets for query performance testing
- **Schema Design**: Prototype database schemas from natural language
- **Data Science**: Generate synthetic datasets for algorithm development
- **Education**: Teach database concepts with realistic examples

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Project**: NL2Data - Natural Language to Synthetic Relational Data Generation System

