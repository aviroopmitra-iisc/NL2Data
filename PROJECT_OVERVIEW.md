# NL2Data Project - Comprehensive Overview

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Realistic Datasets Framework](#realistic-datasets-framework)
5. [Core Components](#core-components)
   - [Intermediate Representations (IRs)](#1-intermediate-representations-irs)
   - [Multi-Agent System](#2-multi-agent-system)
   - [Data Generation Engine](#3-data-generation-engine)
   - [Evaluation Framework](#4-evaluation-framework)
6. [Data Flow](#data-flow)
7. [Key Design Decisions](#key-design-decisions)
8. [Implementation Details](#implementation-details)
9. [Usage Examples](#usage-examples)

---

## Executive Summary

**NL2Data** is a multi-agent system that converts natural language descriptions into synthetic relational datasets. The system uses Large Language Models (LLMs) to progressively refine specifications through multiple Intermediate Representations (IRs), ultimately generating realistic CSV data with proper schema, distributions, and referential integrity.

### Key Capabilities
- **Natural Language Processing**: Converts free-form text descriptions into structured database schemas
- **Multi-Agent Pipeline**: Specialized agents handle different aspects (conceptual design, logical schema, distributions, workloads)
- **Realistic Data Generation**: Supports comprehensive distribution families including Uniform, Normal, Lognormal, Pareto, Poisson, Exponential, Mixture, Zipf, Seasonal, Categorical, Derived columns, and Window functions
- **Window Functions**: Rolling aggregations, lag/lead operations with time-based and row-based windows
- **Event System**: Global events that causally affect data generation (incidents, disruptions, campaigns)
- **Advanced DSL**: Extended expression language with type casting, distributions, string operations, and helper functions
- **Scalable Generation**: Handles millions of rows with streaming/chunked generation
- **Comprehensive Evaluation**: Validates schema correctness, statistical properties, and workload performance
- **Constraint System**: Functional dependencies, implications, and composite primary keys
- **Self-Healing Pipeline**: Automatic repair loops with QA feedback (✅ Integrated in Orchestrator)
- **Quality Metrics Tracking**: Comprehensive metrics collection for monitoring and debugging (✅ Integrated)
- **Constraint Enforcement**: Automatic enforcement of FDs, implications, and nullability (✅ Integrated)
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
│   │   │   ├── roles/               # Agent-specific prompts
│   │   │   │   ├── manager_system.txt
│   │   │   │   ├── manager_user.txt
│   │   │   │   ├── conceptual_system.txt
│   │   │   │   ├── conceptual_user.txt
│   │   │   │   ├── logical_system.txt
│   │   │   │   ├── logical_user.txt
│   │   │   │   ├── dist_system.txt
│   │   │   │   ├── dist_user.txt
│   │   │   │   ├── workload_system.txt
│   │   │   │   ├── workload_user.txt
│   │   │   │   ├── qa_system.txt
│   │   │   │   └── qa_user.txt
│   │   │   └── repair/              # Repair prompt templates
│   │   │       └── categorical_values.txt  # Categorical value repair prompts
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
│   │   │   │   # derived.py removed (deprecated - use derived expression engine)
│   │   │   └── engine/              # Generation pipeline
│   │   │       ├── __init__.py
│   │   │       ├── pipeline.py      # Main generation pipeline
│   │   │       ├── dim_generator.py  # Dimension table generator
│   │   │       ├── fact_generator.py # Fact table generator (streaming)
│   │   │       └── writer.py        # CSV writer utilities
│   │   │   ├── window_eval.py      # Window function evaluation
│   │   │   ├── event_eval.py       # Event effect application
│   │   │   ├── enforce.py          # Constraint enforcement
│   │   │   ├── facts.py            # Fact generation utilities
│   │   │   ├── allocator.py        # Memory-safe FK allocation
│   │   │   ├── type_enforcement.py # Type enforcement utilities
│   │   │   ├── uniqueness.py      # Uniqueness enforcement utilities
│   │   │   ├── ir_helpers.py      # IR extraction utilities
│   │   │   └── constants.py       # Generation constants
│   │   │
│   │   ├── monitoring/             # Quality metrics tracking
│   │   │   ├── __init__.py
│   │   │   └── quality_metrics.py  # Agent and query metrics collection
│   │   │
│   │   ├── evaluation/              # Evaluation framework (restructured)
│   │   │   ├── __init__.py
│   │   │   ├── config.py            # EvaluationConfig, MultiTableEvalConfig
│   │   │   ├── evaluators/          # Evaluation functions
│   │   │   │   ├── __init__.py
│   │   │   │   ├── single_table.py  # Single-table evaluator
│   │   │   │   └── multi_table.py   # Multi-table evaluator
│   │   │   ├── models/              # Report models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── single_table.py  # Single-table models
│   │   │   │   └── multi_table.py   # Multi-table models
│   │   │   ├── metrics/             # Metric computation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── schema/          # Schema metrics (coverage, validation)
│   │   │   │   ├── table/           # Table metrics (marginals, correlations)
│   │   │   │   └── relational/      # Relational metrics (FK, degrees, joins)
│   │   │   ├── matching/             # Schema matching
│   │   │   │   ├── __init__.py
│   │   │   │   ├── table_matcher.py  # Table matching
│   │   │   │   ├── column_matcher.py # Column matching
│   │   │   │   ├── similarity.py     # Similarity utilities
│   │   │   │   └── enhanced_matcher.py # Enhanced matching algorithm
│   │   │   ├── quality/              # Quality evaluation (SD Metrics)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── table_quality.py  # Single-table quality
│   │   │   │   └── multi_table_quality.py # Multi-table quality
│   │   │   ├── execution/            # Execution utilities
│   │   │   │   ├── __init__.py
│   │   │   │   ├── stats.py         # Statistical tests
│   │   │   │   └── workload.py      # Workload execution
│   │   │   ├── aggregation/         # Score aggregation
│   │   │   │   ├── __init__.py
│   │   │   │   ├── schema_score.py  # Schema score aggregation
│   │   │   │   ├── structure_score.py # Structure score aggregation
│   │   │   │   └── utility_score.py # Utility score aggregation
│   │   │   └── utils/                # Evaluation utilities
│   │   │       ├── __init__.py
│   │   │       ├── normalization.py  # Name normalization
│   │   │       └── fd_utils.py       # FD counting and signatures
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
│       ├── test_ir_models.py        # IR model tests
│       ├── test_derived_dsl.py      # Derived DSL tests
│       └── test_derived_regression.py  # Regression tests
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
├── test_phase1_evaluation.py        # Phase 1 IR evaluation script
├── run_all_pipelines.py             # Batch pipeline runner for all descriptions
├── prompt_generator_format.py      # Prompt generation utilities for schema design
├── test_utils/                      # Test utilities module
│   ├── __init__.py
│   ├── query_parser.py              # Query parsing utilities
│   ├── cache_manager.py            # Data caching/existence checks
│   ├── report_formatter.py         # Evaluation report formatting
│   └── test_helpers.py             # Test helper functions
├── example_queries.json             # Example queries in JSON format (replaces example queries.txt)
├── EVALUATION_FRAMEWORK_PLAN.md    # Detailed evaluation framework implementation plan
├── schema_evaluation_results.json   # Schema evaluation results
├── Improvement.md                   # Improvement roadmap and implementation guide
├── Instructions.md                  # Detailed instructions
├── UI.md                            # UI documentation
├── realistic_datasets/              # Realistic dataset integration framework
│   ├── analyze_datasets.py         # Dataset statistics analyzer
│   ├── generate_descriptions.py    # NL description generator from statistics
│   ├── run_pipeline_all.py          # Batch pipeline runner
│   ├── evaluate_all_datasets.py    # Batch evaluation runner
│   ├── generate_all_irs.py         # Batch IR generator
│   ├── prompt_template.txt          # Template for NL description generation
│   │
│   ├── data/                       # Processed datasets
│   │   ├── sakila/                 # Sakila database (MySQL sample DB)
│   │   │   ├── *.csv               # Table CSV files
│   │   │   ├── original_ir.json    # Gold standard schema
│   │   │   ├── statistics.json     # Dataset statistics
│   │   │   └── description_*.txt   # Generated NL descriptions
│   │   ├── world/                  # World database (MySQL sample DB)
│   │   │   ├── *.csv               # Table CSV files
│   │   │   ├── original_ir.json    # Gold standard schema
│   │   │   ├── statistics.json     # Dataset statistics
│   │   │   └── description_*.txt   # Generated NL descriptions
│   │   ├── openml/                 # OpenML datasets
│   │   ├── datagov/                # Data.gov datasets
│   │   ├── worldbank/              # World Bank indicators
│   │   ├── openstreetmap/          # OpenStreetMap data
│   │   └── census/                 # US Census data
│   │
│   ├── sakila/                     # Sakila database scripts
│   │   ├── create_db.py            # Combined IR creation + data extraction
│   │   ├── sakila-schema.sql       # SQL schema file
│   │   └── sakila-data.sql         # SQL data file
│   │
│   ├── world/                      # World database scripts
│   │   ├── create_db.py            # Combined IR creation + data extraction
│   │   └── world.sql               # Combined SQL schema + data file
│   │
│   ├── openml/                     # OpenML integration
│   ├── datagov/                    # Data.gov integration
│   ├── worldbank/                  # World Bank integration
│   ├── openstreetmap/              # OpenStreetMap integration
│   ├── census/                     # Census integration
│   │
│   └── statistics_to_ir/          # Statistics to GenerationIR conversion
│       ├── __init__.py             # Main entry point (create_generation_ir_from_statistics)
│       ├── fd_discovery.py         # Apriori-based FD discovery with sampling
│       ├── candidate_key_discovery.py  # Candidate key detection
│       ├── schema_updater.py       # Schema update with FDs and candidate keys
│       ├── stats_converter.py      # Statistics to GenerationIR conversion
│       ├── distribution_mapper.py  # Distribution fitting and mapping
│       ├── llm_assistant.py        # LLM client for conflict resolution
│       └── utils.py                # Utility functions (load dataframes, save FDs)
│
├── RSchema (Text2Schema)/          # RSchema evaluation framework
│   ├── annotation.jsonl            # Gold standard schema annotations
│   ├── annotation_ddl.jsonl        # DDL format annotations
│   ├── evaluate_rschema.py          # RSchema evaluation script
│   └── evaluation_results.md       # Evaluation results report
│
└── PROJECT_OVERVIEW.md              # This file
```

---

## Realistic Datasets Framework

### Overview

The `realistic_datasets/` directory contains a comprehensive framework for integrating real-world databases and datasets into the NL2Data evaluation pipeline. This framework enables:

1. **Schema Extraction**: Converting SQL schemas or CSV metadata into LogicalIR format
2. **Data Extraction**: Parsing SQL INSERT statements or CSV files into clean CSV format
3. **Statistics Generation**: Analyzing datasets to extract comprehensive statistical properties
4. **NL Description Generation**: Using LLMs to generate diverse natural language descriptions from statistics
5. **Pipeline Integration**: Running the full NL→IR→Data pipeline on real datasets for evaluation

### Directory Structure

The framework supports two directory structures:

1. **One-Level Structure**: `data/dataset/` (e.g., `data/sakila/`, `data/world/`)
2. **Two-Level Structure**: `data/source/dataset/` (e.g., `data/openml/iris/`, `data/worldbank/usa_economic/`)

Both structures are automatically detected and processed by the analysis and description generation scripts.

### Integrated Databases

#### Sakila Database

**Location**: `realistic_datasets/sakila/`

**Description**: MySQL's sample database representing a DVD rental store with 16 tables including customers, films, actors, rentals, payments, and staff.

**Files**:
- `sakila-schema.sql`: SQL CREATE TABLE statements defining the schema
- `sakila-data.sql`: SQL INSERT statements containing all data
- `create_db.py`: Combined script for IR creation and data extraction

**Tables**: 16 tables
- `actor`, `address`, `category`, `city`, `country`, `customer`
- `film`, `film_actor`, `film_category`, `inventory`, `language`
- `payment`, `rental`, `staff`, `store`

**Usage**:
```bash
cd realistic_datasets/sakila
python create_db.py
# or
python -m realistic_datasets.sakila.create_db
```

This generates:
- `data/sakila/original_ir.json`: Gold standard LogicalIR schema
- `data/sakila/*.csv`: Clean CSV files for each table

#### World Database

**Location**: `realistic_datasets/world/`

**Description**: MySQL's sample database containing world geography data with 3 tables: countries, cities, and country languages.

**Files**:
- `world.sql`: Combined SQL file with both schema and data
- `create_db.py`: Combined script for IR creation and data extraction

**Tables**: 3 tables
- `country`: 239 countries with geographic and demographic data
- `city`: 4,079 cities with population data
- `countrylanguage`: 984 language records

**Usage**:
```bash
cd realistic_datasets/world
python create_db.py
# or
python -m realistic_datasets.world.create_db
```

This generates:
- `data/world/original_ir.json`: Gold standard LogicalIR schema
- `data/world/*.csv`: Clean CSV files for each table

### Database Integration Scripts

#### `create_db.py` (Sakila & World)

**Purpose**: Unified script that combines schema parsing and data extraction into a single workflow.

**Key Features**:
1. **SQL Schema Parsing**: Extracts table definitions, columns, primary keys, and foreign keys from MySQL CREATE TABLE statements
2. **Data Extraction**: Parses INSERT statements and extracts clean CSV data
3. **Quote Handling**: Properly strips SQL quotes and handles escaped characters
4. **Comprehensive Logging**: Detailed progress logging for debugging

**Main Functions**:

1. **`mysql_type_to_sql_type()`**: Converts MySQL types to NL2Data SQLType enum
   - Maps `INT`, `BIGINT`, etc. → `INT`
   - Maps `VARCHAR`, `TEXT`, etc. → `TEXT`
   - Maps `DECIMAL`, `FLOAT`, etc. → `FLOAT`
   - Maps `DATE` → `DATE`, `DATETIME` → `DATETIME`

2. **`parse_create_table()`**: Parses CREATE TABLE statements using regex
   - Handles backtick-quoted table/column names
   - Extracts column definitions with types and nullability
   - Identifies PRIMARY KEY constraints
   - Identifies FOREIGN KEY constraints with references

3. **`create_logical_ir_from_*_sql()`**: Converts parsed schema to LogicalIR
   - Creates `TableSpec` objects for each table
   - Creates `ColumnSpec` objects with proper types
   - Creates `ForeignKeySpec` objects for relationships
   - Sets schema mode to "oltp" (Online Transaction Processing)

4. **`parse_insert_statements()`**: Extracts data from INSERT statements
   - Handles single-row and multi-row INSERT statements
   - Properly parses quoted strings (single and double quotes)
   - Handles escaped quotes (`\'`, `\"`)
   - Removes trailing semicolons and parentheses
   - Splits multi-row inserts correctly

5. **`create_csv_files()`**: Writes extracted data to CSV files
   - Uses column names from LogicalIR for headers
   - Handles mismatched column counts (truncates or pads)
   - Creates one CSV file per table

**Data Extraction Improvements**:

The data extraction was enhanced to fix issues with:
- **Quote Stripping**: Values like `'Kabul'` are now correctly extracted as `Kabul` (without quotes)
- **Trailing Syntax**: Removes trailing `);` from SQL statements
- **Escaped Characters**: Properly handles `\'` and `\"` in string values
- **Multi-row Inserts**: Correctly splits `INSERT INTO table VALUES (row1), (row2), ...`

**Example Output**:
```
============================================================
Sakila Database - Create Original IR and Extract Data
============================================================

Schema file: sakila-schema.sql
Data file: sakila-data.sql
Output directory: ../data/sakila

============================================================
Step 1: Creating DatasetIR from SQL schema...
============================================================
Reading schema file: sakila-schema.sql
SQL file read: 45678 characters
  Parsing SQL content (45678 characters)...
  Found 16 CREATE TABLE statements with ENGINE=
  Processing table: actor
    - 4 columns, 1 PK columns, 0 FKs
  ...
  Total tables parsed: 16
Converting 16 tables to LogicalIR format...
Created LogicalIR with 16 tables

============================================================
Step 2: Saving IR to JSON...
============================================================
✓ Saved original_ir.json to ../data/sakila/original_ir.json

Summary:
  Total tables: 16
  - actor: 4 columns, PK: ['actor_id'], FKs: 0
  ...

============================================================
Step 3: Extracting data from INSERT statements...
============================================================
  Reading SQL file: sakila-data.sql
  SQL file read: 1234567 characters
  Searching for INSERT statements...
  Found 16 INSERT statement blocks

Found data for 16 tables:
  - actor: 200 rows
  - address: 603 rows
  ...

============================================================
Step 4: Creating CSV files...
============================================================
  Loading IR from ../data/sakila/original_ir.json
  Creating CSV files...
    Created actor.csv with 200 rows
    Created address.csv with 603 rows
    ...
```

### Dataset Analysis Framework

#### `analyze_datasets.py`

**Purpose**: Analyzes all datasets in the `data/` directory and generates comprehensive statistics.

**Key Features**:
1. **Multi-Table Support**: Handles both single-table and multi-table datasets
2. **Flexible Directory Structure**: Supports both one-level and two-level directory structures
3. **Comprehensive Statistics**: Generates statistics for numeric, categorical, temporal, and correlation data
4. **Distribution Fitting**: Tests for various distribution families (normal, lognormal, pareto, etc.)
5. **Zipf Analysis**: Detects and analyzes Zipf distributions in categorical data

**Updated Directory Detection**:

The script was updated to handle both directory structures:

```python
# Find all dataset directories
# Handle both one-level (data/dataset/) and two-level (data/source/dataset/) structures
dataset_dirs = []
for item in data_dir.iterdir():
    if not item.is_dir():
        continue
    
    # Check if this is a dataset directory (one-level structure)
    csv_files = list(item.glob("*.csv"))
    if len(csv_files) > 0:
        dataset_dirs.append(item)
    else:
        # Check if this is a source directory (two-level structure)
        for dataset_dir in item.iterdir():
            if dataset_dir.is_dir():
                csv_files = list(dataset_dir.glob("*.csv"))
                if len(csv_files) > 0:
                    dataset_dirs.append(dataset_dir)
```

**Generated Statistics**:

The `statistics.json` file contains:
- **Schema Information**: Table names, column names, types, nullability
- **Numeric Statistics**: Mean, std, min, max, distribution fits, skewness, Gini coefficient
- **Categorical Statistics**: Cardinality, value counts, Zipf fits, top-k shares
- **Temporal Statistics**: Date ranges, granularity, seasonal patterns
- **Correlations**: Strong correlations between columns (|r| > 0.7)

**Usage**:
```bash
python -m realistic_datasets.analyze_datasets
```

This processes all datasets and creates `statistics.json` files in each dataset directory.

#### `generate_descriptions.py`

**Purpose**: Generates diverse natural language descriptions from dataset statistics using LLMs.

**Key Features**:
1. **LLM-Based Generation**: Uses the configured LLM to generate descriptions
2. **Diversity Checking**: Ensures descriptions are sufficiently different using TF-IDF similarity
3. **Retry Logic**: Automatically retries if descriptions are too similar
4. **Multi-Table Support**: Generates descriptions for both single-table and multi-table schemas
5. **Flexible Directory Structure**: Supports both one-level and two-level directory structures

**Updated Directory Detection**:

The script was updated to find statistics files in both structures:

```python
# Find all statistics files
# Handle both one-level (data/dataset/) and two-level (data/source/dataset/) structures
stats_files = []
for item in data_dir.iterdir():
    if not item.is_dir():
        continue
    
    # Check if this is a dataset directory (one-level structure)
    stats_path = item / "statistics.json"
    if stats_path.exists():
        stats_files.append((stats_path, "root", item))
    else:
        # Check if this is a source directory (two-level structure)
        for dataset_dir in item.iterdir():
            if dataset_dir.is_dir():
                stats_path = dataset_dir / "statistics.json"
                if stats_path.exists():
                    stats_files.append((stats_path, item.name, dataset_dir))
```

**Description Generation Process**:

1. **Load Statistics**: Reads `statistics.json` for each dataset
2. **Format Statistics**: Formats schema, numeric, categorical, temporal, and correlation data
3. **Generate Variations**: Creates 3 different descriptions with variation instructions
4. **Check Diversity**: Uses TF-IDF cosine similarity to ensure descriptions differ
5. **Retry if Needed**: Retries up to 4 times if descriptions are too similar
6. **Save Descriptions**: Writes `description_1.txt`, `description_2.txt`, `description_3.txt`

**Similarity Threshold**: Default 0.4 (configurable via `--threshold`)

**Usage**:
```bash
python -m realistic_datasets.generate_descriptions
# or with custom threshold
python -m realistic_datasets.generate_descriptions --threshold 0.3
```

#### `generate_all_irs.py`

**Purpose**: Batch generator for creating GenerationIR from statistics for all datasets. This script orchestrates the complete statistics-to-IR conversion pipeline including functional dependency discovery, candidate key detection, and distribution mapping.

**Key Features**:
1. **FD Discovery**: Discovers functional dependencies from data using Apriori algorithm
2. **Candidate Key Detection**: Finds candidate keys and updates primary keys
3. **Distribution Mapping**: Converts statistical properties to distribution specifications
4. **Schema Updates**: Updates original_ir.json with discovered constraints
5. **Progress Logging**: Comprehensive logging for monitoring progress
6. **Error Handling**: Graceful error handling with detailed error messages

**Process Flow**:

1. **FD Discovery** (`statistics_to_ir/fd_discovery.py`):
   
   **Algorithm**: Apriori-based functional dependency discovery
   
   - **Phase 1: Frequent Itemset Discovery (Apriori Algorithm)**:
     - **Level 1**: Find all single columns with support ≥ `min_support`
       - Support = (unique values) / (total rows)
       - Example: Column with 95,000 unique values in 100,000 rows has support = 0.95
       
       ```python
       # Level 1: Single columns
       frequent_1 = []
       for col in columns:
           support = compute_column_support(clean_df, col)
           if support >= min_support:
               frequent_1.append((col,))
       
       def compute_column_support(df: pd.DataFrame, col: str) -> float:
           unique_count = df[col].nunique()
           total_count = len(df)
           return unique_count / total_count if total_count > 0 else 0.0
       ```
     
     - **Level 2 to k**: Generate k-itemset candidates from (k-1)-itemsets
       - Join itemsets that share first (k-2) elements
       - Example: `(A, B)` and `(A, C)` → candidate `(A, B, C)`
       - Compute support for each candidate: unique combinations / total rows
       - Filter candidates with support < `min_support`
       
       ```python
       # Generate k+1 itemsets from k itemsets
       for k in range(2, min(max_size + 1, len(columns) + 1)):
           candidates = generate_candidates(current_level, k)
           
           frequent_k = []
           for candidate in candidates:
               support = compute_itemset_support(clean_df, candidate)
               if support >= min_support:
                   frequent_k.append(candidate)
           
           if not frequent_k:
               break
           current_level = frequent_k
       
       def generate_candidates(frequent_k_minus_1, k):
           """Join itemsets that share first (k-2) elements."""
           candidates = []
           for i in range(len(frequent_k_minus_1)):
               for j in range(i + 1, len(frequent_k_minus_1)):
                   itemset1 = frequent_k_minus_1[i]
                   itemset2 = frequent_k_minus_1[j]
                   
                   if k > 2 and itemset1[:k-2] == itemset2[:k-2]:
                       candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                       if len(candidate) == k:
                           candidates.append(candidate)
           return candidates
       
       def compute_itemset_support(df: pd.DataFrame, itemset: Tuple[str, ...]) -> float:
           """Support = unique combinations / total rows"""
           grouped = df[list(itemset)].groupby(list(itemset))
           unique_combinations = len(grouped)
           total_rows = len(df)
           return unique_combinations / total_rows if total_rows > 0 else 0.0
       ```
     
     - **Maximum LHS Size**: Configurable `max_lhs_size` (default 3)
       - For N columns: generates O(N³) candidates at most
       - Prevents exponential explosion
   
   - **Phase 2: FD Validation**:
     - For each frequent itemset (LHS), check all remaining columns as RHS
     - **Confidence Calculation**:
       - Confidence = P(RHS | LHS) = (rows where LHS uniquely determines RHS) / (total rows)
       - Computed by grouping on LHS and checking if each group has unique RHS
       - Example: If `(A, B)` → `C` with confidence 0.98, then 98% of rows have unique C for each (A,B) pair
     
     ```python
     # For each frequent itemset (LHS), check all possible RHS
     for lhs_cols in frequent_itemsets:
         rhs_candidates = [c for c in non_key_columns if c not in lhs_cols]
         
         for rhs_col in rhs_candidates:
             confidence = compute_fd_confidence(df, list(lhs_cols), rhs_col)
             
             if confidence >= min_confidence:
                 fd = FDConstraint(
                     table=table_name,
                     lhs=list(lhs_cols),
                     rhs=[rhs_col],
                     mode="intra_row"
                 )
                 discovered_fds.append(fd)
     
     def compute_fd_confidence(df: pd.DataFrame, lhs_cols: List[str], rhs_col: str) -> float:
         """Compute confidence: proportion of rows where LHS uniquely determines RHS"""
         clean_df = df[lhs_cols + [rhs_col]].dropna()
         if len(clean_df) == 0:
             return 0.0
         
         grouped = clean_df.groupby(lhs_cols)
         rows_with_unique_rhs = 0
         total_rows = len(clean_df)
         
         for name, group in grouped:
             if group[rhs_col].nunique() == 1:  # Unique RHS for this LHS
                 rows_with_unique_rhs += len(group)
         
         return rows_with_unique_rhs / total_rows if total_rows > 0 else 0.0
     ```
     
     - Only FDs with confidence ≥ `min_confidence` (default 0.95) are kept
   
   - **Performance Optimizations**:
     - **Automatic Sampling**: Tables with >100K rows are automatically sampled to 100K rows for FD discovery
       - Uses `random_state=42` for reproducibility
       - Reduces processing time from hours to minutes for large datasets
     - **Progress Logging**: Detailed progress logging per table, itemset level, and FD candidate checks
       - Table-level: `[1/7] name.basics: 14,907,745 rows -> sampling 100,000 rows`
       - Apriori level: `Level 2: 15/15 frequent itemsets`
       - Itemset processing: `Processing itemset 1/41: ('title',)`
       - Summary: `Completed: checked 150 FD candidates, found 89 FDs`
     - **Configurable Thresholds**: `min_support` (default 0.95) and `min_confidence` (default 0.95)
   
   - **Column Filtering**:
     - Excludes primary key columns (PK → all columns is trivial)
     - Excludes foreign key columns (FK → referenced PK is already defined)
     - Only considers non-key columns for FD discovery
   
   - **Output**: List of `FDConstraint` objects with:
     - `table`: Table name
     - `lhs`: List of LHS column names
     - `rhs`: List of RHS column names (typically single column)
     - `mode`: "intra_row" (functional dependency within a row)
     - Support and confidence stored in separate map

2. **Candidate Key Processing** (`statistics_to_ir/candidate_key_discovery.py`):
   
   **Algorithm**: Candidate key discovery from functional dependencies
   
   - **Definition**: A candidate key is a set of columns (LHS) that functionally determines ALL other columns in the table with perfect confidence (1.0) and support (1.0)
   
   - **FD Closure Computation**:
     - Build direct FD map: LHS → set of RHS columns (for FDs with confidence = 1.0)
     - Compute transitive closure using fixed-point iteration:
       - Initialize: Each LHS determines itself
       - Iterate: If LHS determines columns X, and X determines columns Y, then LHS determines Y
       - Continue until no new columns are added (fixed point)
     - Example: If `A → B`, `B → C`, then closure of `A` includes `{A, B, C}`
     
     ```python
     # Build direct FD map (only perfect FDs: confidence = 1.0)
     direct_fds: Dict[Tuple[str, ...], Set[str]] = {}
     for fd in table_fds:
         lhs_tuple = tuple(sorted(fd.lhs))
         rhs_col = fd.rhs[0]
         support, confidence = support_confidence_map[(table_name, tuple(fd.lhs), rhs_col)]
         if support >= 1.0 and confidence >= 1.0:
             direct_fds[lhs_tuple].add(rhs_col)
     
     # Compute transitive closure using fixed-point iteration
     fd_closure: Dict[Tuple[str, ...], Set[str]] = {}
     for lhs_tuple in direct_fds.keys():
         fd_closure[lhs_tuple] = set(lhs_tuple)  # Initialize: LHS determines itself
     
     changed = True
     while changed:
         changed = False
         for lhs_tuple in list(fd_closure.keys()):
             determined = fd_closure[lhs_tuple].copy()
             
             # Check all FDs where LHS is a subset of determined columns
             for fd_lhs_tuple, rhs_cols in direct_fds.items():
                 if set(fd_lhs_tuple).issubset(determined):
                     for rhs_col in rhs_cols:
                         if rhs_col not in determined:
                             determined.add(rhs_col)
                             changed = True
             
             fd_closure[lhs_tuple] = determined
     
     # Find candidate keys: LHS that determine ALL columns
     candidate_keys = []
     for lhs_tuple, determined in fd_closure.items():
         if determined.issuperset(available_columns_set):
             candidate_keys.append(list(lhs_tuple))
     ```
   
   - **Candidate Key Detection**:
     - For each LHS in FD closure, check if it determines all columns in the table
     - Also check single columns that are unique (nunique == row count)
     - Filter to minimal candidate keys (no proper subset is also a candidate key)
     
     ```python
     # Find candidate keys: LHS that determine ALL columns
     candidate_keys = []
     for lhs_tuple, determined in fd_closure.items():
         if determined.issuperset(available_columns_set):
             candidate_keys.append(list(lhs_tuple))
     
     # Also check unique single columns
     for col in available_columns:
         if df[col].nunique() == len(df[col].dropna()):
             # Column is unique, check if it determines all columns
             col_tuple = (col,)
             if col_tuple in fd_closure:
                 if fd_closure[col_tuple].issuperset(available_columns_set):
                     if [col] not in candidate_keys:
                         candidate_keys.append([col])
     
     # Find minimal candidate keys
     def find_minimal_candidate_keys(candidate_keys):
         """Remove candidate keys that are supersets of others."""
         sorted_keys = sorted(candidate_keys, key=len)
         minimal_keys = []
         
         for candidate in sorted_keys:
             candidate_set = set(candidate)
             is_superset = any(
                 set(minimal).issubset(candidate_set) and len(minimal) < len(candidate)
                 for minimal in minimal_keys
             )
             if not is_superset:
                 minimal_keys.append(candidate)
         
         return minimal_keys
     ```
   
   - **Primary Key Selection**:
     - If existing primary key from DDL is a candidate key, keep it
     - Otherwise, select smallest candidate key (prefer single-column keys)
     - Uses LLM for conflict resolution if multiple minimal candidate keys exist (optional)
   
   - **FD Separation**:
     - Separates FDs into candidate-key FDs (LHS is a candidate key) and regular FDs
     - Candidate-key FDs are used for primary key inference
     - Regular FDs are used for constraint enforcement during data generation

3. **Schema Updates** (`statistics_to_ir/schema_updater.py`):
   
   **Process**: Integrates discovered constraints into the LogicalIR schema
   
   - **FD Integration**:
     - Separates candidate-key FDs from regular FDs
     - Candidate-key FDs: Used to infer/update primary keys
     - Regular FDs: Added to `LogicalIR.constraints.fds` as `FDConstraint` objects
   
   - **Candidate Key Integration**:
     - Adds discovered candidate keys to `TableSpec.candidate_keys`
     - Updates `TableSpec.primary_key` if:
       - No primary key exists in DDL, or
       - Existing primary key is not a candidate key
     - Prefers minimal candidate keys (smallest sets)
   
   - **Primary Key Selection**:
     - If multiple candidate keys exist, selects based on:
       1. Existing primary key from DDL (if it's a candidate key)
       2. Smallest candidate key (prefer single-column)
       3. LLM assistance for ambiguous cases (optional)
   
   - **File Update**:
     - Loads existing `original_ir.json` (may be LogicalIR or DatasetIR)
     - Updates LogicalIR with new constraints and candidate keys
     - Saves back to `original_ir.json`
     - Preserves other IR components (GenerationIR, WorkloadIR) if present
     
     ```python
     def update_original_ir_file(original_ir_path: Path, updated_logical_ir: LogicalIR):
         """Update original_ir.json with discovered constraints."""
         # Load existing IR
         existing_ir = load_ir_from_json(original_ir_path)
         
         # Update LogicalIR
         if hasattr(existing_ir, 'logical'):
             existing_ir.logical = updated_logical_ir
         else:
             # It's just LogicalIR, replace it
             existing_ir = updated_logical_ir
         
         # Save back
         save_ir_to_json(existing_ir, original_ir_path)
     
     # Example: Adding FDs to constraints
     for table_name, table in updated_logical_ir.tables.items():
         table_fds = [fd for fd in regular_fds if fd.table == table_name]
         if table_fds:
             if not updated_logical_ir.constraints:
                 updated_logical_ir.constraints = ConstraintSpec()
             
             table_fd_constraint = TableFDConstraint(
                 table=table_name,
                 fds=table_fds
             )
             updated_logical_ir.constraints.fds.append(table_fd_constraint)
     ```

4. **Statistics to GenerationIR** (`statistics_to_ir/stats_converter.py`):
   
   **Process**: Converts statistical properties from `statistics.json` to `GenerationIR` distribution specifications
   
   - **Column-by-Column Conversion**:
     - For each column in LogicalIR, retrieves statistics from `statistics.json`
     - Determines if column is numeric or categorical based on statistics
     - Handles type conflicts (both numeric and categorical stats exist) using LLM or fallback logic
   
   - **Numeric Distribution Mapping** (`statistics_to_ir/distribution_mapper.py`):
     - **Supported Distributions**:
       - **Uniform**: `DistUniform(low, high)` - for columns with no clear pattern
       - **Normal**: `DistNormal(mean, std)` - for normally distributed data
       - **Lognormal**: `DistLognormal(mean, sigma)` - for right-skewed positive data
       - **Pareto**: `DistPareto(alpha, xm)` - for power-law distributions
       - **Exponential**: `DistExponential(scale)` - for exponential decay patterns
       - **Poisson**: `DistPoisson(lam)` - for count data
     - **Selection Process**:
       1. Check `distribution_fit.best_fit` from statistics (from KS tests during analysis)
       2. If `best_pvalue < 0.05` and LLM available: Ask LLM to choose best distribution
       3. Otherwise: Use `best_fit` from statistics
       4. Extract parameters from `distribution_fit.fits[best_fit]`
       5. Fallback to uniform distribution if no fit is suitable
     
     ```python
     def convert_numeric_stats_to_distribution(
         numeric_stats: Dict[str, Any],
         llm_client: Optional[LLMClient] = None
     ) -> Distribution:
         dist_fit = numeric_stats.get("distribution_fit", {})
         best_fit = dist_fit.get("best_fit")
         best_pvalue = dist_fit.get("best_pvalue")
         
         # Handle None p-values
         if best_pvalue is None:
             best_pvalue = 0.0
         
         fits = dist_fit.get("fits", {})
         
         # If p-value is too low, ask LLM to choose
         if best_pvalue < 0.05 and llm_client:
             decision = llm_client.select_distribution(
                 table_name="", column_name="", column_type="FLOAT",
                 statistical_properties={...},
                 distribution_fits=[...]
             )
             if decision and "selected_distribution" in decision:
                 best_fit = decision["selected_distribution"]
                 fit_params = decision.get("parameters", fits.get(best_fit, {}))
         else:
             fit_params = fits.get(best_fit, {})
         
         # Convert based on best_fit
         if best_fit == "normal":
             return DistNormal(
                 kind="normal",
                 mean=fit_params.get("mean", numeric_stats.get("mean", 0.0)),
                 std=fit_params.get("std", numeric_stats.get("std", 1.0))
             )
         elif best_fit == "lognormal":
             return DistLognormal(
                 kind="lognormal",
                 mean=fit_params.get("shape", 1.0),
                 sigma=max(fit_params.get("scale", 1.0), 0.001)
             )
         # ... other distributions ...
         
         # Fallback to uniform
         return DistUniform(
             kind="uniform",
             low=numeric_stats.get("min", 0.0),
             high=numeric_stats.get("max", 1.0)
         )
     ```
     - **Parameter Mapping**:
       - Normal: `mean`, `std` from fit parameters
       - Lognormal: `shape` → `mean`, `scale` → `sigma` (with minimum 0.001)
       - Pareto: `alpha`, `scale` → `xm` (with minimum 0.001)
       - Exponential: `scale` (with minimum 0.001)
       - Poisson: `lambda` → `lam` (with minimum 0.001)
     - **Bug Fix**: Properly handles `None` p-values from statistics (defaults to 0.0)
       - Prevents `TypeError: '<' not supported between instances of 'NoneType' and 'float'`
       - Occurs when distribution fitting fails or statistics are incomplete
   
   - **Categorical Distribution Mapping**:
     - **Decision Logic**:
       - High cardinality (>1000) or skewed (top-1 share > 0.1 with cardinality > 100) → Consider Zipf
       - Otherwise → Use categorical distribution
     - **Zipf Distribution** (`DistZipf(s, n)`):
       - Parameter `s`: Zipf exponent (from `zipf_fit.s` in statistics)
       - Parameter `n`: Cardinality (number of unique values)
       - Uses LLM to decide between categorical and Zipf if ambiguous (optional)
     - **Categorical Distribution** (`DistCategorical(domain)`):
       - `domain.values`: List of unique values from `value_counts`
       - `domain.probs`: Normalized probabilities from `value_counts`
       - Probabilities renormalized if sum ≠ 1.0
     
     ```python
     def convert_categorical_stats_to_distribution(
         categorical_stats: Dict[str, Any],
         table_name: str,
         column: ColumnSpec,
         llm_client: Optional[LLMClient] = None
     ) -> Tuple[Distribution, Optional[ProviderRef]]:
         cardinality = categorical_stats.get("cardinality", 0)
         value_counts = categorical_stats.get("value_counts", {})
         top_1_share = categorical_stats.get("top_1_share", 0.0)
         zipf_fit = categorical_stats.get("zipf_fit")
         
         # High cardinality or skewed → consider Zipf
         if cardinality > 1000 or (top_1_share > 0.1 and cardinality > 100):
             if llm_client:
                 decision = llm_client.decide_categorical_vs_zipf(
                     table_name, column.name, categorical_stats, zipf_fit
                 )
                 if decision and decision.get("distribution_type") == "zipf":
                     return DistZipf(
                         kind="zipf",
                         s=decision["parameters"].get("s", 1.2),
                         n=cardinality
                     ), None
             elif zipf_fit and zipf_fit.get("s"):
                 return DistZipf(
                     kind="zipf",
                     s=zipf_fit["s"],
                     n=cardinality
                 ), None
         
         # Use categorical distribution
         values = list(value_counts.keys())
         counts = list(value_counts.values())
         total = sum(counts)
         probs = [c / total for c in counts] if total > 0 else None
         
         # Normalize probabilities
         if probs and abs(sum(probs) - 1.0) > 0.01:
             total_prob = sum(probs)
             probs = [p / total_prob for p in probs]
         
         return DistCategorical(
             kind="categorical",
             domain=CategoricalDomain(values=values, probs=probs)
         ), None
     ```
   
   - **Provider Suggestion** (Optional):
     - LLM can suggest using external providers for certain columns
     - Example: Use a name provider for person names, address provider for addresses
     - Returns `ProviderRef` with provider name and configuration
   
   - **Missing Statistics Handling**:
     - If no statistics available for a column, uses LLM to infer distribution (optional)
     - Falls back to uniform distribution for numeric, categorical for text columns

**FD Discovery Improvements**:

The FD discovery process has been optimized for large datasets:

- **Sampling**: Large tables (>100K rows) are automatically sampled to 100K rows, significantly speeding up processing for datasets like IMDB (millions of rows)
- **Progress Logging**: 
  - Per-table progress: `[1/7] name.basics: 14,907,745 rows -> sampling 100,000 rows`
  - Apriori level progress: `Level 2: 15/15 frequent itemsets`
  - Itemset processing: `Processing itemset 1/41: ('title',)`
  - Summary: `Completed: checked 150 FD candidates, found 89 FDs`

**LLM Integration** (`statistics_to_ir/llm_assistant.py`):

The module supports optional LLM assistance for ambiguous cases:

- **Distribution Selection**: When p-value is too low (<0.05), LLM chooses best distribution from candidates
- **Type Conflict Resolution**: Resolves conflicts when column has both numeric and categorical statistics
- **Categorical vs Zipf Decision**: Decides between categorical and Zipf distributions for high-cardinality columns
- **Provider Suggestion**: Suggests external providers (e.g., name providers, address providers) for certain columns
- **Missing Statistics Inference**: Infers distributions for columns without statistics
- **Primary Key Selection**: Helps select primary key when multiple candidate keys exist

**LLM Configuration**:

- Set `use_llm=True` in `create_generation_ir_from_statistics()` to enable
- Provide `llm_config` dictionary with API key, model name, etc.
- Falls back to deterministic logic if LLM is disabled or unavailable

**Distribution Mapper Fix**:

Fixed `TypeError` when `best_pvalue` is `None` in statistics:
- Added explicit `None` check before p-value comparison
- Defaults to 0.0 if p-value is missing or `None`
- Prevents crashes on datasets with incomplete distribution fits
- Common cause: Distribution fitting failed during statistics generation

**Usage**:
```bash
python -m realistic_datasets.generate_all_irs
```

This processes all datasets and creates `generation_ir.json` files in each dataset directory.

**Programmatic Usage**:

```python
from pathlib import Path
from realistic_datasets.statistics_to_ir import create_generation_ir_from_statistics

# Generate GenerationIR for a single dataset
stats_path = Path("realistic_datasets/data/imdb/imdb/statistics.json")
original_ir_path = Path("realistic_datasets/data/imdb/imdb/original_ir.json")
data_dir = Path("realistic_datasets/data/imdb/imdb")
output_path = Path("realistic_datasets/data/imdb/imdb/generation_ir.json")

generation_ir = create_generation_ir_from_statistics(
    stats_path=stats_path,
    original_ir_path=original_ir_path,
    data_dir=data_dir,
    output_path=output_path,
    min_support=0.95,
    min_confidence=0.95,
    use_llm=False  # Set to True when LLM is configured
)
```

**FD Discovery Usage**:

```python
from realistic_datasets.statistics_to_ir.fd_discovery import discover_functional_dependencies
from nl2data.utils.ir_io import load_ir_from_json
import pandas as pd

# Load data and schema
dfs = {
    "table1": pd.read_csv("table1.csv"),
    "table2": pd.read_csv("table2.csv")
}
logical_ir = load_ir_from_json("original_ir.json").logical

# Discover FDs
discovered_fds, support_confidence_map = discover_functional_dependencies(
    dfs=dfs,
    logical_ir=logical_ir,
    min_support=0.95,      # Minimum support threshold
    min_confidence=0.95,  # Minimum confidence threshold
    max_lhs_size=3        # Maximum columns in LHS
)

# Access results
for fd in discovered_fds:
    print(f"{fd.table}: {fd.lhs} → {fd.rhs}")
    key = (fd.table, tuple(fd.lhs), fd.rhs[0])
    support, confidence = support_confidence_map[key]
    print(f"  Support: {support:.3f}, Confidence: {confidence:.3f}")
```

**Output Files**:

- **`discovered_fds.json`**: JSON file containing:
  - `fds`: List of discovered functional dependencies
    - Each FD: `{table, lhs: [columns], rhs: [columns], mode: "intra_row"}`
  - `support_confidence`: Map of `(table, tuple(lhs), rhs)` → `(support, confidence)`
  - Example:
    ```json
    {
      "fds": [
        {
          "table": "title.akas",
          "lhs": ["title", "region"],
          "rhs": ["language"],
          "mode": "intra_row"
        },
        {
          "table": "name.basics",
          "lhs": ["primaryName"],
          "rhs": ["birthYear"],
          "mode": "intra_row"
        }
      ],
      "support_confidence": {
        "('title.akas', ('title', 'region'), 'language')": [0.98, 0.97],
        "('name.basics', ('primaryName',), 'birthYear')": [0.95, 0.96]
      }
    }
    ```

- **`generation_ir.json`**: GenerationIR with distribution specifications
  - `columns`: List of `ColumnGenSpec` objects
    - Each spec: `{table, column, distribution, provider, constraints}`
  - `events`: List of event specifications (typically empty)
  
  Example:
  ```json
  {
    "columns": [
      {
        "table": "title.basics",
        "column": "primaryTitle",
        "distribution": {
          "kind": "categorical",
          "domain": {
            "values": ["The Matrix", "Inception", ...],
            "probs": [0.001, 0.0008, ...]
          }
        },
        "provider": null,
        "constraints": []
      },
      {
        "table": "title.ratings",
        "column": "averageRating",
        "distribution": {
          "kind": "normal",
          "mean": 6.5,
          "std": 1.2
        },
        "provider": null,
        "constraints": []
      },
      {
        "table": "name.basics",
        "column": "primaryName",
        "distribution": {
          "kind": "zipf",
          "s": 1.3,
          "n": 50000
        },
        "provider": null,
        "constraints": []
      }
    ],
    "events": []
  }
  ```

- **Updated `original_ir.json`**: DatasetIR with:
  - `logical`: Updated LogicalIR with discovered FDs and candidate keys
  - `generation`: GenerationIR merged into DatasetIR
  - `workload`: Optional WorkloadIR

**Statistics Structure** (from `analyze_datasets.py`):

The `statistics.json` file contains comprehensive statistical properties:

- **Schema Information**: Table names, column names, SQL types, nullability
- **Numeric Statistics** (for numeric columns):
  - Basic: `mean`, `std`, `min`, `max`, `median`, `q25`, `q75`
  - Distribution fits: `distribution_fit.fits` with KS test results for each distribution
  - Best fit: `distribution_fit.best_fit`, `distribution_fit.best_pvalue`
  - Shape: `skewness`, `kurtosis`
  - Inequality: `gini_coefficient`
  
  Example:
  ```json
  {
    "tables": {
      "title.ratings": {
        "columns": {
          "averageRating": {
            "type": "FLOAT",
            "numeric_stats": {
              "mean": 6.5,
              "std": 1.2,
              "min": 1.0,
              "max": 10.0,
              "median": 6.7,
              "skewness": -0.3,
              "kurtosis": 2.1,
              "distribution_fit": {
                "best_fit": "normal",
                "best_pvalue": 0.15,
                "fits": {
                  "normal": {
                    "mean": 6.5,
                    "std": 1.2,
                    "ks_pvalue": 0.15
                  },
                  "lognormal": {
                    "shape": 1.8,
                    "scale": 0.2,
                    "ks_pvalue": 0.03
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  ```
- **Categorical Statistics** (for categorical/text columns):
  - `cardinality`: Number of unique values
  - `value_counts`: Map of value → count
  - `top_k_shares`: Proportion of top-k values
  - `zipf_fit`: Zipf distribution parameters if applicable (`s`, `r_squared`)
  
  Example:
  ```json
  {
    "tables": {
      "name.basics": {
        "columns": {
          "primaryName": {
            "type": "STRING",
            "categorical_stats": {
              "cardinality": 50000,
              "value_counts": {
                "John Smith": 150,
                "Jane Doe": 120,
                ...
              },
              "top_1_share": 0.003,
              "top_10_share": 0.025,
              "zipf_fit": {
                "s": 1.3,
                "r_squared": 0.94
              }
            }
          }
        }
      }
    }
  }
  ```
- **Temporal Statistics** (for date/datetime columns):
  - `date_range`: Min/max dates
  - `granularity`: Day/week/month/year
  - `seasonal_patterns`: Month/weekday distributions
- **Correlations**: Strong correlations (|r| > 0.7) between numeric columns

### Complete Workflow

**Step 1: Create Database IRs and Extract Data**

```bash
# For Sakila
cd realistic_datasets/sakila
python create_db.py

# For World
cd realistic_datasets/world
python create_db.py
```

This creates:
- `data/sakila/original_ir.json` and `data/world/original_ir.json`
- CSV files for all tables

**Step 2: Generate Statistics**

```bash
cd realistic_datasets
python analyze_datasets.py
```

This creates:
- `data/sakila/statistics.json` and `data/world/statistics.json`

**Step 3: Generate NL Descriptions**

```bash
cd realistic_datasets
python generate_descriptions.py
```

This creates:
- `data/sakila/description_1.txt`, `description_2.txt`, `description_3.txt`
- `data/world/description_1.txt`, `description_2.txt`, `description_3.txt`

**Step 3.5: Generate GenerationIR from Statistics** (Optional but recommended)

```bash
cd realistic_datasets
python generate_all_irs.py
```

This creates:
- `data/sakila/generation_ir.json` and `data/world/generation_ir.json`
- `data/sakila/discovered_fds.json` and `data/world/discovered_fds.json`
- Updated `data/sakila/original_ir.json` and `data/world/original_ir.json` (with FDs and candidate keys)

**Note**: This step discovers functional dependencies, finds candidate keys, and converts statistics to distribution specifications. It's particularly useful for datasets with complex schemas and large tables.

**Performance Characteristics**:

- **Time Complexity**:
  - FD Discovery: O(N × M² × R) where N = tables, M = columns, R = rows (after sampling)
  - With sampling: O(N × M² × 100K) for large tables
  - Candidate Key Discovery: O(M³) per table (FD closure computation)
  - Distribution Mapping: O(M) per table (linear in number of columns)
  
- **Space Complexity**:
  - FD Discovery: O(M²) for frequent itemsets and FD candidates
  - Candidate Keys: O(M²) for FD closure map
  
- **Typical Processing Times** (on modern hardware):
  - Small datasets (<10K rows): <1 second per table
  - Medium datasets (10K-100K rows): 1-10 seconds per table
  - Large datasets (>100K rows, sampled): 10-60 seconds per table
  - IMDB dataset (7 tables, ~100M total rows): ~5-10 minutes total
  - TPC-H dataset (8 tables, ~8M total rows): ~2-5 minutes total

**Step 4: Run Pipeline**

```bash
cd realistic_datasets
python run_pipeline_all.py
```

This runs the NL→IR→Data pipeline for each description and generates synthetic data.

**Step 5: Evaluate**

```bash
cd realistic_datasets
python evaluate_all_datasets.py
```

This evaluates the generated data against the original schemas.

### Integration with Other Data Sources

The framework also supports integration with:

- **OpenML**: Machine learning datasets via API
- **Data.gov**: US government open data
- **World Bank**: Economic and social indicators
- **OpenStreetMap**: Geographic data
- **US Census**: Demographic data

Each data source has its own directory with:
- `create_original_ir.py`: Schema extraction script
- `fetch_*.py`: Data fetching script
- `README.md`: Source-specific documentation

### Benefits

1. **Realistic Evaluation**: Test the NL2Data pipeline on real-world schemas and data
2. **Schema Diversity**: Access to diverse database schemas (OLTP, OLAP, etc.)
3. **Automated Workflow**: End-to-end automation from SQL to evaluation
4. **Reproducibility**: Consistent process for all datasets
5. **Extensibility**: Easy to add new data sources

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
    DistLognormal,    # Log-normal distribution (mean, sigma) - right-skewed
    DistPareto,       # Pareto distribution (alpha, xm) - heavy-tailed
    DistPoisson,      # Poisson distribution (lam) - count distributions
    DistExponential,  # Exponential distribution (scale) - inter-arrival times
    DistMixture,      # Mixture distribution (components with weights) - multi-modal
    DistZipf,         # Zipf distribution (s exponent, n domain size) - power-law
    DistSeasonal,     # Seasonal dates (weights, granularity: month/week/hour)
    DistCategorical,  # Categorical (values, probabilities) - discrete values
    DistDerived,      # Derived column (expression, dtype, depends_on) - computed
    DistWindow,       # Window function (expression, partition_by, order_by, frame) - rolling aggregations
]
```

**Key Features**:
- Specifies how each column should be generated
- **Comprehensive Distribution Support**: 12 distribution types covering numeric, discrete, temporal, and computed patterns:
  - **Numeric**: Uniform, Normal, Lognormal, Pareto, Poisson, Exponential, Mixture
  - **Discrete**: Zipf (power-law), Categorical
  - **Temporal**: Seasonal (month/week/hour granularity)
  - **Computed**: Derived (DSL expressions), Window (rolling aggregations)
- Derived columns with dependency tracking
- Window functions for rolling aggregations and lag/lead operations
- Event system for temporal patterns (pay-day spikes, incidents, etc.)
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

The multi-agent system is the core orchestration mechanism that transforms natural language descriptions into structured Intermediate Representations (IRs). It uses a **Blackboard Pattern** for agent communication, where specialized agents sequentially refine the specification through progressive IR transformations.

#### 2.1 Blackboard Pattern (`agents/base.py`)

**Purpose**: Centralized shared state for inter-agent communication.

**Architecture**:
The Blackboard is a Pydantic model that serves as the single source of truth for all IRs during pipeline execution. It implements a **write-once, read-many** pattern where each agent:
1. **Reads** from previous IRs (its inputs)
2. **Writes** its output IR (its contribution)
3. **Never modifies** IRs written by other agents

```python
class Blackboard(BaseModel):
    """
    Shared blackboard for multi-agent communication.
    
    Agents read from and write to the blackboard to pass
    intermediate representations between stages.
    """
    requirement_ir: Optional[RequirementIR] = None
    conceptual_ir: Optional[ConceptualIR] = None
    logical_ir: Optional[LogicalIR] = None
    generation_ir: Optional[GenerationIR] = None
    workload_ir: Optional[WorkloadIR] = None
    dataset_ir: Optional[DatasetIR] = None
```

**Design Rationale**:
- **Type Safety**: Pydantic validation ensures IRs are valid before being stored
- **Immutability**: Once an IR is written, it's not modified by subsequent agents (agents create new IRs)
- **Sequential Dependencies**: Clear data flow - each agent depends on previous IRs
- **Debugging**: Can inspect blackboard state at any point in the pipeline
- **Testability**: Can create blackboards with specific IRs for unit testing

**Blackboard Lifecycle**:
```
Initial State:
  Blackboard(
    requirement_ir=None,
    conceptual_ir=None,
    logical_ir=None,
    generation_ir=None,
    workload_ir=None,
    dataset_ir=None
  )

After ManagerAgent:
  Blackboard(
    requirement_ir=RequirementIR(...),  ← Written by ManagerAgent
    conceptual_ir=None,
    ...
  )

After ConceptualDesigner:
  Blackboard(
    requirement_ir=RequirementIR(...),  ← Read by ConceptualDesigner
    conceptual_ir=ConceptualIR(...),   ← Written by ConceptualDesigner
    ...
  )

... and so on for each agent
```

**Access Patterns**:
- **ManagerAgent**: Writes `requirement_ir` (no dependencies)
- **ConceptualDesigner**: Reads `requirement_ir`, writes `conceptual_ir`
- **LogicalDesigner**: Reads `requirement_ir` + `conceptual_ir`, writes `logical_ir`
- **DistributionEngineer**: Reads `requirement_ir` + `logical_ir`, writes `generation_ir`
- **WorkloadDesigner**: Reads `requirement_ir` + `logical_ir`, writes `workload_ir`
- **QACompilerAgent**: Reads all IRs, writes `dataset_ir` (combines all)

#### 2.2 Base Agent Architecture (`agents/base.py`)

**Purpose**: Abstract base class defining the agent interface and common behavior.

**Complete BaseAgent Implementation**:

```python
class BaseAgent:
    """Base class for all agents in the multi-agent system."""
    
    name: str = "base_agent"  # Must be overridden by subclasses
    
    def _produce(self, board: Blackboard) -> Blackboard:
        """
        Produce initial IR from blackboard.
        
        This is the main production method that agents should implement.
        By default, it calls run() for backward compatibility.
        
        Process:
        1. Read required IRs from blackboard
        2. Load prompts (system + user templates)
        3. Render user prompt with IR data
        4. Call LLM API
        5. Extract JSON from response
        6. Validate as IR model (Pydantic)
        7. Write IR to blackboard
        8. Return updated blackboard
        
        Args:
            board: Current blackboard state
            
        Returns:
            Updated blackboard with new IR written
        """
        # Default implementation calls run() for backward compatibility
        return self.run(board)
    
    def _repair(self, board: Blackboard, qa_items: List["QaIssue"]) -> Blackboard:
        """
        Repair IR given QA feedback.
        
        Default implementation is a no-op. Agents can override this
        to implement repair logic (typically LLM-based repair).
        
        Repair Process (if implemented):
        1. Build repair prompt from QaIssue list
        2. Include current IR state in prompt
        3. Call LLM with repair prompt
        4. Extract and validate fixed IR
        5. Update blackboard with fixed IR
        
        Args:
            board: Current blackboard state
            qa_items: List of QaIssue objects from validation
            
        Returns:
            Updated blackboard (unchanged by default)
        """
        logger.warning(
            f"Agent {self.name} does not implement _repair(), "
            f"ignoring {len(qa_items)} QA issues"
        )
        return board
    
    def run(self, board: Blackboard) -> Blackboard:
        """
        Execute the agent's task.
        
        This is the legacy entry point. New code should use _produce()
        and _repair() for better integration with repair loops.
        
        Args:
            board: Current blackboard state
            
        Returns:
            Updated blackboard
        """
        logger.info(f"Running agent: {self.name}")
        raise NotImplementedError(f"Agent {self.name} must implement run() or _produce()")
```

**Agent Method Responsibilities**:

1. **`_produce()`**: 
   - **Purpose**: Generate initial IR from blackboard inputs
   - **When Called**: First attempt to create IR, before validation
   - **Must Implement**: All agents that generate IRs
   - **Returns**: Blackboard with new IR written

2. **`_repair()`**:
   - **Purpose**: Fix IR based on validation feedback
   - **When Called**: After validation finds issues, up to `max_retries` times
   - **Optional**: Agents can override for custom repair logic
   - **Default**: No-op (logs warning and returns unchanged board)
   - **Returns**: Blackboard with fixed IR written

3. **`run()`**:
   - **Purpose**: Legacy entry point for backward compatibility
   - **When Called**: By Orchestrator or direct agent invocation
   - **Default Behavior**: Calls `_produce()` (for agents that don't override `run()`)
   - **Returns**: Blackboard with IR written

**Agent Execution Pattern**:
All agents follow this pattern in their `run()` or `_produce()` methods:

```python
def run(self, board: Blackboard) -> Blackboard:
    # Step 1: Check prerequisites
    if board.requirement_ir is None:
        logger.warning("Missing prerequisite IR, skipping")
        return board
    
    # Step 2: Load prompts
    sys_tmpl = load_prompt("roles/{agent}_system.txt")
    usr_tmpl = load_prompt("roles/{agent}_user.txt")
    
    # Step 3: Render user prompt with IR data
    user_content = render_prompt(
        usr_tmpl,
        REQUIREMENT_JSON=board.requirement_ir.model_dump_json(indent=2),
        # ... other IRs as needed
    )
    
    # Step 4: Build messages
    messages = [
        {"role": "system", "content": sys_tmpl},
        {"role": "user", "content": user_content},
    ]
    
    # Step 5: Call LLM with retry logic
    for attempt in range(max_retries):
        try:
            raw = chat(messages)
            data = extract_json(raw)
            board.{ir_field} = {IRModel}.model_validate(data)
            break  # Success!
        except (JSONParseError, ValidationError) as e:
            if attempt < max_retries - 1:
                # Add error feedback to messages and retry
                messages.append({"role": "user", "content": f"Error: {e}..."})
            else:
                raise
    
    return board
```

#### 2.3 Agent Sequence and Detailed Execution Flows

The system uses **6 specialized agents** that execute in a fixed sequence. Each agent transforms one or more IRs into the next IR in the pipeline.

**Agent Execution Order**:
1. **ManagerAgent**: NL → RequirementIR
2. **ConceptualDesigner**: RequirementIR → ConceptualIR
3. **LogicalDesigner**: ConceptualIR + RequirementIR → LogicalIR
4. **DistributionEngineer**: LogicalIR + RequirementIR → GenerationIR
5. **WorkloadDesigner**: LogicalIR + RequirementIR → WorkloadIR
6. **QACompilerAgent**: All IRs → DatasetIR

**Complete Agent Execution Lifecycle**:

Each agent follows this detailed lifecycle:

```
┌─────────────────────────────────────────────────────────────────┐
│              Agent Execution Lifecycle                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 1: Prerequisite Check                              │  │
│  │   - Check if required IRs exist in blackboard            │  │
│  │   - If missing: log warning and return unchanged board  │  │
│  │   - If present: proceed to Phase 2                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 2: Prompt Loading                                   │  │
│  │   - Load system prompt: roles/{agent}_system.txt         │  │
│  │   - Load user prompt template: roles/{agent}_user.txt   │  │
│  │   - Both prompts are plain text files                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 3: Prompt Rendering                                │  │
│  │   - Serialize required IRs to JSON (indent=2)           │  │
│  │   - Render user template with IR data as placeholders    │  │
│  │   - Example: render_prompt(template,                     │  │
│  │              REQUIREMENT_JSON=ir.model_dump_json())     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 4: LLM Message Construction                        │  │
│  │   - Build messages list:                                 │  │
│  │     [                                                     │  │
│  │       {"role": "system", "content": sys_tmpl},         │  │
│  │       {"role": "user", "content": user_content}         │  │
│  │     ]                                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 5: LLM Call with Retry Logic                       │  │
│  │   - Attempt 1: Call chat(messages)                      │  │
│  │     ├─ Success: proceed to Phase 6                      │  │
│  │     └─ Failure: add error message to messages, retry    │  │
│  │   - Attempt 2: Call chat(messages) with error context   │  │
│  │     ├─ Success: proceed to Phase 6                      │  │
│  │     └─ Failure: raise exception                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 6: JSON Extraction                                  │  │
│  │   - Strategy 1: Extract from markdown code blocks        │  │
│  │     Pattern: ```json {...} ```                          │  │
│  │   - Strategy 2: Extract plain JSON object                │  │
│  │     Pattern: {...}                                       │  │
│  │   - Strategy 3: Parse entire text as JSON                │  │
│  │   - If all fail: raise JSONParseError                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 7: Data Preprocessing (Optional)                    │  │
│  │   - Fix common LLM mistakes (e.g., INT66 → INT)         │  │
│  │   - Convert None to empty dicts                          │  │
│  │   - Wrap lists in dicts if needed                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 8: IR Validation                                    │  │
│  │   - Validate JSON against Pydantic IR model             │  │
│  │   - Check required fields, types, constraints            │  │
│  │   - If invalid: add error to messages, retry (Phase 5)   │  │
│  │   - If valid: proceed to Phase 9                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 9: Blackboard Update                                │  │
│  │   - Write validated IR to blackboard                      │  │
│  │   - Example: board.logical_ir = LogicalIR(...)           │  │
│  │   - Log success message                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 10: Return Updated Blackboard                       │  │
│  │   - Return board with new IR written                      │  │
│  │   - Next agent in sequence will read this IR             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**ManagerAgent** (`agents/roles/manager.py`)

**Purpose**: Extract structured requirements from natural language.

**Input**: Natural language description (string)
**Output**: RequirementIR
**Dependencies**: None (first agent in sequence)

**Detailed Execution**:

```python
class ManagerAgent(BaseAgent):
    """Extracts structured RequirementIR from natural language input."""
    
    name = "manager"
    
    def __init__(self, nl_request: str):
        """Initialize with natural language description."""
        self.nl_request = nl_request
    
    def run(self, board: Blackboard) -> Blackboard:
        """
        Extract RequirementIR from natural language.
        
        Detailed Steps:
        1. Load prompts:
           - System: roles/manager_system.txt (role definition, JSON schema)
           - User: roles/manager_user.txt (template with {NARRATIVE} placeholder)
        
        2. Render user prompt:
           - Replace {NARRATIVE} with self.nl_request
        
        3. Build LLM messages:
           - System message: role definition and output format
           - User message: rendered template with NL description
        
        4. Call LLM with retry (max 2 attempts):
           - Attempt 1: Call chat(messages)
           - If JSONParseError: add "return only JSON" message, retry
           - If ValidationError: add error details, retry
           - Attempt 2: Call chat(messages) with error context
        
        5. Extract JSON:
           - Try markdown code blocks first
           - Fall back to plain JSON extraction
           - Fix common issues (None params → empty dict)
        
        6. Validate:
           - RequirementIR.model_validate(data)
           - Pydantic ensures all fields are correct types
        
        7. Write to blackboard:
           - board.requirement_ir = validated_ir
        
        8. Return updated board
        """
        logger.info("Manager agent: Extracting RequirementIR from NL")
        
        # Load prompts
        sys_tmpl = load_prompt("roles/manager_system.txt")
        usr_tmpl = load_prompt("roles/manager_user.txt")
        
        # Render user prompt
        user_content = render_prompt(usr_tmpl, NARRATIVE=self.nl_request)
        
        # Build messages
        messages = [
            {"role": "system", "content": sys_tmpl},
            {"role": "user", "content": user_content},
        ]
        
        # Retry logic (max 2 attempts)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Call LLM
                raw = chat(messages)
                
                # Extract JSON
                data = extract_json(raw)
                
                # Fix common LLM mistakes
                if isinstance(data, dict) and "distributions" in data:
                    for dist in data["distributions"]:
                        if isinstance(dist, dict) and dist.get("params") is None:
                            dist["params"] = {}
                
                # Validate as RequirementIR
                board.requirement_ir = RequirementIR.model_validate(data)
                
                # Success!
                break
                
            except JSONParseError as e:
                if attempt < max_retries - 1:
                    messages.append({
                        "role": "user",
                        "content": "Please return ONLY valid JSON, no markdown formatting."
                    })
                else:
                    raise
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    error_summary = str(e)[:200]
                    messages.append({
                        "role": "user",
                        "content": f"Validation error: {error_summary}. Please fix JSON structure."
                    })
                else:
                    raise
        
        return board
```

**Key Features**:
- **No Prerequisites**: First agent, doesn't check for existing IRs
- **NL Input**: Takes natural language string directly (not from blackboard)
- **Simple Output**: Single IR (RequirementIR)
- **Error Handling**: Retries with specific error messages

**ConceptualDesigner** (`agents/roles/conceptual_designer.py`)

**Purpose**: Design conceptual ER model from requirements.

**Input**: RequirementIR (from blackboard)
**Output**: ConceptualIR
**Dependencies**: Requires `board.requirement_ir` to exist

**Key Differences from ManagerAgent**:
- **Prerequisite Check**: Verifies `board.requirement_ir` exists before proceeding
- **Multiple IRs in Prompt**: Includes RequirementIR JSON in user prompt
- **Output Type**: ConceptualIR (entities, relationships, attributes)

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

#### 2.4 Prompt System (`prompts/loader.py`)

**Purpose**: File-based prompt management for easy iteration and version control.

**Architecture**:
Prompts are stored as plain text files in `prompts/roles/` directory. Each agent has two prompt files:
- **`{agent}_system.txt`**: System prompt defining the agent's role and output format
- **`{agent}_user.txt`**: User prompt template with placeholders for IR data

**Prompt Loading**:

```python
def load_prompt(path: str) -> str:
    """
    Load a prompt file from the prompts directory.
    
    Process:
    1. Resolve path relative to prompts/ directory
    2. Read file contents as UTF-8 text
    3. Return as string
    
    Args:
        path: Relative path, e.g., 'roles/manager_system.txt'
        
    Returns:
        Prompt file contents
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    full_path = PROMPTS_DIR / path
    return full_path.read_text(encoding="utf-8")
```

**Prompt Rendering**:

```python
def render_prompt(template: str, **kwargs: Any) -> str:
    """
    Render a prompt template with placeholders.
    
    Uses Python's str.format() for templating.
    Placeholders in templates use {PLACEHOLDER_NAME} format.
    
    Example:
        template = "RequirementIR:\n\n{REQUIREMENT_JSON}"
        rendered = render_prompt(template, REQUIREMENT_JSON=ir_json)
    
    Args:
        template: Template string with {PLACEHOLDER} placeholders
        **kwargs: Values to fill placeholders
        
    Returns:
        Rendered prompt string
        
    Raises:
        KeyError: If placeholder is missing from kwargs
    """
    return template.format(**kwargs)
```

**Prompt File Structure**:

**System Prompt** (`roles/{agent}_system.txt`):
- Defines agent's role and responsibilities
- Specifies output JSON schema
- Provides guidelines and constraints
- Example: "You are the Logical Designer agent. You must return a JSON object matching this structure: {...}"

**User Prompt Template** (`roles/{agent}_user.txt`):
- Contains placeholders for IR data: `{REQUIREMENT_JSON}`, `{CONCEPTUAL_JSON}`, etc.
- Provides context about what to transform
- Example: "ConceptualIR:\n\n{CONCEPTUAL_JSON}\n\nRequirementIR:\n\n{REQUIREMENT_JSON}\n\nDesign a logical relational schema..."

**Benefits**:
- **Easy Iteration**: Edit prompts without code changes
- **Version Control**: Track prompt changes in git
- **A/B Testing**: Try different prompt variations
- **Non-Developer Friendly**: Domain experts can edit prompts
- **Separation of Concerns**: Logic in code, instructions in prompts

#### 2.5 LLM Call Flow (`agents/tools/llm_client.py`)

**Purpose**: Unified interface for multiple LLM providers with automatic fallback.

**Provider Priority**:
1. **OpenAI** (if `OPENAI_API_KEY` and `MODEL_NAME` set)
2. **Local API** (if `LLM_URL` and `MODEL` set) - OpenAI-compatible
3. **Gemini** (if `GEMINI_API_KEY` and `GEMINI_MODEL` set)

**Unified Interface**:

```python
def chat(messages: List[Dict[str, str]]) -> str:
    """
    Unified interface for multiple LLM providers.
    
    Process:
    1. Check environment variables to determine provider
    2. Route to appropriate provider function
    3. Return response text
    
    Args:
        messages: List of message dicts with "role" and "content"
                 Format: [{"role": "system", "content": "..."}, ...]
    
    Returns:
        LLM response text (may contain JSON, markdown, etc.)
    
    Raises:
        RuntimeError: If no provider is configured
        API errors: Provider-specific exceptions
    """
    if use_openai:
        return _chat_openai(messages)
    elif use_local:
        return _chat_local(messages)
    else:
        return _chat_gemini(messages)
```

**Provider-Specific Details**:

**OpenAI Provider**:
- Uses official `openai` Python SDK
- Standard chat completion API
- Temperature control via settings
- Error handling for rate limits

**Local Provider** (OpenAI-compatible):
- Supports local LLM servers (Ollama, vLLM, etc.)
- Custom base URL configuration
- **Retry Logic**: Handles "Model reloaded" and 503 errors
- 10-minute timeout for slow local models
- Exponential backoff on transient errors

**Gemini Provider**:
- Uses Google Generative AI SDK
- Combines system + user prompts (Gemini format)
- Handles response parts (text or parts array)
- Error handling for quota limits

**Retry Logic** (`agents/tools/retry.py`):

```python
def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout_errors: List[Type[Exception]] = None,
    operation_name: str = "operation"
) -> Any:
    """
    Retry function with exponential backoff.
    
    Handles transient errors (network issues, rate limits, model reloads).
    
    Process:
    1. Call func()
    2. If transient error and retries remaining:
       - Wait: base_delay * (2 ** attempt)
       - Retry
    3. If max_retries exceeded: raise exception
    
    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        timeout_errors: List of exception types to retry on
        operation_name: Name for logging
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if is_transient_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise
```

#### 2.6 JSON Extraction (`agents/tools/json_parser.py`)

**Purpose**: Robust extraction of JSON from LLM responses that may contain markdown, explanations, or formatting.

**Problem**: LLMs often wrap JSON in markdown code blocks or add explanatory text:
```
Here's the JSON:

```json
{
  "tables": {...}
}
```

This represents the schema...
```

**Solution**: Multiple extraction strategies with fallback:

```python
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM output using multiple strategies.
    
    Strategies (in order):
    1. Markdown code blocks: ```json {...} ``` or ``` {...} ```
    2. Plain JSON object: {...} (first complete JSON object)
    3. Entire text as JSON
    
    Args:
        text: LLM response text (may contain markdown, explanations, etc.)
        
    Returns:
        Parsed JSON as dict
        
    Raises:
        JSONParseError: If no valid JSON found
    """
    # Strategy 1: Markdown code blocks
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Plain JSON object (first complete object)
    json_obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
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
    
    raise JSONParseError("Could not extract valid JSON from LLM response")
```

**Key Features**:
- **Multiple Strategies**: Handles various LLM output formats
- **Regex-Based**: Efficient extraction without full parsing
- **Error Handling**: Clear error messages with context
- **Logging**: Debug information for troubleshooting

#### 2.7 Validation System (`ir/validators.py`)

**Purpose**: Structured validation of IRs with detailed issue reporting.

**QaIssue Model**:

```python
@dataclass
class QaIssue:
    """QA issue found during validation."""
    
    stage: Literal["LogicalIR", "GenerationIR", "PostGen"]
    code: str  # e.g., "MISSING_PK", "FK_REF_INVALID"
    location: str  # e.g., "table_name" or "table_name.column_name"
    message: str  # Human-readable error message
    details: dict = field(default_factory=dict)  # Additional context
```

**Validation Functions**:

**1. `validate_logical(ir: DatasetIR) -> List[QaIssue]`**:
- Checks primary keys exist and reference valid columns
- Validates foreign keys reference existing tables/columns
- Returns list of QaIssue objects (empty if validation passes)

**2. `validate_generation(ir: DatasetIR) -> List[QaIssue]`**:
- Ensures all generation specs reference existing tables/columns
- Checks for orphaned specs
- Validates distribution parameters

**3. `validate_derived_columns(ir: DatasetIR) -> List[QaIssue]`**:
- Validates derived column expressions
- Checks dependencies exist
- Detects circular dependencies

**4. `collect_issues(validators: List[Callable], board: Blackboard) -> List[QaIssue]`**:
- Runs multiple validators on blackboard
- Collects all issues into single list
- Used by repair loop

**Issue Codes**:
- `MISSING_PK`: Table missing primary key
- `PK_COL_MISSING`: Primary key column doesn't exist
- `FK_REF_TABLE_MISSING`: Foreign key references missing table
- `FK_COL_MISSING`: Foreign key column doesn't exist
- `FK_REF_COL_MISSING`: Foreign key references missing column
- `GEN_TABLE_MISSING`: Generation spec references unknown table
- `GEN_COL_MISSING`: Generation spec references unknown column
- And many more...

**Validation Process**:

```python
def validate_logical(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate logical schema constraints.
    
    Process:
    1. Iterate through all tables in LogicalIR
    2. For each table:
       - Check primary key exists
       - Check primary key columns exist
       - Check foreign keys reference valid tables/columns
    3. Collect all issues into list
    4. Return list (empty if validation passes)
    
    Returns:
        List of QaIssue objects
    """
    issues: List[QaIssue] = []
    
    for table_name, table in ir.logical.tables.items():
        # Check primary key
        if not table.primary_key:
            issues.append(QaIssue(
                stage="LogicalIR",
                code="MISSING_PK",
                location=table_name,
                message=f"{table_name}: missing primary key",
                details={"table": table_name}
            ))
        
        # Check foreign keys
        for fk in table.foreign_keys:
            if fk.ref_table not in ir.logical.tables:
                issues.append(QaIssue(
                    stage="LogicalIR",
                    code="FK_REF_TABLE_MISSING",
                    location=f"{table_name}.{fk.column}",
                    message=f"FK '{fk.column}' references missing table '{fk.ref_table}'",
                    details={"table": table_name, "fk_column": fk.column}
                ))
    
    return issues
```

#### 2.8 Agent Runner with Repair Loop (`agents/runner.py`)

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

#### 2.9 Orchestrator (`agents/orchestrator.py`)

**Purpose**: Coordinates sequential execution of all agents in the pipeline.

**Architecture**:
The Orchestrator is responsible for:
1. **Agent Sequencing**: Executes agents in the correct order
2. **State Management**: Passes blackboard between agents
3. **Error Handling**: Catches and logs agent failures
4. **Progress Tracking**: Logs execution progress

**Complete Implementation**:

```python
class Orchestrator:
    """Orchestrates the execution of multiple agents in sequence with repair loops and metrics tracking."""
    
    def __init__(
        self,
        agents: List[BaseAgent],
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        enable_repair: bool = True,
        enable_metrics: bool = True,
    ):
        """
        Initialize orchestrator with a list of agents.
        
        Args:
            agents: List of agents to execute in order
            query_id: Optional query ID for metrics tracking
            query_text: Optional query text for metrics tracking
            enable_repair: Whether to use repair loop system (default: True)
            enable_metrics: Whether to track quality metrics (default: True)
        """
        self.agents = agents
        self.query_id = query_id
        self.query_text = query_text
        self.enable_repair = enable_repair
        self.enable_metrics = enable_metrics
        logger.info(f"Initialized orchestrator with {len(agents)} agents")
    
    def execute(self, board: Blackboard) -> Blackboard:
        """
        Execute all agents in sequence with repair loops and metrics tracking.
        
        Process:
        1. Start metrics tracking (if enabled)
        2. Start with initial blackboard (empty or pre-populated)
        3. For each agent in sequence:
           a. Start agent metrics tracking
           b. If repair enabled: Use run_with_repair() with appropriate validators
           c. Else: Call agent.run() directly (legacy mode)
           d. End agent metrics tracking (success/failure)
           e. Log agent completion
        4. Return final blackboard with all IRs populated
        
        Args:
            board: Initial blackboard state
            
        Returns:
            Final blackboard state after all agents execute
            
        Raises:
            Exception: If any agent fails (stops pipeline)
        """
        logger.info("Starting orchestrator execution")
        
        # Start metrics tracking if enabled
        collector = None
        if self.enable_metrics:
            # Only enable metrics if we have query information
            # Without query_id or query_text, metrics cannot be properly tracked
            if self.query_id or self.query_text:
                collector = get_metrics_collector()
                if self.query_id and self.query_text:
                    collector.start_query(self.query_id, self.query_text)
                elif self.query_text:
                    # Generate query ID from text hash if not provided
                    import hashlib
                    query_id = hashlib.md5(self.query_text.encode()).hexdigest()[:8]
                    collector.start_query(query_id, self.query_text)
                elif self.query_id:
                    # Only query_id provided, use empty string for query_text
                    collector.start_query(self.query_id, "")
            else:
                logger.warning(
                    "Metrics tracking is enabled but both query_id and query_text are None. "
                    "Metrics will not be tracked. Provide at least query_text to enable metrics."
                )
        
        current_board = board

        for i, agent in enumerate(self.agents, 1):
            logger.info(f"Executing agent {i}/{len(self.agents)}: {agent.name}")
            
            # Start agent metrics tracking
            if collector:
                collector.start_agent(agent.name)
            
            try:
                if self.enable_repair:
                    # Determine validators based on agent type
                    validators = self._get_validators_for_agent(agent)
                    
                    # Only use repair loop if agent implements _repair() and validators are assigned
                    # Check if agent's class overrides _repair() (different from BaseAgent's default no-op)
                    agent_class = type(agent)
                    agent_repair_method = getattr(agent_class, '_repair', None)
                    base_repair_method = getattr(BaseAgent, '_repair', None)
                    can_repair = agent_repair_method is not None and agent_repair_method is not base_repair_method
                    
                    if validators and can_repair:
                        # Use repair loop system
                        current_board = run_with_repair(
                            agent,
                            current_board,
                            validators,
                            max_retries=2,
                        )
                    elif validators and not can_repair:
                        # Validators assigned but agent can't repair - log warning and run without repair loop
                        logger.warning(
                            f"Agent {agent.name} has validators assigned but does not implement _repair(). "
                            f"Running without repair loop. Validation issues will not be automatically fixed."
                        )
                        current_board = agent.run(current_board)
                    else:
                        # No validators - direct execution
                        current_board = agent.run(current_board)
                else:
                    # Direct execution (legacy mode)
                    current_board = agent.run(current_board)
                
                # End agent metrics tracking (success)
                if collector:
                    collector.end_agent(agent.name, success=True)
                
                logger.info(f"Agent {agent.name} completed successfully")
            except Exception as e:
                # End agent metrics tracking (failure)
                if collector:
                    collector.end_agent(agent.name, success=False, error_message=str(e))
                logger.error(f"Agent {agent.name} failed: {e}", exc_info=True)
                raise

        logger.info("Orchestrator execution completed")
        return current_board
    
    def _get_validators_for_agent(self, agent: BaseAgent) -> List:
        """
        Get appropriate validators for an agent based on its type.
        
        Returns:
            List of validator functions (validate_logical_blackboard, validate_generation_blackboard, etc.)
        """
        validators = []
        
        # Logical Designer needs logical validation
        if agent.name == "logical_designer":
            validators.append(validate_logical_blackboard)
        
        # Distribution Engineer needs generation validation
        if agent.name == "dist_engineer":
            validators.append(validate_generation_blackboard)
        
        # QA Compiler validates everything (but doesn't need repair loop)
        # Other agents don't need validation (they produce IRs that are validated later)
        
        return validators
```

**Execution Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Execution                        │
│                                                                  │
│  Initial Blackboard (empty)                                      │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 1: ManagerAgent                                     │  │
│  │   Input: NL description (from constructor)                │  │
│  │   Output: board.requirement_ir                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 2: ConceptualDesigner                              │  │
│  │   Input: board.requirement_ir                            │  │
│  │   Output: board.conceptual_ir                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 3: LogicalDesigner                                 │  │
│  │   Input: board.requirement_ir + board.conceptual_ir      │  │
│  │   Output: board.logical_ir                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 4: DistributionEngineer                            │  │
│  │   Input: board.requirement_ir + board.logical_ir         │  │
│  │   Output: board.generation_ir                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 5: WorkloadDesigner                                 │  │
│  │   Input: board.requirement_ir + board.logical_ir         │  │
│  │   Output: board.workload_ir                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Agent 6: QACompilerAgent                                 │  │
│  │   Input: All IRs from blackboard                         │  │
│  │   Output: board.dataset_ir (combined IR)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  Final Blackboard (all IRs populated)                            │
└─────────────────────────────────────────────────────────────────┘
```

**Error Handling**:
- If any agent fails, the orchestrator:
  1. Logs the error with full traceback
  2. Raises the exception (stops pipeline)
  3. Blackboard state is preserved up to the point of failure

**Key Features**:

1. **Smart Repair Loop Detection**: 
   - Only uses repair loop if agent implements `_repair()` method
   - Checks if agent's class overrides the base `_repair()` method
   - Falls back to direct execution if agent can't repair
   - Logs warnings when validators are assigned but agent can't repair

2. **Conditional Validator Assignment**:
   - `logical_designer`: Gets `validate_logical_blackboard` validator
   - `dist_engineer`: Gets `validate_generation_blackboard` validator
   - Other agents: No validators (they produce IRs validated later)

3. **Metrics Tracking**:
   - Requires `query_id` or `query_text` to enable metrics
   - Auto-generates query ID from text hash if only `query_text` provided
   - Warns if metrics enabled but no query information provided
   - Tracks agent execution time and success/failure

4. **Flexible Execution Modes**:
   - **Repair Mode** (`enable_repair=True`): Uses repair loops for agents that support it
   - **Legacy Mode** (`enable_repair=False`): Direct execution without repair loops

**Repair Loop Decision Logic**:

The orchestrator intelligently determines whether to use repair loops:

1. **Check if repair is enabled** (`enable_repair=True`)
2. **Get validators for agent** (based on agent type via `_get_validators_for_agent()`)
3. **Check if agent can repair** (implements `_repair()` method):
   - Compares agent's `_repair` method with BaseAgent's default no-op
   - Only uses repair if agent actually overrides the method
4. **Execute accordingly**:
   - If validators AND can repair → Use `run_with_repair()` with validators
   - If validators BUT can't repair → Log warning, run directly (validation issues won't be fixed)
   - If no validators → Run directly

This ensures repair loops are only used when they can actually help, avoiding unnecessary overhead and ensuring agents without repair capabilities still execute correctly.

**Usage**:

```python
from nl2data.agents.orchestrator import Orchestrator
from nl2data.agents.base import Blackboard
from nl2data.utils.agent_factory import create_agent_list
import hashlib

# Create agent sequence
nl_description = "Generate a retail sales dataset..."
agents = create_agent_list(nl_description)

# Generate query ID for metrics tracking
query_id = hashlib.md5(nl_description.encode()).hexdigest()[:8]

# Create orchestrator with repair loops and metrics tracking enabled
orchestrator = Orchestrator(
    agents,
    query_id=query_id,
    query_text=nl_description,
    enable_repair=True,  # Use repair loops (default)
    enable_metrics=True  # Track quality metrics (default)
)

# Execute pipeline
initial_board = Blackboard()
final_board = orchestrator.execute(initial_board)

# Access final IR
dataset_ir = final_board.dataset_ir
```

#### 2.10 Agent Factory (`utils/agent_factory.py`)

**Purpose**: Factory functions for creating standardized agent sequences.

**Functions**:

```python
def create_agent_sequence(nl_description: str) -> List[Tuple[str, BaseAgent]]:
    """
    Create the standard agent sequence for NL → IR pipeline.
    
    Returns list of (agent_name, agent_instance) tuples.
    Useful for UI/CLI that need agent names for display.
    
    Args:
        nl_description: Natural language description of the dataset
        
    Returns:
        List of tuples: [("manager", ManagerAgent(...)), ...]
    """
    return [
        ("manager", ManagerAgent(nl_description)),
        ("conceptual_designer", ConceptualDesigner()),
        ("logical_designer", LogicalDesigner()),
        ("dist_engineer", DistributionEngineer()),
        ("workload_designer", WorkloadDesigner()),
        ("qa_compiler", QACompilerAgent()),
    ]


def create_agent_list(nl_description: str) -> List[BaseAgent]:
    """
    Create the standard agent list (without names) for use with Orchestrator.
    
    Args:
        nl_description: Natural language description of the dataset
        
    Returns:
        List of agent instances: [ManagerAgent(...), ConceptualDesigner(...), ...]
    """
    return [
        ManagerAgent(nl_description),
        ConceptualDesigner(),
        LogicalDesigner(),
        DistributionEngineer(),
        WorkloadDesigner(),
        QACompilerAgent(),
    ]
```

**Benefits**:
- **Standardization**: Ensures consistent agent order
- **Convenience**: Single function call to create entire pipeline
- **Maintainability**: Change agent order in one place
- **Flexibility**: Can create custom sequences by modifying factory

#### 2.11 Multi-Agent Framework Summary

**Complete Data Flow**:

```
Natural Language Input
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Pipeline                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. ManagerAgent                                          │  │
│  │    - Load prompts                                        │  │
│  │    - Call LLM with NL                                    │  │
│  │    - Extract & validate JSON                             │  │
│  │    - Write RequirementIR to blackboard                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. ConceptualDesigner                                    │  │
│  │    - Read RequirementIR from blackboard                 │  │
│  │    - Load prompts                                        │  │
│  │    - Render prompt with RequirementIR JSON              │  │
│  │    - Call LLM                                            │  │
│  │    - Extract & validate JSON                             │  │
│  │    - Write ConceptualIR to blackboard                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. LogicalDesigner                                       │  │
│  │    - Read RequirementIR + ConceptualIR                  │  │
│  │    - Load prompts                                        │  │
│  │    - Render prompt with both IRs                        │  │
│  │    - Call LLM                                            │  │
│  │    - Extract & validate JSON                             │  │
│  │    - Preprocess (fix common mistakes)                    │  │
│  │    - Write LogicalIR to blackboard                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. DistributionEngineer                                  │  │
│  │    - Read RequirementIR + LogicalIR                     │  │
│  │    - Load prompts                                        │  │
│  │    - Render prompt with IRs                             │  │
│  │    - Call LLM                                            │  │
│  │    - Extract & validate JSON                             │  │
│  │    - Write GenerationIR to blackboard                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. WorkloadDesigner                                      │  │
│  │    - Read RequirementIR + LogicalIR                     │  │
│  │    - Load prompts                                        │  │
│  │    - Render prompt with IRs                             │  │
│  │    - Call LLM                                            │  │
│  │    - Extract & validate JSON                             │  │
│  │    - Postprocess (wrap lists if needed)                   │  │
│  │    - Write WorkloadIR to blackboard                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. QACompilerAgent                                       │  │
│  │    - Read all IRs from blackboard                       │  │
│  │    - Combine into DatasetIR                              │  │
│  │    - Run validators (validate_logical, etc.)             │  │
│  │    - Attempt automatic repair if issues found            │  │
│  │    - Write DatasetIR to blackboard                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│                    Final Blackboard                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - requirement_ir: RequirementIR                           │  │
│  │ - conceptual_ir: ConceptualIR                             │  │
│  │ - logical_ir: LogicalIR                                   │  │
│  │ - generation_ir: GenerationIR                             │  │
│  │ - workload_ir: WorkloadIR                                │  │
│  │ - dataset_ir: DatasetIR (combined)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    DatasetIR (ready for data generation)
```

**Key Design Principles**:

1. **Separation of Concerns**: Each agent has a single, well-defined responsibility
2. **Progressive Refinement**: Each IR builds on previous IRs
3. **Type Safety**: Pydantic validation ensures correctness at each stage
4. **Error Resilience**: Retry logic and validation catch issues early
5. **Extensibility**: Easy to add new agents or modify existing ones
6. **Testability**: Can test agents in isolation with mock blackboards
7. **Observability**: Comprehensive logging at each stage

**Agent Communication Pattern**:

- **Read-Only Access**: Agents read previous IRs but never modify them
- **Write-Once**: Each agent writes exactly one IR to the blackboard
- **Sequential Dependencies**: Clear data flow - no circular dependencies
- **Optional IRs**: Some IRs (workload_ir) are optional and may be None

**Error Handling Strategy**:

- **Per-Agent Retries**: Each agent retries LLM calls on JSON/validation errors
- **Validation Feedback**: QaIssue objects provide structured error information
- **Repair Loops**: Optional repair loops can fix validation issues automatically
- **Fail-Fast**: Pipeline stops on unrecoverable errors (after all retries)

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

**Purpose**: Generates fact table data in streaming chunks with multi-phase processing.

**Key Features**:
- **Streaming Generation**: Yields data in chunks for memory efficiency
- **Multi-Phase Processing**: Base columns → Events → Dimension Joins → Constraints → Derived → Windows
- **Constraint Enforcement**: Applies FDs, implications, and nullability constraints (✅ Integrated as Phase 1.7)
- **Window Function Support**: Full table materialization for window operations

**Generation Phases**:

The fact generator uses a multi-phase approach for efficient streaming generation:

**Phase 1: Generate Base Columns**
- Sample all non-derived, non-window columns using distribution samplers
- Handle foreign keys with referential integrity
- Apply type enforcement

**Phase 1.5: Apply Event Effects**
- Apply global event effects (e.g., incidents, storms) to base columns
- Modify distributions based on active events

**Phase 1.6: Join Dimension Tables**
- Left join dimension tables for derived column lookups
- Allows derived expressions to reference dimension attributes

**Phase 1.7: Enforce Constraints** ✅ **NEW**
- Apply functional dependencies (FDs)
- Apply implication constraints
- Enforce nullability (fill nulls in non-nullable columns)

**Phase 2: Compute Derived Columns**
- Evaluate derived expressions in dependency order
- Uses vectorized DataFrame operations

**Phase 3: Compute Window Columns** (if needed)
- Full table materialization for window functions
- Compute rolling aggregations, lag/lead, etc.
- Split back into chunks for streaming

**Note**: DerivedSampler has been removed. Derived columns are now handled exclusively by the derived expression engine (derived_program.py, derived_eval.py).

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
- `<`, `<=`, `>`, `>=`, `==`, `!=`, `in`, `not in` (membership testing)

**Conditional Logic**:
- `where(condition, value_if_true, value_if_false)`: Vectorized conditional
- `value_if_true if condition else value_if_false`: Ternary expression
- `case_when(cond1, val1, cond2, val2, ..., default)`: Multi-condition macro (evaluates to nested `where()` calls)

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

**Type Casting Functions**:
- `int(x)`: Convert to integer
- `float(x)`: Convert to float
- `bool(x)`: Convert to boolean
- `str(x)`: Convert to string

**Distribution Functions** (for random number generation):
- `normal(mean, std)`: Generate normal distribution samples
- `lognormal(mean, sigma)`: Generate log-normal distribution samples
- `pareto(alpha)`: Generate Pareto distribution samples

**String Operations**:
- `concat(str1, str2, ...)`: Concatenate multiple strings
- `format(template, *args)`: Format string template with values (supports `{0}`, `{1}` style)
- `substring(s, start, length=None)`: Extract substring from string(s)

**Helper Functions**:
- `between(x, a, b)`: Check if x is between a and b (inclusive), equivalent to `x >= a and x <= b`
- `geo_distance(lat1, lon1, lat2, lon2)`: Calculate great-circle distance in kilometers using Haversine formula
- `ts_diff(unit, t1, t2)`: Calculate time difference between t1 and t2 in specified unit ('seconds', 'minutes', 'hours', 'days')
- `overlap_days(start1, end1, start2, end2)`: Calculate number of overlapping days between two date intervals

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

**Window Functions**:

The system supports window operations as first-class IR nodes (`DistWindow`), enabling rolling aggregations and time-series analysis:

- **Window Types**:
  - `RANGE`: Time-based windows (e.g., "7d", "24h")
  - `ROWS`: Row-based windows (e.g., "100", "50")

- **Supported Operations**:
  - `mean(column)`, `sum(column)`, `count(*)`, `std(column)`: Aggregation functions
  - `lag(column, n=1)`: Access previous row values
  - `lead(column, n=1)`: Access next row values

- **Partitioning**: Window functions can be partitioned by one or more columns
- **Ordering**: Must specify an `order_by` column (typically a timestamp)

Example:
```json
{
  "kind": "window",
  "expression": "mean(latency_p95)",
  "partition_by": ["tenant_id"],
  "order_by": "timestamp",
  "frame": {
    "type": "RANGE",
    "preceding": "7d"
  }
}
```

**Event/Incident System**:

Global events can affect data generation across multiple tables and columns:

- **Event Specification**: Events have a name, time range (start/end), and effects
- **Effect Types**:
  - `multiply_distribution`: Multiply column values by a factor
  - `add_offset`: Add a constant offset to column values
  - `set_value`: Set column values to a specific value (with optional probability)

Example:
```json
{
  "events": [{
    "name": "winter_storm",
    "start_time": "2024-01-10T00:00:00Z",
    "end_time": "2024-01-13T00:00:00Z",
    "effects": [{
      "table": "fact_shipment",
      "column": "lead_time_days",
      "effect_type": "multiply_distribution",
      "value": 1.7
    }]
  }]
}
```

**Validation**:

The system validates derived columns at multiple stages:

1. **Compile-time**: Expression syntax and allowed functions are checked
2. **IR Validation**: Dependencies are verified to exist in the schema
3. **Runtime**: Missing columns or evaluation errors provide helpful error messages
4. **Nuance Coverage**: Automated lint checks that NL requirements are reflected in IR constructs

**Implementation Details**:

**Dependency Tracking** (`generation/derived_registry.py`):

```python
class DerivedRegistry:
    """Registry of compiled derived expressions and window columns."""
    programs: Dict[DerivedKey, DerivedProgram]  # (table, col) → compiled program
    windows: Dict[DerivedKey, DistWindow]       # (table, col) → window specification
    order: Dict[str, List[str]]                 # table → [cols in topological order]
    window_order: Dict[str, List[str]]          # table → [window cols in order]

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

#### 3.7 Window Function Evaluation (`generation/window_eval.py`)

**Window Operations**:

The system supports window functions as first-class IR constructs for rolling aggregations and time-series analysis:

- **Window Frame Types**:
  - `RANGE`: Time-based windows (e.g., "7d", "24h", "30m")
  - `ROWS`: Row-based windows (e.g., "100", "50")

- **Supported Aggregations**:
  - `mean(column)`: Rolling mean
  - `sum(column)`: Rolling sum
  - `count(*)`: Rolling count
  - `std(column)`: Rolling standard deviation
  - `lag(column, n=1)`: Access previous row values
  - `lead(column, n=1)`: Access next row values

- **Partitioning**: Window functions can be partitioned by one or more columns
- **Ordering**: Must specify an `order_by` column (typically a timestamp)

**Implementation**:

```python
def eval_window_expression(
    df: pd.DataFrame,
    window_spec: DistWindow,
    expr_col: Optional[str] = None
) -> pd.Series:
    """
    Evaluate window function on DataFrame.
    
    Process:
    1. Parse window size (time-based or row-based)
    2. Sort DataFrame by order_by column
    3. Apply partitioning if specified
    4. Compute rolling aggregation or lag/lead
    5. Restore original row order
    """
    # For RANGE windows: use pandas rolling with time window
    # For ROWS windows: use pandas rolling with integer window
    # For lag/lead: use pandas shift()
```

**Memory Management**:

Tables with window columns require full materialization (all rows must be generated before computing windows). The generator automatically detects window columns and switches from streaming to batch mode for those tables.

#### 3.8 Event System (`generation/event_eval.py`)

**Event Effects**:

Global events can causally affect data generation across multiple tables:

- **Event Specification**: Events have a name, time range, and list of effects
- **Time Parsing**: Supports ISO datetime strings, relative percentages ("50%"), and row numbers
- **Effect Types**:
  - `multiply_distribution`: Multiply column values by a factor (e.g., 1.7 for 70% increase)
  - `add_offset`: Add constant offset to column values
  - `set_value`: Set column values to specific value (with optional probability)

**Application**:

Events are applied after base column generation but before derived columns, allowing derived expressions to use modified values. Effects are applied per-chunk during streaming generation.

Example use cases:
- Weather events affecting shipment delays
- System incidents affecting latency metrics
- Marketing campaigns affecting transaction volumes

#### 3.9 Value Providers (`generation/providers/`)

**Provider System Architecture**:

The provider system generates realistic data values using external libraries (Faker, Mimesis) or lookup datasets (geographic data):

- **Provider Protocol**: All providers implement `ValueProvider` protocol
- **Registry System**: Centralized registry maps provider names to factory functions
- **Automatic Assignment**: Heuristic-based assignment of providers to columns based on name patterns

**Supported Providers**:

1. **Faker Providers** (`faker_provider.py`):
   - `faker.name`, `faker.email`, `faker.phone_number`
   - `faker.address`, `faker.city`, `faker.country`
   - `faker.company`, `faker.job`
   - `faker.date`, `faker.date_time`

2. **Mimesis Providers** (`mimesis_provider.py`):
   - `mimesis.full_name`, `mimesis.email`, `mimesis.telephone`
   - `mimesis.address`

3. **Geo Lookup Providers** (`lookup_geo.py`):
   - `lookup.city`: Real city names from GeoNames dataset
   - `lookup.country`: Real country names from GeoNames dataset
   - Uses Parquet files for efficient lookups

**Provider Assignment** (`providers/assign.py`):

The system automatically assigns providers to columns based on name patterns:

```python
def assign_default_providers(board: Blackboard) -> Blackboard:
    """
    Heuristically assign providers to columns based on name patterns.
    
    Patterns:
    - "email" → faker.email
    - "phone" → faker.phone_number
    - "name" (person names) → faker.name
    - "city" → lookup.city
    - etc.
    """
```

**Provider Registry** (`providers/registry.py`):

```python
PROVIDERS = {
    "faker.email": lambda cfg: FakerProvider(field="email", **cfg),
    "faker.name": lambda cfg: FakerProvider(field="name", **cfg),
    "lookup.city": lambda cfg: GeoLookupProvider(dataset="geonames.cities", **cfg),
    # ... more providers
}

def get_provider(name: str, config: Dict[str, Any]) -> ValueProvider:
    """Get provider instance by name."""
```

#### 3.10 Memory-Safe FK Allocation (`generation/allocator.py`)

**Guaranteed Coverage**:

The FK allocation system ensures referential integrity while maintaining Zipf skew:

- **Coverage Guarantee**: Every primary key gets at least one foreign key reference
- **Zipf Skew**: Remaining rows follow Zipf distribution
- **Memory Efficiency**: Streaming allocation without materializing large arrays

**Implementation**:

```python
def fk_assignments(
    pk_ids: np.ndarray,
    n_rows: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    batch: int = 5_000_000,
) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Generate FK assignments with guaranteed coverage.
    
    Process:
    1. Allocate one FK per PK (guaranteed coverage)
    2. Allocate remaining rows by Zipf probabilities
    3. Stream results in batches to avoid memory issues
    """
```

**Alpha Clipping**:

```python
def clip_alpha_for_max_share(K: int, max_top1_share: float) -> float:
    """
    Find maximum alpha such that top item probability <= max_top1_share.
    Uses binary search for efficient computation.
    """
```

#### 3.11 Additional Generation Utilities

**Type Enforcement** (`generation/type_enforcement.py`):
- Enforces SQL types after sampling
- Handles type coercion and validation

**Uniqueness Enforcement** (`generation/uniqueness.py`):
- Pattern-based detection of category columns
- Uniqueness enforcement for categorical columns

**IR Helpers** (`generation/ir_helpers.py`):
- Builds distribution maps from IR
- Builds provider maps from IR
- Utility functions for IR extraction

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

#### 3.12 Distribution Samplers

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

The evaluation framework validates generated data against IR specifications across multiple dimensions. The framework has been restructured into a modular architecture supporting both single-table and multi-table evaluation:

- **Schema Validation**: Primary key and foreign key integrity
- **Statistical Alignment**: Distribution fitting (Zipf, categorical, numeric)
- **Workload Performance**: Query execution time and correctness
- **Relational Metrics**: Join selectivity, FK coverage, degree histograms
- **Schema Coverage**: Comparison with gold/reference schemas using enhanced matching algorithms
- **Table-Level Fidelity**: Marginal distributions and correlations
- **Multi-Table Evaluation**: Comprehensive schema matching, quality assessment, and utility scoring
- **SD Metrics Integration**: Statistical quality evaluation using SD Metrics library

**New Modular Structure**:
- `evaluation/evaluators/`: Single-table and multi-table evaluators
- `evaluation/models/`: Data models for evaluation reports
- `evaluation/metrics/`: Schema, table, and relational metrics
- `evaluation/matching/`: Enhanced schema matching algorithms
- `evaluation/quality/`: SD Metrics quality evaluation
- `evaluation/execution/`: Workload execution and statistical tests
- `evaluation/aggregation/`: Score aggregation utilities
- `evaluation/utils/`: Helper utilities (normalization, FD utilities)

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
    """Complete evaluation configuration for single-table evaluation."""
    thresholds: EvalThresholds = EvalThresholds()

# Multi-table evaluation configuration

class SchemaMatchingConfig(BaseModel):
    """Configuration for enhanced schema matching."""
    # Column-level similarity weights
    w_name_col: float = 0.1
    w_range_col: float = 0.40
    w_role_col: float = 0.30
    w_FD_col: float = 0.20
    
    # Table-level similarity weights
    alpha: float = 0.1  # Table-name similarity
    beta: float = 0.6  # Column-alignment strength
    gamma: float = 0.1  # Cardinality similarity
    delta: float = 0.1  # FD signature similarity
    
    # Matching thresholds
    tau_table: float = 0.5  # Table matching threshold
    tau_col: float = 0.6  # Column matching threshold

class MultiTableEvalConfig(BaseModel):
    """Complete configuration for multi-table evaluation."""
    matching: SchemaMatchingConfig = SchemaMatchingConfig()
    coverage: CoverageConfig = CoverageConfig()
    structure: StructureConfig = StructureConfig()
    utility: UtilityConfig = UtilityConfig()
    global_score: GlobalScoreConfig = GlobalScoreConfig()
    quality: QualityEvaluationConfig = QualityEvaluationConfig()
```

#### 4.2 Evaluation Report Models (`evaluation/models/`)

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
    """Complete evaluation report for single-table evaluation."""
    schema: List[TableReport] = Field(default_factory=list)
    columns: List[ColumnReport] = Field(default_factory=list)
    workloads: List[WorkloadReport] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False

# Multi-table evaluation models

class SchemaMatchResult(BaseModel):
    """Complete schema matching result."""
    table_matches: List[TableMatch]
    column_matches: Dict[str, List[ColumnMatch]]  # table_name -> [ColumnMatch]
    unmatched_real_tables: List[str]
    unmatched_synth_tables: List[str]
    unmatched_real_columns: Dict[str, List[str]]
    unmatched_synth_columns: Dict[str, List[str]]
    table_coverage: float  # C_T
    column_coverage: Dict[str, float]  # table_name -> C_k
    quality_scores: Optional[Dict[str, QualityScore]] = None

class MultiTableEvaluationReport(BaseModel):
    """Complete multi-table evaluation report."""
    schema_match: SchemaMatchResult
    schema_score: float
    structure_scores: Dict[str, float]
    utility_scores: Dict[str, float]
    global_score: Optional[float] = None
```

#### 4.3 Evaluators (`evaluation/evaluators/`)

**Single-Table Evaluator** (`evaluators/single_table.py`):
- Evaluates individual tables against IR specifications
- Validates schema, distributions, and workloads
- Returns `EvaluationReport` with detailed metrics

**Multi-Table Evaluator** (`evaluators/multi_table.py`):
- Evaluates multi-table datasets with enhanced schema matching
- Computes schema coverage, structure scores, and utility scores
- Integrates SD Metrics for quality assessment
- Returns `MultiTableEvaluationReport` with comprehensive metrics

#### 4.4 Schema Matching (`evaluation/matching/`)

**Enhanced Matching Algorithm** (`matching/enhanced_matcher.py`):
- Multi-signal similarity combining name, range, role, and FD participation
- Hungarian algorithm for optimal table and column matching
- Deterministic preprocessing with normalization
- Threshold-based filtering for high-quality matches

**Key Components**:
- `table_matcher.py`: Table-level matching with similarity scoring
- `column_matcher.py`: Column-level matching within matched tables
- `similarity.py`: Similarity computation utilities
- `enhanced_matcher.py`: Main enhanced matching algorithm

#### 4.5 Metrics (`evaluation/metrics/`)

The metrics module is organized into three subdirectories:

**Schema Metrics** (`metrics/schema/`):
- `validation.py`: Primary key and foreign key validation (`check_pk_fk`)
- `coverage.py`: Schema coverage computation for multi-table evaluation

**Table Metrics** (`metrics/table/`):
- `marginals.py`: Marginal distribution metrics (numeric and categorical)
- `correlations.py`: Correlation metrics between column pairs
- `fidelity.py`: Table-level fidelity score computation

**Relational Metrics** (`metrics/relational/`):
- `integrity.py`: Foreign key referential integrity checking (`fk_coverage`)
- `degrees.py`: Degree distribution computation for FK relationships
- `joins.py`: Join selectivity and relationship metrics

#### 4.6 Schema Validation (`evaluation/metrics/schema/validation.py`)

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

**Referential Integrity Checking** (`evaluation/metrics/relational/integrity.py`):

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

#### 4.7 Workload Evaluation (`evaluation/execution/workload.py`)

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

#### 4.8 Single-Table Evaluation Process (`evaluation/evaluators/single_table.py`)

**Main Function**: `evaluate(ir, dfs, cfg) -> EvaluationReport`

**Process**:
1. Schema validation (PK/FK checks)
2. Foreign key coverage computation
3. Table report generation
4. Column distribution evaluation (Zipf, categorical, numeric, seasonal)
5. Workload query execution
6. Summary computation and pass/fail determination

**Backward Compatibility**: This evaluator maintains backward compatibility with the previous evaluation API.

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

#### 4.9 Relational Metrics (`evaluation/metrics/relational/`)

**Purpose**: Evaluates relational properties of multi-table datasets.

**Key Modules**:
- `integrity.py`: FK referential integrity and coverage
- `degrees.py`: Degree distribution analysis (children per parent)
- `joins.py`: Join selectivity and relationship metrics

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

#### 4.10 Schema Coverage Metrics (`evaluation/metrics/schema/coverage.py`)

**Purpose**: Computes schema coverage factors for multi-table evaluation.

**Key Functions**:
- `compute_coverage_factors()`: Computes table and column coverage factors (C_T, C_k)

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

#### 4.11 Table Metrics (`evaluation/metrics/table/`)

**Purpose**: Evaluates data-level properties of individual tables.

**Key Modules**:
- `marginals.py`: Marginal distribution metrics for numeric and categorical columns
- `correlations.py`: Correlation metrics (Pearson, Spearman) between column pairs
- `fidelity.py`: Aggregate table fidelity score computation

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

#### 4.12 Statistical Tests (`evaluation/execution/stats.py`)

**Purpose**: Statistical test functions for distribution validation.

**Key Functions**:
- `zipf_fit()`: Fit Zipf distribution and return R² and exponent
- `chi_square_test()`: Chi-square goodness-of-fit test
- `ks_test()`: Kolmogorov-Smirnov test for numeric distributions
- `wasserstein_distance_metric()`: Earth Mover's Distance
- `cosine_similarity()`: Cosine similarity for distributions
- `gini_coefficient()`: Gini coefficient for measuring inequality
- `top_k_share()`: Top-k share computation

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

**Supported Tests**:

- **Kolmogorov-Smirnov Test**: Compare numeric distributions
- **Chi-Square Test**: Compare categorical distributions
- **Wasserstein Distance**: Earth Mover's Distance for numeric data
- **Cosine Similarity**: Vector similarity for distributions
- **Zipf Fitting**: R² and exponent estimation for Zipf distributions

#### 4.10 Schema Accuracy Evaluation (`evaluate_schema_accuracy.py`)

**Purpose**: Evaluates IR generation accuracy against gold standard schemas from `annotation.jsonl`.

**Key Features**:
1. **Gold Schema Parsing**: Converts gold schemas from JSON format to LogicalIR
2. **Pipeline Execution**: Runs NL→IR pipeline for each description
3. **Schema Matching**: Uses fuzzy matching to align predicted and gold schemas
4. **Comprehensive Metrics**: Calculates table, attribute, PK, FK, and data type accuracy
5. **Partial Evaluation**: Supports `--limit` and `--start` for partial runs
6. **Dual Output**: Generates both JSON results and Markdown report

**Metrics Calculated**:

1. **Table Metrics**:
   - **Precision**: Correctly predicted tables / Total predicted tables
   - **Recall**: Correctly predicted tables / Total gold tables
   - **F1 Score**: Harmonic mean of precision and recall
   - **Accuracy**: Exact match accuracy

2. **Attribute Metrics**:
   - **F1 Score**: Harmonic mean of attribute precision and recall
   - **Accuracy**: Exact match accuracy

3. **Primary Key Accuracy**: Percentage of correctly identified primary keys

4. **Foreign Key Accuracy**: Percentage of correctly identified foreign keys

5. **Data Type Accuracy**: Percentage of correctly identified data types

**Schema Matching Algorithm**:

```python
def match_tables(gold_tables: Set[str], pred_tables: Set[str], threshold: float = 0.7) -> Dict[str, str]:
    """
    Match predicted tables to gold tables using similarity.
    
    Process:
    1. Try exact matches first (case-insensitive, normalized)
    2. Then try similarity matching for remaining tables
    3. Uses both normalized name comparison and direct similarity
    4. Returns mapping: predicted_table -> gold_table
    """
```

**Matching Features**:
- **Normalization**: Removes underscores, spaces, case differences
- **Similarity Threshold**: Default 0.7 (configurable)
- **Bipartite Matching**: Ensures one-to-one table matching
- **Attribute Matching**: Matches attributes within matched tables

**Command-Line Arguments**:

```bash
python evaluate_schema_accuracy.py [--limit N] [--start M]
```

- `--limit N`: Process only first N entries (useful for testing)
- `--start M`: Start from line number M (useful for resuming)

**Output Files**:

1. **`schema_evaluation_results.json`**: Detailed JSON results for each entry
   ```json
   {
     "id": "entry_id",
     "status": "success",
     "table_precision": 0.95,
     "table_recall": 0.90,
     "table_f1": 0.92,
     "attr_f1": 0.88,
     "pk_accuracy": 1.0,
     "fk_accuracy": 0.85,
     "datatype_accuracy": 0.92
   }
   ```

2. **`schema_evaluation_report.md`**: Human-readable Markdown report with:
   - Overall statistics
   - Per-entry breakdown
   - Summary tables
   - Error analysis

**Logging**:

- All output is logged to `evaluation_run.log` for debugging
- Console output shows progress and results
- Detailed error messages for failed entries

**Usage Example**:

```bash
# Evaluate all entries
python evaluate_schema_accuracy.py

# Evaluate first 10 entries (for testing)
python evaluate_schema_accuracy.py --limit 10

# Resume from entry 50
python evaluate_schema_accuracy.py --start 50

# Evaluate entries 50-60
python evaluate_schema_accuracy.py --start 50 --limit 10
```

**Integration with Orchestrator**:

The script uses the Orchestrator with:
- `query_id`: Generated from description hash
- `query_text`: Natural language description
- `enable_repair=True`: Uses repair loops (default)
- `enable_metrics=True`: Tracks quality metrics (default)

This ensures consistent pipeline execution with all features enabled.

#### 4.13 Quality Evaluation (`evaluation/quality/`)

**SD Metrics Integration**:
- Uses SD Metrics library for comprehensive synthetic data quality assessment
- Evaluates column shape, column pair trends, and multi-table relationships
- Provides overall quality scores and detailed per-column/pair metrics

**Key Functions**:
- `evaluate_table_quality()`: Single-table quality evaluation using SD Metrics QualityReport
- `evaluate_multi_table_quality()`: Multi-table quality evaluation with relationship analysis
- `extract_relationship_mappings()`: Extract FK relationships for multi-table evaluation
- `compute_quality_scores()`: Compute quality scores for matched tables/columns

**Quality Metrics**:
- Column shape metrics (KS Complement, TV Complement, Range Coverage, Category Coverage)
- Column pair trends (Correlation Similarity, Contingency Similarity, Contrast)
- Multi-table metrics (Cardinality Shape Similarity, Parent-Child Distribution Similarity)

#### 4.14 Score Aggregation (`evaluation/aggregation/`)

**Purpose**: Aggregates scores from different evaluation dimensions into final scores.

**Key Functions**:
- `compute_schema_score()`: Computes S_schema from coverage factors and matching results
- `compute_intra_structure_score()`: Computes S_structure,intra from table-level fidelity scores
- `compute_inter_structure_score()`: Computes S_structure,inter from referential integrity, cardinality, and trends
- `compute_local_utility()`: Computes S_utility,local using ML model performance
- `compute_query_utility()`: Computes S_utility,queries from query execution results
- `compute_utility_score()`: Aggregates all utility components into S_utility

**Components**:
- `schema_score.py`: Schema score computation (S_schema)
- `structure_score.py`: Structure score computation (S_structure,intra, S_structure,inter)
- `utility_score.py`: Utility score computation (S_utility)

#### 4.15 Multi-Table Evaluation Workflow

**Complete Pipeline** (`evaluators/multi_table.py`):

1. **Schema Matching**: Match tables and columns between real and synthetic schemas
2. **Schema Score**: Compute S_schema from coverage and matching results
3. **Intra-Table Structure**: Compute S_structure,intra from table fidelity scores
4. **Inter-Table Structure**: Compute S_structure,inter from relational metrics
5. **Utility Score**: Compute S_utility from ML tasks and query performance (optional)
6. **Global Score**: Aggregate all scores into S_global (optional)
7. **Quality Scores**: Compute SD Metrics quality scores for matched tables (optional)

**Output**: `MultiTableEvaluationReport` with comprehensive metrics and scores.

#### 4.16 Nuance Coverage Lint (`ir/validators.py`)

**Automated Coverage Checking**:

The system includes a nuance coverage lint that checks if NL requirements are reflected in the IR:

```python
def check_nuance_coverage(nl_text: Optional[str], ir: DatasetIR) -> List[QaIssue]:
    """
    Check if NL requirements are reflected in IR constructs.
    
    Detects missing constructs when NL mentions:
    - "rolling", "window" → checks for DistWindow
    - "incident", "event" → checks for EventSpec
    - "log-normal" → checks for lognormal() function
    - "Zipf" → checks for DistZipf
    - "proration" → checks for overlap_days() function
    - "pay-day" → checks for seasonal patterns
    """
```

**Integration**:

The nuance coverage check is integrated into `validate_dataset()` and can be enabled by passing the NL text:

```python
issues = validate_dataset(ir, nl_text="Generate rolling averages...")
# Will detect if NL mentions "rolling" but IR lacks window functions
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

## Additional Tools and Frameworks

### RSchema (Text2Schema) Evaluation Framework

**Location**: `RSchema (Text2Schema)/`

**Purpose**: Evaluation framework for comparing NL2Data schema generation against RSchema (Text2Schema) gold standard annotations.

**Key Components**:
- `annotation.jsonl`: Gold standard schema annotations in JSONL format
- `annotation_ddl.jsonl`: DDL format annotations for comparison
- `evaluate_rschema.py`: Evaluation script that runs NL2Data pipeline on RSchema descriptions and compares results
- `evaluation_results.md`: Comprehensive evaluation report with metrics and analysis

**Usage**:
```bash
cd "RSchema (Text2Schema)"
python evaluate_rschema.py
```

**Evaluation Metrics**:
- Schema accuracy (tables, columns, PKs, FKs)
- Data type accuracy
- Precision, recall, and F1 scores
- Comparison with RSchema baseline

### Prompt Generator Format (`prompt_generator_format.py`)

**Purpose**: Utility functions for generating prompts for database schema design tasks, particularly for entity-relationship modeling and schema generation. This module provides a comprehensive set of prompt generation functions for multi-step schema design workflows.

**Key Features**:
- **Question Analysis**: `get_question_analysis_prompt()` - Analyzes user requirements
- **Entity Identification**: Multiple variants for entity and attribute extraction
  - `get_entity_analysis_prompt()` - Basic entity identification
  - `get_entity_all_analysis_prompt()` - Entity and attribute identification
  - English and Chinese variants available
- **Relationship Analysis**: 
  - `get_relation_analysis_prompt()` - Basic relationship identification
  - `get_relation_all_analysis_prompt()` - Relationship with cardinality and attributes
  - `get_relation_analysis_type_prompt()` - Relationship cardinality types
- **Functional Dependency Analysis**:
  - `get_entity_functional_dependency_analysis_prompt()` - Entity FD analysis
  - `get_relation_functional_dependency_analysis_prompt()` - Relationship FD analysis
- **Consensus and Verification**:
  - `get_consensus_prompt()` - Expert consensus checking
  - `get_verification_entity_prompt()` - Entity verification
  - `get_dependency_consensus_prompt()` - FD consensus checking
- **Direct Schema Generation**:
  - `get_direct_prompt()` - Direct schema generation from requirements
  - `get_direct_few_shot_prompt()` - Few-shot schema generation
  - `get_cot_prompt()` - Chain-of-thought schema generation

**Usage**: Imported by other scripts for generating structured prompts for LLM-based schema design tasks. Supports both Chinese and English prompt generation.

### Batch Pipeline Runner (`run_all_pipelines.py`)

**Purpose**: Runs the NL2Data pipeline for all description files in the realistic datasets framework and saves outputs in dataset folders.

**Key Features**:
- Processes all description files in realistic_datasets directories
- Generates unique query IDs from description hashes
- Saves IR and generated data in organized output structure
- Handles errors gracefully and continues processing
- Supports resuming from failures

**Usage**:
```bash
python run_all_pipelines.py
```

**Output Structure**:
- Creates output directories for each description
- Saves `dataset_ir.json` and `ir_evaluation.json` in each output directory
- Organizes outputs by dataset source and name

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

### Utility Functions

#### IR I/O (`utils/ir_io.py`)

**Purpose**: Load and save DatasetIR to/from JSON files.

**Functions**:

```python
def load_ir_from_json(ir_path: Path) -> DatasetIR:
    """
    Load DatasetIR from a JSON file.
    
    Features:
    - Validates file exists
    - Checks for empty/corrupted files
    - Provides helpful error messages
    - Uses Pydantic TypeAdapter for validation
    
    Raises:
    - FileNotFoundError: If file doesn't exist
    - ValueError: If file is empty or corrupted
    - ValidationError: If JSON doesn't match DatasetIR schema
    """
```

```python
def save_ir_to_json(ir: DatasetIR, ir_path: Path) -> None:
    """
    Save DatasetIR to a JSON file.
    
    Features:
    - Creates parent directories if needed
    - Pretty-prints JSON with 2-space indentation
    - Uses UTF-8 encoding
    """
```

**Usage**:
```python
from nl2data.utils.ir_io import load_ir_from_json, save_ir_to_json
from pathlib import Path

# Save IR
save_ir_to_json(dataset_ir, Path("output/dataset_ir.json"))

# Load IR
ir = load_ir_from_json(Path("output/dataset_ir.json"))
```

#### Data Loader (`utils/data_loader.py`)

**Purpose**: Load CSV files into pandas DataFrames.

**Functions**:

```python
def load_csv_files(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a directory into DataFrames.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping table names (file stems) to DataFrames
        
    Example:
        If data_dir contains "fact_sales.csv" and "dim_product.csv",
        returns {"fact_sales": DataFrame, "dim_product": DataFrame}
    """
```

**Usage**:
```python
from nl2data.utils.data_loader import load_csv_files
from pathlib import Path

dfs = load_csv_files(Path("output/"))
# dfs = {"fact_sales": DataFrame, "dim_product": DataFrame, ...}
```

#### Agent Factory (`utils/agent_factory.py`)

**Purpose**: Create agent sequences from natural language input.

**Functions**:

```python
def create_agent_list(nl_request: str) -> List[BaseAgent]:
    """
    Create a list of agents for the pipeline.
    
    Args:
        nl_request: Natural language description
        
    Returns:
        List of agents in execution order:
        [ManagerAgent, ConceptualDesigner, LogicalDesigner,
         DistributionEngineer, WorkloadDesigner, QACompilerAgent]
    """
```

```python
def create_agent_sequence(nl_request: str) -> List[Tuple[str, BaseAgent]]:
    """
    Create agent sequence with names for logging.
    
    Returns:
        List of (agent_name, agent) tuples
    """
```

**Usage**:
```python
from nl2data.utils.agent_factory import create_agent_list

nl_description = "Generate a retail sales dataset..."
agents = create_agent_list(nl_description)

# Execute with orchestrator
from nl2data.agents.orchestrator import Orchestrator
from nl2data.agents.base import Blackboard

board = Orchestrator(agents).execute(Blackboard())
```

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

The NL2Data CLI provides four main commands for running the pipeline end-to-end or in stages.

**Installation**:
```bash
# Install the package in development mode
cd nl2data
pip install -e .
```

**Command Structure**:
```bash
python -m nl2data.cli.app <command> [arguments]
# Or using the script:
python nl2data/scripts/nl2data.py <command> [arguments]
```

#### 1. End-to-End Pipeline (`end2end`)

Runs the complete pipeline from natural language to generated data.

**Usage**:
```bash
python scripts/nl2data.py end2end <description_file> <out_dir>
```

**Arguments**:
- `description_file`: Path to a text file containing the natural language description
- `out_dir`: Output directory where IR and CSV files will be written

**Example**:
```bash
python scripts/nl2data.py end2end example_description.txt output/
```

**Output**:
- `out_dir/dataset_ir.json`: Complete DatasetIR JSON file
- `out_dir/*.csv`: Generated CSV files (one per table)

**Process**:
1. Reads natural language description from file
2. Executes multi-agent pipeline (Manager → Conceptual → Logical → Distribution → Workload → QA)
3. Saves DatasetIR to JSON
4. Generates CSV files from IR
5. Reports completion status

#### 2. Natural Language to IR (`nl2ir`)

Converts natural language to DatasetIR JSON without generating data.

**Usage**:
```bash
python scripts/nl2data.py nl2ir <description_file> <out_ir>
```

**Arguments**:
- `description_file`: Path to natural language description file
- `out_ir`: Output path for DatasetIR JSON file

**Example**:
```bash
python scripts/nl2data.py nl2ir example_description.txt dataset_ir.json
```

**Use Cases**:
- Inspect IR structure before generating data
- Debug agent outputs
- Modify IR manually before generation
- Share IR files without regenerating

#### 3. Generate Data from IR (`generate`)

Generates CSV files from an existing DatasetIR JSON file.

**Usage**:
```bash
python scripts/nl2data.py generate <ir_json> <out_dir>
```

**Arguments**:
- `ir_json`: Path to DatasetIR JSON file
- `out_dir`: Output directory for CSV files

**Example**:
```bash
python scripts/nl2data.py generate dataset_ir.json output/
```

**Configuration**:
- Uses `seed` from settings (default: 7) for reproducibility
- Uses `chunk_rows` from settings (default: 1,000,000) for fact table streaming
- Can be overridden via environment variables (see Configuration section)

**Output**:
- `out_dir/*.csv`: One CSV file per table in the IR

#### 4. Evaluate Generated Data (`evaluate_data`)

Evaluates generated CSV files against the DatasetIR specifications.

**Usage**:
```bash
python scripts/nl2data.py evaluate_data <ir_json> <data_dir> <out_report>
```

**Arguments**:
- `ir_json`: Path to DatasetIR JSON file
- `data_dir`: Directory containing generated CSV files
- `out_report`: Output path for evaluation report JSON

**Example**:
```bash
python scripts/nl2data.py evaluate_data dataset_ir.json output/ evaluation_report.json
```

**Evaluation Checks**:
- **Schema Validation**: PK/FK integrity, column types, nullability
- **Statistical Validation**: Distribution fitting (Zipf, KS test, chi-square)
- **Workload Testing**: Query execution time, result correctness

**Output**:
- `out_report`: JSON file containing:
  - Table reports (row counts, PK/FK status)
  - Column reports (distribution metrics, statistical tests)
  - Workload reports (query performance, pass/fail status)
  - Summary (total failures, overall pass/fail)

**Example Report Structure**:
```json
{
  "schema": [
    {
      "name": "fact_sales",
      "row_count": 5000000,
      "pk_ok": true,
      "fk_ok": true
    }
  ],
  "columns": [
    {
      "table": "fact_sales",
      "column": "product_id",
      "metrics": [
        {
          "name": "zipf_r2",
          "value": 0.95,
          "passed": true
        }
      ]
    }
  ],
  "workloads": [
    {
      "sql": "SELECT product_id, COUNT(*) FROM fact_sales GROUP BY product_id",
      "type": "group_by",
      "elapsed_sec": 2.3,
      "passed": true
    }
  ],
  "summary": {
    "failures": 0,
    "total_checks": 15
  },
  "passed": true
}
```

#### CLI Error Handling

All commands include error handling:
- **File Not Found**: Clear error messages for missing input files
- **Invalid JSON**: Validation errors with helpful context
- **Generation Failures**: Continues with remaining tables, reports failures
- **LLM Errors**: Retry logic with exponential backoff (see Configuration)

#### Environment Variables

CLI commands respect environment variables from `.env` file:
- `GEMINI_API_KEY`: Google Gemini API key
- `GEMINI_MODEL`: Model name (e.g., "gemini-1.5-pro")
- `OPENAI_API_KEY`: OpenAI API key (legacy)
- `MODEL_NAME`: OpenAI model name (legacy)
- `LLM_URL`: Local LLM API endpoint
- `MODEL`: Model name for local LLM
- `SEED`: Random seed for generation (default: 7)
- `CHUNK_ROWS`: Rows per chunk for fact tables (default: 1,000,000)
- `LOG_LEVEL`: Logging level (default: "INFO")

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
        "kind": "uniform" | "normal" | "lognormal" | "pareto" | "poisson" | "exponential" | "mixture" | "zipf" | "seasonal" | "categorical" | "derived" | "window",
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
   - Use for symmetric bell curve distributions
   - Example: {"kind": "normal", "mean": 50.0, "std": 10.0}

3. Lognormal: {"kind": "lognormal", "mean": NUMBER, "sigma": NUMBER}
   - Use for right-skewed distributions (e.g., income, transaction amounts)
   - mean: Mean of underlying normal distribution, sigma: Standard deviation (must be > 0)
   - Example: {"kind": "lognormal", "mean": 3.0, "sigma": 1.0}

4. Pareto: {"kind": "pareto", "alpha": NUMBER, "xm": NUMBER}
   - Use for heavy-tailed distributions (e.g., wealth, file sizes)
   - alpha: Shape parameter (must be > 0), xm: Scale parameter/minimum value (must be > 0, default: 1.0)
   - Example: {"kind": "pareto", "alpha": 2.0, "xm": 1.0}

5. Poisson: {"kind": "poisson", "lam": NUMBER}
   - Use for count distributions (session lengths, event counts, arrivals)
   - lam: Rate parameter (must be > 0, typically 1-20)
   - Example: {"kind": "poisson", "lam": 5.0}

6. Exponential: {"kind": "exponential", "scale": NUMBER}
   - Use for inter-arrival times, waiting times, lifetimes
   - scale: Scale parameter (1/lambda, must be > 0, default: 1.0)
   - Example: {"kind": "exponential", "scale": 2.5}

7. Mixture: {"kind": "mixture", "components": [{"weight": NUMBER, "distribution": {...}}, ...]}
   - Use for multi-modal distributions (multiple peaks)
   - weights must sum to approximately 1.0
   - Example: {"kind": "mixture", "components": [{"weight": 0.7, "distribution": {"kind": "normal", "mean": 10, "std": 2}}, {"weight": 0.3, "distribution": {"kind": "normal", "mean": 30, "std": 3}}]}

8. Zipf: {"kind": "zipf", "s": NUMBER, "n": INTEGER | null}
   - Use for discrete popularity distributions (power-law, 80/20 rule)
   - s = exponent (typically 1.2-2.0), n = domain size
   - Example: {"kind": "zipf", "s": 1.2, "n": 1000}

9. Seasonal: {"kind": "seasonal", "granularity": "month" | "week" | "hour", "weights": {...}}
   - Use for date columns with seasonal patterns
   - granularity: "month", "week", or "hour"
   - weights must be a dictionary mapping period names to numbers
   - Example: {"kind": "seasonal", "granularity": "month", "weights": {"December": 0.15, "November": 0.12, "January": 0.08, ...}}
   - Example: {"kind": "seasonal", "granularity": "hour", "weights": {"07:00-09:00": 2.5, "12:00-14:00": 1.8, ...}}

10. Categorical: {"kind": "categorical", "domain": {"values": ["string1", "string2", ...], "probs": [0.1, 0.2, ...] | null}}
   - Use for discrete values with known options (status codes, categories, boolean flags, enums)
   - values MUST be an array of STRINGS (convert booleans/numbers to strings)
   - Example: {"kind": "categorical", "domain": {"values": ["true", "false"], "probs": [0.3, 0.7]}}
   - Example: {"kind": "categorical", "domain": {"values": ["2021", "2022", "2023"], "probs": null}}

11. Derived: {"kind": "derived", "expression": "expression_string", "dtype": "float" | "int" | "bool" | "date" | "datetime" | null}
   - Use for columns computed from other columns
   - expression: DSL expression (Python-like syntax)
   - dtype: Optional type hint
   - Example: {"kind": "derived", "expression": "price * quantity", "dtype": "float"}

12. Window: {"kind": "window", "expression": "sum(amount)" | "lag(amount, 1)" | ..., "partition_by": [...], "order_by": "timestamp", "frame": {"type": "RANGE" | "ROWS", "preceding": "7d", "following": null}}
   - Use for rolling aggregations (sum, mean, count over windows) and lag/lead operations
   - expression: Aggregation function or column reference (lag/lead allowed here)
   - partition_by: Columns to partition by (optional)
   - order_by: Column to order by (must be datetime for RANGE windows)
   - frame: Window frame specification
   - Example: {"kind": "window", "expression": "rolling_sum(amount)", "partition_by": ["customer_id"], "order_by": "timestamp", "frame": {"type": "RANGE", "preceding": "30d"}}

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

The system includes comprehensive error handling at multiple levels to ensure robustness and provide helpful error messages.

### Error Handling Architecture

**Layers of Error Handling**:
1. **LLM Call Level**: Retry logic with exponential backoff
2. **Agent Level**: Validation and repair loops
3. **Generation Level**: Graceful failure with partial results
4. **Evaluation Level**: Continue evaluation despite individual failures

### LLM Response Handling

**Retry Logic** (`agents/tools/retry.py`):

The system implements sophisticated retry logic for LLM API calls:

```python
def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 5.0,
    timeout_errors: bool = True,
    operation_name: str = "operation"
) -> T:
    """
    Retry a function with exponential backoff.
    
    Features:
    - Exponential backoff: delay = base_delay * (2 ** attempt)
    - Transient error detection: retries on network errors, timeouts, rate limits
    - Non-transient errors: raises immediately (validation errors, etc.)
    - Configurable timeout handling
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if not is_transient_error(e) or attempt == max_retries - 1:
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)
```

**Transient Error Detection**:

```python
def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient (should be retried).
    
    Transient errors:
    - Network errors (ConnectionError, TimeoutError)
    - Rate limiting (429 status codes)
    - Server errors (500, 502, 503, 504)
    - Model reloading (local LLMs)
    
    Non-transient errors:
    - Validation errors (400, 422)
    - Authentication errors (401, 403)
    - Not found errors (404)
    """
```

**LLM Client Error Handling** (`agents/tools/llm_client.py`):

Each LLM provider includes specific error handling:

- **Gemini**: Handles quota exceeded, rate limits, and API errors
- **OpenAI**: Handles rate limits, server errors, and timeout errors
- **Local LLM**: Handles model reloading and connection errors

### Agent-Level Error Handling

**JSON Extraction Errors** (`agents/tools/json_parser.py`):

```python
def extract_json(text: str, agent_name: str = "agent") -> dict:
    """
    Extract JSON from LLM response with comprehensive error handling.
    
    Error Handling:
    1. Tries to find JSON in markdown code blocks
    2. Tries to find JSON between ```json and ``` markers
    3. Tries to find JSON object/array directly
    4. Provides helpful error messages with context
    """
    try:
        # Try multiple extraction strategies
        json_str = _extract_from_markdown(text) or _extract_direct_json(text)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Provide helpful error message with context
        error_msg = f"Failed to extract JSON from {agent_name} response"
        # Include snippet of problematic text
        raise ValueError(f"{error_msg}: {e}")
```

**Validation Errors** (`ir/validators.py`):

```python
def validate_dataset(ir: DatasetIR) -> List[QaIssue]:
    """
    Validate DatasetIR and return list of issues.
    
    Validation Checks:
    - Primary key existence
    - Foreign key references
    - Column type consistency
    - Distribution compatibility
    - Derived column dependencies
    
    Returns:
        List of QaIssue objects with:
        - code: Issue code (e.g., "MISSING_PRIMARY_KEY")
        - location: Where the issue occurs (e.g., "table:fact_sales")
        - message: Human-readable error message
    """
```

**Agent Retry Utility** (`agents/tools/agent_retry.py`):

```python
def call_llm_with_retry(
    messages: List[Dict[str, str]],
    model_class: Type[T],
    max_retries: int = 2,
    pre_process: Optional[Callable] = None,
    custom_validation: Optional[Callable] = None
) -> T:
    """
    Call LLM and parse response with retry logic.
    
    Process:
    1. Call LLM with retry logic
    2. Extract JSON from response
    3. Pre-process JSON (fix common LLM mistakes)
    4. Validate against Pydantic model
    5. Run custom validation if provided
    6. Retry on validation errors (up to max_retries)
    
    Error Handling:
    - JSON extraction errors: retries with error message in prompt
    - Validation errors: retries with validation errors in prompt
    - Non-retryable errors: raises immediately
    """
```

### Generation-Level Error Handling

**Table Generation Failures** (`generation/engine/pipeline.py`):

```python
def generate_from_ir(ir: DatasetIR, out_dir: Path, seed: int, chunk_rows: int) -> None:
    """
    Generate data with graceful failure handling.
    
    Error Handling:
    - Dimension generation failures: logs error, continues with remaining tables
    - Fact generation failures: logs error, continues with remaining tables
    - Reports all failures at the end
    - Does not stop pipeline on single table failure
    """
    failed_dimensions = []
    failed_tables = []
    
    # Generate dimensions
    for name, table_spec in dims.items():
        try:
            df = generate_dimension(table_spec, ir, rng, derived_reg)
            write_csv(df, output_path)
        except Exception as e:
            logger.error(f"Failed to generate dimension '{name}': {e}")
            failed_dimensions.append(name)
            # Continue with next table
    
    # Generate facts
    for name, table_spec in facts.items():
        try:
            stream = generate_fact_stream(...)
            write_csv_stream(stream, output_path)
        except Exception as e:
            logger.error(f"Failed to generate fact table '{name}': {e}")
            failed_tables.append(name)
            # Continue with next table
    
    # Report results
    if failed_dimensions or failed_tables:
        logger.warning(f"Generation completed with {len(failed_dimensions) + len(failed_tables)} failures")
```

**Derived Column Evaluation Errors** (`generation/engine/fact_generator.py`):

```python
# Phase 2: Compute derived columns
for col_name in derived_cols:
    try:
        df_chunk[col_name] = eval_derived(prog, df_chunk, rng=rng)
    except KeyError as e:
        # Missing column dependency
        missing_col = str(e).strip("'")
        logger.error(
            f"Failed to compute derived column '{col_name}': "
            f"Missing column '{missing_col}'. "
            f"Available columns: {list(df_chunk.columns)}. "
            f"Expression: {prog.expr}. "
            f"Dependencies: {prog.dependencies}."
        )
        raise ValueError(
            f"Derived column '{col_name}' depends on column '{missing_col}' "
            f"which is not available."
        ) from e
    except Exception as e:
        logger.error(
            f"Failed to compute derived column '{col_name}': {e}. "
            f"Expression: {prog.expr}."
        )
        raise
```

### Evaluation-Level Error Handling

**Statistical Test Failures** (`evaluation/report_builder.py`):

```python
def evaluate(ir: DatasetIR, dfs: Dict[str, pd.DataFrame], cfg: EvaluationConfig) -> EvaluationReport:
    """
    Evaluate data with graceful failure handling.
    
    Error Handling:
    - Missing tables: logs warning, skips evaluation
    - Missing columns: logs warning, skips column evaluation
    - Statistical test failures: marks as failed, continues
    - Query execution errors: marks as failed, continues
    - Returns partial results even if some checks fail
    """
    # Schema validation
    issues = check_pk_fk(ir)  # May return issues, but doesn't stop
    
    # Column evaluation
    for cg in ir.generation.columns:
        try:
            # Run statistical tests
            metrics = compute_metrics(df, cg, cfg)
        except Exception as e:
            logger.warning(f"Failed to evaluate column {cg.table}.{cg.column}: {e}")
            # Continue with next column
    
    # Workload evaluation
    for w in ir.workload.targets:
        try:
            result = run_workload(w, dfs, cfg)
        except Exception as e:
            logger.warning(f"Failed to execute workload: {e}")
            # Continue with next workload
```

### Logging System

**Logging Configuration** (`config/logging.py`):

```python
def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure logging for the application.
    
    Features:
    - Console logging with colored output
    - Optional file logging
    - Configurable log levels
    - Structured log format with timestamps
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
```

**Logger Usage**:

```python
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

logger.info("Starting data generation")
logger.warning("Table generation failed, continuing...")
logger.error("Critical error occurred", exc_info=True)
logger.debug("Detailed debug information")
```

### Error Message Best Practices

**Helpful Error Messages**:

1. **Context**: Include what operation was being performed
2. **Location**: Specify where the error occurred (table, column, etc.)
3. **Available Information**: Show what was available (columns, values, etc.)
4. **Suggestions**: Provide hints on how to fix the issue
5. **Truncation**: Long error messages are truncated to prevent log spam

**Example Error Messages**:

```python
# Good error message
raise ValueError(
    f"Derived column '{col_name}' in table '{table.name}' depends on "
    f"column '{missing_col}' which is not available. "
    f"Available columns: {list(df_chunk.columns)}. "
    f"Expression: {prog.expr}. "
    f"Dependencies: {prog.dependencies}."
)

# Bad error message
raise ValueError("Column not found")
```

### Robustness Features

1. **Partial Results**: System continues even if some components fail
2. **Graceful Degradation**: Falls back to simpler operations when possible
3. **Error Recovery**: Retries transient errors automatically
4. **Comprehensive Logging**: All errors are logged with context
5. **User-Friendly Messages**: Errors include helpful context and suggestions

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

## Testing Framework

The project includes comprehensive testing utilities for validating the pipeline on multiple example queries.

### Test Scripts

#### 1. Full Pipeline Testing (`test_all_queries.py`)

**Purpose**: Test the complete pipeline (NL → IR → Data → Evaluation) on a set of example queries.

**Structure**:
```python
def test_query(
    query_num: int,
    query_text: str,
    output_base: Path,
    ir_only: bool = False
) -> Tuple[bool, str]:
    """
    Test a single query through the full pipeline.
    
    Args:
        query_num: Query number for identification
        query_text: Natural language description
        output_base: Base directory for outputs
        ir_only: If True, only generate IR (skip data generation)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # 1. Create output directory
    output_dir = output_base / f"query_{query_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Run NL → IR pipeline
    agents = create_agent_list(query_text)
    board = Orchestrator(agents).execute(Blackboard())
    ir = board.dataset_ir
    
    # 3. Save IR
    save_ir_to_json(ir, output_dir / "dataset_ir.json")
    
    if ir_only:
        return True, "IR generated successfully"
    
    # 4. Generate data
    generate_from_ir(ir, output_dir, seed=7, chunk_rows=1_000_000)
    
    # 5. Evaluate (optional)
    # ...
    
    return True, "Success"
```

**Usage**:
```bash
python test_all_queries.py
```

**Features**:
- Tests multiple queries from `example queries.txt`
- Generates IR and data for each query
- Creates separate output directories per query
- Reports success/failure for each query
- Can run in IR-only mode for faster testing

#### 2. Phase 1 Evaluation (`test_phase1_evaluation.py`)

**Purpose**: Evaluate IR quality (validation + derived column compilation) without generating data.

**Structure**:
```python
def evaluate_ir(ir: DatasetIR) -> Tuple[bool, Dict]:
    """
    Evaluate IR quality (validation + derived column compilation).
    
    Returns:
        Tuple of (is_valid, evaluation_summary)
    """
    summary = {
        "validation_issues": [],
        "derived_compilation_errors": [],
        "num_tables": len(ir.logical.tables),
        "num_columns": sum(len(t.columns) for t in ir.logical.tables.values()),
        "num_derived_columns": 0,
        "has_primary_keys": True,
        "has_foreign_keys": False,
        "derived_columns_valid": True,
    }
    
    # 1. Validate IR structure
    validation_issues = validate_dataset(ir)
    summary["validation_issues"] = [...]
    
    # 2. Check primary keys
    for table_name, table in ir.logical.tables.items():
        if not table.primary_key:
            summary["has_primary_keys"] = False
        if table.foreign_keys:
            summary["has_foreign_keys"] = True
    
    # 3. Count derived columns
    derived_cols = count_derived_columns(ir)
    summary["num_derived_columns"] = len(derived_cols)
    
    # 4. Try to compile derived columns
    try:
        derived_reg = build_derived_registry(ir)
        summary["derived_columns_valid"] = True
        summary["derived_programs"] = len(derived_reg.programs)
    except Exception as e:
        summary["derived_columns_valid"] = False
        summary["derived_compilation_errors"] = [str(e)]
    
    # Overall validity
    is_valid = (
        len(validation_issues) == 0 and
        summary["has_primary_keys"] and
        summary["derived_columns_valid"]
    )
    
    return is_valid, summary
```

**Usage**:
```bash
python test_phase1_evaluation.py
```

**Features**:
- Fast evaluation (no data generation)
- Validates IR structure
- Checks derived column compilation
- Reports validation issues
- Useful for debugging agent outputs

### Test Output Structure

**Directory Layout**:
```
test_output/
├── query_1/
│   ├── dataset_ir.json
│   ├── fact_sales.csv
│   ├── dim_product.csv
│   └── ir_evaluation.json
├── query_2/
│   └── ...
└── test_report.md
```

**Test Report** (`test_output/test_report.md`):
- Summary of all queries tested
- Status for each query (✅ Generated, ❌ Failed)
- IR evaluation results
- Common issues and patterns

### Running Tests

**Full Pipeline Test**:
```bash
# Test all queries
python test_all_queries.py

# Test specific query
python test_all_queries.py --query 1
```

**Phase 1 Evaluation**:
```bash
# Evaluate IR quality for all queries
python test_phase1_evaluation.py
```

**Test Configuration**:
- Output directory: `test_output/`
- Seed: 7 (for reproducibility)
- Chunk rows: 1,000,000 (for fact tables)

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

**Document Version**: 2.0  
**Last Updated**: 2024  
**Project**: NL2Data - Natural Language to Synthetic Relational Data Generation System

---

## Additional Components

### Monitoring System (`monitoring/quality_metrics.py`)

**Quality Metrics Tracking**:

The system tracks quality metrics across the pipeline for monitoring and debugging:

- **Agent Metrics**: Execution time, success/failure, error messages per agent
- **Query Metrics**: Total issues, spec coverage, repair attempts, processing time
- **Issue Aggregation**: Counts of validation issues by code

**Usage**:

```python
from nl2data.monitoring.quality_metrics import QualityMetricsCollector

collector = QualityMetricsCollector()
collector.start_query("query_1", "Generate a retail dataset...")
collector.start_agent("ManagerAgent")
# ... agent execution ...
collector.end_agent("ManagerAgent", success=True)
metrics = collector.get_query_metrics("query_1")
```

**Metrics Collected**:
- Agent execution times
- Validation issue counts by code
- Generation spec coverage percentage
- Repair loop attempts and success
- Total processing time per query

### Utility Modules

**IR I/O** (`utils/ir_io.py`):

```python
def load_ir_from_json(ir_path: Path) -> DatasetIR:
    """Load DatasetIR from JSON file with error handling."""
    
def save_ir_to_json(ir: DatasetIR, ir_path: Path) -> None:
    """Save DatasetIR to JSON file."""
```

**Data Loading** (`utils/data_loader.py`):

```python
def load_csv_files(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from directory into DataFrames."""
```

**Agent Factory** (`utils/agent_factory.py`):

```python
def create_agent_list(nl_description: str) -> List[BaseAgent]:
    """Create list of agents in execution order."""
    
def create_agent_sequence(nl_description: str) -> List[Tuple[str, BaseAgent]]:
    """Create agent sequence with names for logging."""
```

### CLI Commands

The CLI provides four main commands:

1. **`end2end`**: Complete pipeline (NL → IR → Data)
2. **`nl2ir`**: Convert NL to IR only
3. **`generate`**: Generate data from existing IR
4. **`evaluate_data`**: Evaluate generated data against IR

See [Usage Examples](#usage-examples) section for detailed command documentation.

---

## Appendix: Complete Feature List

### DSL Functions (Complete Reference)

**Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**`

**Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`, `in`, `not in`

**Boolean**: `and`, `or`, `not`

**Math**: `abs`, `sqrt`, `log`, `exp`, `clip`

**Date/Time Extraction**: `hour`, `date`, `day_of_week`, `day_of_month`, `month`, `year`

**Time Arithmetic**: `seconds`, `minutes`, `hours`, `days`

**Type Casting**: `int`, `float`, `bool`, `str`

**Distributions**: `normal`, `lognormal`, `pareto`, `uniform`

**Conditional**: `where`, `case_when`

**String Operations**: `concat`, `format`, `substring`

**Helper Functions**: `between`, `geo_distance`, `ts_diff`, `overlap_days`

**Null Checks**: `isnull`, `notnull`

**Weighted Choice**: `weighted_choice`, `weighted_choice_if`

### Distribution Types

- **Uniform**: `DistUniform` (low, high)
- **Normal**: `DistNormal` (mean, std)
- **Zipf**: `DistZipf` (s, n)
- **Seasonal**: `DistSeasonal` (granularity, weights)
- **Categorical**: `DistCategorical` (domain with values/probs)
- **Derived**: `DistDerived` (expression, dtype)
- **Window**: `DistWindow` (expression, partition_by, order_by, frame)

### Event Effect Types

- **multiply_distribution**: Multiply column values by factor
- **add_offset**: Add constant offset to values
- **set_value**: Set values to specific value (with optional probability)
- **change_distribution**: Change distribution (future enhancement)

### Window Frame Types

- **RANGE**: Time-based windows (requires datetime order_by column)
- **ROWS**: Row-based windows (works with any order_by column)

### Provider Types

- **Faker**: `faker.*` (name, email, phone_number, address, city, country, company, job, date, date_time)
- **Mimesis**: `mimesis.*` (full_name, email, telephone, address)
- **Geo Lookup**: `lookup.*` (city, country from GeoNames datasets)

### Evaluation Metrics

- **Schema**: PK/FK integrity, table/column coverage
- **Statistical**: KS test, chi-square, Wasserstein distance, Zipf R²
- **Relational**: FK coverage, join selectivity, degree histograms
- **Workload**: Query runtime, correctness
- **Table-Level**: Marginal distributions, correlations, mutual information

