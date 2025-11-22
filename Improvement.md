# NL2Data IR Enhancement Plan

> **Purpose**: Tactical engineering guide for making IRs richer and more aligned with natural language descriptions.

---

## ⚠️ CRITICAL: Prompt Inconsistencies Found

**Status**: **URGENT FIX NEEDED** - Prompts incorrectly tell LLMs to avoid features that are actually supported.

**Key Issues**:
1. **DSL Function Mismatch**: Prompts say `str()`, `int()`, `float()`, `bool()`, `normal()`, `lognormal()`, `pareto()` are FORBIDDEN, but code actually allows them
2. **Seasonal Granularity**: Prompt says `"month" | "week"` but code supports `"hour"` too
3. **Window Functions**: Confusing guidance about `lag()`/`lead()` - forbidden in derived expressions but allowed in window specs
4. **Domain Bias**: 761-line prompt heavily focused on fraud/finance, may cause over-eager fraud column creation in other domains

**Impact**: LLMs are avoiding valid DSL functions, leading to suboptimal derived expressions and missing features.

**Fix Priority**: **HIGH** - See Task 1 below.

---

## 🗺️ Roadmap: You Are Here

### ✅ Before Next Test Run Checklist

- [x] LognormalSampler & ParetoSampler implemented + unit tests green ✅ DONE
- [x] MixtureSampler implemented + unit tests green ✅ DONE
- [ ] Prompts updated & spot-checked on 2–3 NL descriptions (see Task 3 below)
- [x] Nuance coverage checker extended & reporting in test_report ✅ DONE (basic implementation exists)
- [ ] Integration tests: re-run NL-3 (fraud) and NL-1 (clickstream) to verify new features appear in IR

---

### ✅ **Done** (Implemented)
- ✅ DSL extensions: `in`/`not in`, casts (`int`, `float`, `bool`, `str`), `normal`/`lognormal`/`pareto` functions
- ✅ `case_when` macro and `between` helper
- ✅ Window functions (`DistWindow`, `window_eval.py`, Phase 3 in generator)
- ✅ Event/Incident system (`EventSpec`, `EventEffect`, `event_eval.py`, Phase 1.5)
- ✅ Basic nuance coverage checker (`check_nuance_coverage()`)
- ✅ Agent repair loop system (integrated in Orchestrator)
- ✅ Quality metrics tracking (integrated in Orchestrator)
- ✅ Constraint enforcement (integrated in fact generator)
- ✅ **LognormalSampler & ParetoSampler** (implemented in `generation/distributions/numeric.py`)
- ✅ **MixtureSampler** (implemented in `generation/distributions/mixture.py`)
- ✅ **All IR models** (DistLognormal, DistPareto, DistMixture in `ir/generation.py`)

### 🎯 **Now** (High Impact, Low-Medium Risk)
1. ~~**LognormalSampler & ParetoSampler**~~ ✅ **DONE** - Already implemented
2. ~~**Mixture distribution**~~ ✅ **DONE** - Already implemented
3. **Fix prompt inconsistencies** (HIGH impact, LOW effort, LOW risk) - **CRITICAL**: Prompts don't match actual DSL capabilities
4. **Add Poisson & Exponential distributions** (HIGH impact, LOW effort, LOW risk) - Critical for count and time-based patterns
5. **Enhance nuance coverage** with more patterns (MEDIUM impact, MEDIUM effort, LOW risk) - Partially done, can be expanded

### 📋 **Later** (Nice-to-have)
- Additional distributions: Gamma, Beta, Binomial, Truncated Normal (MEDIUM impact, LOW effort each)
- Conditional/correlated distributions (MEDIUM impact, HIGH effort)
- State machine constraints (MEDIUM impact, MEDIUM effort)
- `within_days_of()` helper (MEDIUM impact, LOW effort)
- WorkloadIR integration (MEDIUM impact, MEDIUM effort)

---

## 📊 Quick Status Table

| Feature | Status | Impact | Risk | Effort | Priority |
|---------|--------|--------|------|--------|----------|
| DSL: `in`/casts/distributions | ✅ Done | HIGH | - | - | - |
| Window functions | ✅ Done | HIGH | - | - | - |
| Event/Incident system | ✅ Done | HIGH | - | - | - |
| Nuance coverage (basic) | ✅ Done | MEDIUM | - | - | - |
| Lognormal/Pareto samplers | ✅ Done | HIGH | - | - | - |
| Mixture distribution | ✅ Done | HIGH | - | - | - |
| **Fix prompt inconsistencies** | 🎯 Now | **HIGH** | **LOW** | **LOW** | **1** |
| **Poisson & Exponential distributions** | 🎯 Now | **HIGH** | **LOW** | **LOW** | **2** |
| Nuance coverage (enhanced) | 🎯 Now | MEDIUM | LOW | MEDIUM | **3** |
| Conditional distributions | 📋 Later | MEDIUM | MEDIUM | HIGH | 5 |
| State machines | 📋 Later | MEDIUM | MEDIUM | MEDIUM | 6 |
| `within_days_of()` helper | 📋 Later | MEDIUM | LOW | LOW | 7 |
| WorkloadIR integration | 📋 Later | MEDIUM | LOW | MEDIUM | 8 |

---

## 🎯 Remaining High-Impact Tasks

### Task 1: Fix Prompt Inconsistencies (CRITICAL)

**Goal**: Fix mismatches between what prompts say is allowed/forbidden vs what the code actually supports.

**Impact**: HIGH | **Risk**: LOW | **Effort**: LOW

**Problem**: The `dist_system.txt` prompt incorrectly tells LLMs NOT to use features that are actually supported, causing them to avoid valid DSL functions.

**Critical Issues Found**:

1. **DSL Function Allowlist Mismatch**:
   - **Prompt says FORBIDDEN**: `str()`, `int()`, `float()`, `bool()`, `normal()`, `lognormal()`, `pareto()`
   - **Code actually allows**: All of these are in `ALLOWED_FUNCS` in `derived_program.py`
   - **Impact**: LLMs avoid using valid functions, leading to suboptimal derived expressions

2. **Seasonal Granularity Missing**:
   - **Prompt says**: `granularity: "month" | "week"`
   - **Code supports**: `"month" | "week" | "hour"`
   - **Impact**: LLMs can't specify hourly seasonality even though it's supported

3. **Window Functions Confusion**:
   - **Prompt says**: "You CANNOT use window functions like LAG() or LEAD()"
   - **Reality**: `lag()` and `lead()` ARE supported in `DistWindow` expressions (e.g., `expression: "lag(amount, 1)"`)
   - **Impact**: LLMs think lag/lead are forbidden when they're actually valid in window specs

4. **Prompt Length/Domain Bias**:
   - `dist_system.txt` is 761 lines, heavily focused on fraud/finance patterns
   - Risk: LLMs may over-eagerly create fraud-related columns even in non-finance domains
   - **Impact**: Schema bloat and domain mismatches

5. **Missing Distribution Usage Guidance**:
   - **Zipf**: No "Use for" or "When NL mentions" guidance - only has parameters
   - **Categorical**: No "Use for" guidance - only has examples
   - **No Quick Reference**: Missing decision tree or quick selection guide for choosing distributions
   - **Impact**: LLMs may not know when to use Zipf vs other distributions, or when categorical is appropriate

**Files to Update**:
- `prompts/roles/dist_system.txt` - Fix ALLOWED/FORBIDDEN functions list, add "hour" to seasonal, clarify window functions
- `prompts/roles/dist_user.txt` - Clarify lag/lead distinction

**Steps**:
1. Update "ALLOWED FUNCTIONS" section in `dist_system.txt` to match `ALLOWED_FUNCS` from `derived_program.py`:
   - Add: `int()`, `float()`, `bool()`, `str()`, `normal()`, `lognormal()`, `pareto()`, `concat()`, `format()`, `substring()`, `case_when()`, `between()`, `geo_distance()`, `ts_diff()`, `overlap_days()`
   - Remove from FORBIDDEN list: All the above functions
   - Clarify: Distribution *kinds* (e.g., `{"kind": "lognormal"}`) vs distribution *functions* (e.g., `lognormal(mean, sigma)` in DSL) are different things

2. Fix seasonal granularity:
   - Change: `granularity: "month" | "week"` → `granularity: "month" | "week" | "hour"`

3. Clarify window functions:
   - Add section: "Window Functions vs Derived Expressions"
   - Explain: `lag()`/`lead()` are FORBIDDEN in derived expressions, but ALLOWED in window distribution expressions
   - Example: `{"kind": "window", "expression": "lag(amount, 1)", ...}` is valid

4. Reorganize `dist_system.txt`:
   - Split into: Core section (distributions, DSL, providers) + Domain Patterns section (fraud, pay-day, velocity)
   - Make domain-specific sections explicitly conditional: "IF NL mentions fraud/credit-cards, THEN apply these rules..."

5. Add missing distribution usage guidance:
   - **Zipf**: Add "Use for" and "When NL mentions" sections:
     ```
     6. Zipf: {"kind": "zipf", "s": NUMBER, "n": INTEGER | null}
        - Use for discrete popularity distributions (e.g., product popularity, user activity, page views, SKU sales)
        - s = exponent (typically 1.2-2.0), n = domain size
        - Example: {"kind": "zipf", "s": 1.2, "n": 1000}
        - When NL mentions: "Zipf", "popularity", "power-law", "skewed", "80/20", "top products", "most popular"
     ```
   - **Categorical**: Add "Use for" guidance:
     ```
     8. Categorical: {"kind": "categorical", "domain": {...}}
        - Use for discrete values with known options (e.g., status codes, categories, boolean flags, enums, types)
        - values MUST be an array of STRINGS (convert booleans/numbers to strings)
        - When NL mentions: "categories", "status", "types", "enums", "discrete values", "options"
        - Example: {"kind": "categorical", "domain": {"values": ["true", "false"], "probs": [0.3, 0.7]}}
     ```

6. Add Quick Distribution Selection Guide:
   - Add a new section at the top of distribution types:
     ```
     ## Quick Distribution Selection Guide
     
     Choose a distribution based on your data pattern:
     - **Numeric, no pattern/range**: Uniform
     - **Numeric, symmetric bell curve**: Normal
     - **Numeric, right-skewed/heavy tail**: Lognormal or Pareto
     - **Numeric, multi-modal (multiple peaks)**: Mixture
     - **Discrete popularity/skew (power-law)**: Zipf
     - **Date/time with seasonal patterns**: Seasonal
     - **Discrete known values (categories, status)**: Categorical
     - **Computed from other columns**: Derived
     - **Rolling aggregations (sum, mean, count over windows)**: Window
     ```

**Sanity Checklist**:
- [ ] ALLOWED_FUNCS list in prompt matches `derived_program.py` exactly
- [ ] Seasonal granularity includes "hour"
- [ ] Window function section clarifies lag/lead distinction
- [ ] Domain-specific patterns are clearly marked as conditional
- [ ] Zipf distribution has "Use for" and "When NL mentions" guidance
- [ ] Categorical distribution has "Use for" guidance
- [ ] Quick Distribution Selection Guide added at top of distribution types section
- [ ] Test with 2-3 queries to verify LLMs use previously-forbidden functions correctly

---

### Task 2: Add Poisson & Exponential Distributions

**Goal**: Add support for count distributions (Poisson) and inter-arrival time distributions (Exponential).

**Impact**: HIGH | **Risk**: LOW | **Effort**: LOW (~1-2 hours)

**Problem**: Current distributions lack support for:
- **Count patterns**: Session lengths, event counts, arrivals (Poisson)
- **Time intervals**: Inter-arrival times, waiting times, lifetimes (Exponential)

**Use Cases from Queries**:
- Query #1: "Session lengths must follow a heavy-tailed distribution" → Poisson + Pareto mixture
- Query #2: "random missing intervals" → Exponential for inter-arrival times
- Query #8: "Time gaps between sessions" → Exponential
- Query #10: "widely varying time lags" → Exponential or Gamma

**Implementation Steps**:

1. **Add IR Models** (`ir/generation.py`):
   ```python
   class DistPoisson(BaseModel):
       """Poisson distribution specification."""
       kind: Literal["poisson"] = "poisson"
       lam: float  # Lambda parameter (rate, must be > 0)
       
       @field_validator("lam")
       @classmethod
       def validate_lam(cls, v: float) -> float:
           if v <= 0:
               raise ValueError("lam must be positive")
           return v
   
   class DistExponential(BaseModel):
       """Exponential distribution specification."""
       kind: Literal["exponential"] = "exponential"
       scale: float = 1.0  # Scale parameter (1/lambda, must be > 0)
       
       @field_validator("scale")
       @classmethod
       def validate_scale(cls, v: float) -> float:
           if v <= 0:
               raise ValueError("scale must be positive")
           return v
   ```
   - Add to `Distribution` union (after `DistPareto`)

2. **Add Samplers** (`generation/distributions/numeric.py`):
   ```python
   class PoissonSampler(BaseSampler):
       """Poisson distribution sampler."""
       
       def __init__(self, lam: float):
           if lam <= 0:
               raise ValueError("lam must be positive")
           self.lam = lam
           logger.debug(f"Initialized PoissonSampler: λ={lam}")
       
       def sample(self, n: int, **kwargs) -> np.ndarray:
           rng = kwargs.get("rng", np.random.default_rng())
           return rng.poisson(self.lam, size=n)
   
   class ExponentialSampler(BaseSampler):
       """Exponential distribution sampler."""
       
       def __init__(self, scale: float):
           if scale <= 0:
               raise ValueError("scale must be positive")
           self.scale = scale
           logger.debug(f"Initialized ExponentialSampler: scale={scale}")
       
       def sample(self, n: int, **kwargs) -> np.ndarray:
           rng = kwargs.get("rng", np.random.default_rng())
           return rng.exponential(self.scale, size=n)
   ```

3. **Update Factory** (`generation/distributions/factory.py`):
   - Add imports: `DistPoisson`, `DistExponential`, `PoissonSampler`, `ExponentialSampler`
   - Add factory cases:
   ```python
   if isinstance(dist, DistPoisson):
       return PoissonSampler(dist.lam)
   
   if isinstance(dist, DistExponential):
       return ExponentialSampler(dist.scale)
   ```

4. **Update Prompts** (`prompts/roles/dist_system.txt`):
   - Add to distribution types list:
   ```
   8. Poisson: {"kind": "poisson", "lam": NUMBER}
      - Use for count distributions (session lengths, event counts, arrivals)
      - lam: Rate parameter (must be > 0, typically 1-20)
      - Example: {"kind": "poisson", "lam": 5.0}
      - When NL mentions: "Poisson", "count distribution", "session lengths", "event counts", "arrivals"
   
   9. Exponential: {"kind": "exponential", "scale": NUMBER}
      - Use for inter-arrival times, waiting times, lifetimes
      - scale: Scale parameter (1/lambda, must be > 0, default: 1.0)
      - Example: {"kind": "exponential", "scale": 2.5}
      - When NL mentions: "exponential", "inter-arrival", "waiting time", "time between events"
   ```
   - Update kind union: `"uniform" | "normal" | "lognormal" | "pareto" | "poisson" | "exponential" | "mixture" | ...`

5. **Add Unit Tests** (`tests/test_derived_dsl.py` or new file):
   - Test Poisson sampler with various lambda values
   - Test Exponential sampler with various scale values
   - Test edge cases (very small/large parameters)

**Files to Update**:
- `ir/generation.py` - Add `DistPoisson`, `DistExponential` models
- `generation/distributions/numeric.py` - Add `PoissonSampler`, `ExponentialSampler`
- `generation/distributions/factory.py` - Add factory cases
- `generation/distributions/__init__.py` - Export new samplers
- `prompts/roles/dist_system.txt` - Add distribution documentation
- `tests/` - Add unit tests

**Sanity Checklist**:
- [ ] IR models compile and validate
- [ ] Samplers generate correct distributions
- [ ] Factory returns correct samplers
- [ ] Prompts mention Poisson and Exponential
- [ ] Unit tests pass
- [ ] Integration test with Query #1 or #2

**Future Enhancements** (Medium Priority):
- **Gamma distribution**: More flexible than Exponential (shape + scale parameters)
- **Beta distribution**: For proportions, probabilities, bounded values (0-1)
- **Binomial distribution**: For success/failure counts
- **Truncated Normal**: Normal distribution with bounds

---

### Task 3: Enhance Nuance Coverage (Optional Enhancement)

**Goal**: Expand nuance coverage checker to catch more missing patterns.

**Impact**: MEDIUM | **Risk**: LOW | **Effort**: MEDIUM

**Current State**: Basic implementation exists with good coverage (rolling, incident, heavy_tail, log-normal, pareto, mixture, zipf, pay-day, surge, churn, fraud, seasonal)

**Potential Enhancements**:
- Add more keyword patterns (e.g., "readmission", "proration", "within_days_of")
- Improve semantic understanding (e.g., "pay day" → check for day_of_month logic)
- Integrate into repair loop (automatically fix missing nuances)

**Note**: This is already well-implemented. The current `check_nuance_coverage()` function in `validators.py` has comprehensive keyword detection and IR construct checking. Optional enhancements could include:
- More semantic understanding (beyond keyword matching)
- Automatic repair suggestions
- Integration with repair loop

---

## 📋 Later Tasks (Lower Priority)

### Task 4: Add Additional Distributions (Gamma, Beta, Binomial, Truncated Normal)

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

**Implementation**: Follow same pattern as Task 2 (Poisson/Exponential)

**Priority**: Can be done after Poisson/Exponential if needed for specific queries

---

### Task 5: Add Conditional/Correlated Distributions

**Goal**: Enable region-specific categories, fraud rings, correlated metrics.

**Impact**: MEDIUM | **Risk**: MEDIUM | **Effort**: HIGH

**Note**: Likely deferred until after v1.0 - complex implementation, lower immediate impact than Tasks 1-4.

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

### Task 6: Add State Machine Constraints

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

### Task 7: Add `within_days_of()` Helper Function

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

### Task 8: Integrate WorkloadIR into Distribution Engineering

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

## 📚 Reference: Feature Specifications

### DSL Extensions (✅ Done)

**Operators**: `in`, `not in` (membership testing)
**Casts**: `int()`, `float()`, `bool()`, `str()`
**Distributions**: `normal()`, `lognormal()`, `pareto()` (DSL functions)
**Conditionals**: `case_when(cond1, val1, cond2, val2, ..., default)`
**Helpers**: `between(x, a, b)`, `geo_distance()`, `ts_diff()`, `overlap_days()`

**Files**: `derived_program.py`, `derived_eval.py`

---

### Window Functions (✅ Done)

**IR Model**: `DistWindow` with `expression`, `partition_by`, `order_by`, `frame`
**Evaluation**: `window_eval.py` with `compute_window_columns()`
**Integration**: Phase 3 in `fact_generator.py` (after derived columns)

**Supported**: `rolling_mean`, `rolling_sum`, `rolling_count`, `rolling_std`, `lag`, `lead`

---

### Event/Incident System (✅ Done)

**IR Model**: `EventSpec` with `name`, `scope`, `interval`, `effects`
**Effects**: `multiply_distribution`, `add_offset`, `set_value`
**Evaluation**: `event_eval.py` with `apply_events_to_chunk()`
**Integration**: Phase 1.5 in `fact_generator.py` (after base columns, before derived)

---

### Constraint Enforcement (✅ Done)

**Types**: Functional dependencies, implications, nullability
**Enforcement**: `enforce.py` with `enforce_batch()`
**Integration**: Phase 1.7 in `fact_generator.py` (after dimension joins)

---

## 🛠️ Implementation Patterns

### Adding a New Distribution Type

1. **IR Model** (`ir/generation.py`):
   - Add Pydantic model with `kind: Literal["..."]`
   - Add to `Distribution` union (line ~90-100)
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
   - Add to `ALLOWED_FUNCS` (line ~8-34)

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

1. **Forgetting Distribution union**: Always add new distribution types to `Distribution` union in `ir/generation.py` (line ~90-100)

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

---

## 🎯 Next Steps

1. **Task 1 (Fix Prompt Inconsistencies)** - **CRITICAL PRIORITY** - Fixes prevent LLMs from avoiding valid features
2. **Task 2 (Add Poisson & Exponential)** - **HIGH PRIORITY** - Critical for count and time-based patterns in queries
3. Task 3 (Enhanced nuance coverage) - Optional, already well-implemented
4. Later tasks (additional distributions, conditional distributions, state machines, etc.) - Lower priority
