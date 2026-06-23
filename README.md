# Process Mining For ER Department

**Farzad Rahim Khanian**  
Università degli Studi di Milano  
Email: farzad.rahimkhanian@studenti.unimi.it  
Matriculation Number: 987094  
[Project Code](https://github.com/Farzad-RK/ERDepartmentProcessMining)

---

## 1. Preprocessing

### 1.1 Case-level Missing Data Analysis

To ensure the integrity of our process mining analysis, we conducted a comprehensive data quality assessment focusing on case-level attributes. We identified seven key case attributes from the event log for analysis, as these attributes are essential for case characterization and subsequent analytical procedures. The selected attributes are presented in the table below.

**Selected Case-Level Attributes**

| Attribute | Description |
|---|---|
| case:concept:name | Case identifier |
| case:arrival_transport | Mode of patient arrival |
| case:disposition | Patient discharge disposition |
| case:gender | Patient gender |
| case:race | Patient race/ethnicity |
| case:acuity | Clinical acuity level |
| case:chiefcomplaint | Primary complaint at presentation |

We performed a systematic missing value analysis across all selected case attributes to assess data completeness. The table below presents the results of this analysis, showing both the absolute count and percentage of missing values for each attribute across the entire case population (*n* = 1,820 cases).

**Missing Value Analysis for Case-Level Attributes**

| Attribute | Missing Values | Percentage (%) |
|---|---|---|
| case:concept:name | 0 | 0.00 |
| case:arrival_transport | 0 | 0.00 |
| case:disposition | 0 | 0.00 |
| case:gender | 0 | 0.00 |
| case:race | 0 | 0.00 |
| case:chiefcomplaint | 0 | 0.00 |
| case:acuity | 56 | 3.08 |

The analysis reveals that the dataset exhibits high completeness, with only the `case:acuity` attribute containing missing values. Specifically, 56 cases (3.08% of the total case population) lack acuity information. Given that the acuity attribute represents the clinical severity classification—a critical indicator for process analysis and patient stratification—and considering the relatively low missing data rate, we adopted a listwise deletion approach for this attribute.

Rather than imputing or removing the affected cases entirely, we retained all cases in the dataset but excluded cases with missing acuity values from analyses where this attribute is required, specifically during Key Performance Indicator (KPI) calculations and case clustering procedures. This approach preserves the maximum amount of data for general process analysis while ensuring analytical rigor when acuity-dependent metrics are computed.

### 1.2 Activity-Level Missing Data Analysis

Activity-level attributes are conditionally recorded based on the specific activities performed during a case. To assess data completeness, we employed a case-aware coverage metric that measures the fraction of cases in which each attribute was observed at least once.

The figure below presents the case-level coverage for all activity attributes in the event log, revealing distinct patterns of conditional recording based on clinical workflows.

![Case-level coverage of activity attributes, showing the fraction of cases in which each attribute was recorded at least once.](/figures/activity_level_attributes.png)

**Case-Level Coverage Rates for Activity Attributes**

| Attribute | Coverage Rate |
|---|---|
| time:timestamp | 1.000 |
| concept:name | 1.000 |
| diagnosis_description | 0.998 |
| diagnosis_sequence | 0.998 |
| diagnosis_code | 0.998 |
| pain | 0.981 |
| heartrate | 0.978 |
| resprate | 0.978 |
| o2sat | 0.978 |
| sbp | 0.977 |
| dbp | 0.977 |
| temperature | 0.971 |
| drug_name | 0.828 |
| generic_drug_code | 0.826 |
| administering_nurse_id | 0.669 |
| national_drug_code | 0.561 |
| reconciliation_nurse_id | 0.561 |
| drug_class_code | 0.549 |
| drug_class_classification | 0.549 |
| rhythm | 0.058 |

The analysis reveals three categories: (1) **Vital signs** with near-complete coverage (≥ 0.97) excluding `rhythm`; (2) **conditionally recorded attributes** (0.55–0.83 coverage), such as medication-related fields present only when pharmacological interventions occur; and (3) **sparsely recorded attributes**, notably `rhythm` with only 5.8% coverage.

The observed missing patterns are not missing at random but reflect clinical decision-making and protocol-driven documentation. The absence of medication attributes indicates no medication was administered, not a missing value. Since process discovery and performance analysis rely solely on activity names and timestamps—both with complete coverage—no imputation or removal of events is required. The `rhythm` attribute will be excluded from clustering analysis due to its high missingness rate. This preservation-based strategy maintains fidelity to the underlying clinical processes and enables correct pattern-based feature generation in subsequent analysis stages.

### 1.3 Temporal Validation and Event Aggregation

To ensure temporal consistency, we performed three validation checks on the event log: (1) detection of backward timestamps within cases, (2) verification of domain-specific activity precedence rules, and (3) identification of concurrent activities at identical timestamps.

#### Temporal Validation Results

The temporal validation revealed no chronological inconsistencies. Specifically, no backward timestamps were detected—all events within each case maintain strict chronological order. Domain-based order validation confirmed that all cases adhere to the expected emergency department process flow: every case begins with `Enter the ED`, followed by `Triage in the ED`, and terminates with `Discharge from the ED`. No precedence violations were observed.

However, concurrent activity detection identified a substantial number of events occurring at identical timestamps within cases. The concurrent activity co-occurrence matrix for the raw event log reveals two primary patterns: (1) high self-concurrency in `Medicine reconciliation` (16,692 co-occurrences) and `Medicine dispensations` (3,936 co-occurrences), and (2) `Discharge from the ED` self-concurrency (2,948 co-occurrences). This manifests as inflated activity frequencies, particularly for medication and discharge events.

| Raw Event Log | Aggregated Event Log |
|---|---|
| ![Concurrent activity co-occurrence matrix — raw event log](/figures/raw_log_concurrent_events_heatmap.png) | ![Concurrent activity co-occurrence matrix — aggregated event log](/figures/aggregated_log_concurrent_events_heatmap.png) |

*Figure: Concurrent activity co-occurrence matrices before and after aggregation. Self-concurrency in medication and discharge activities is substantially reduced through nurse-based aggregation.*

| Raw Event Log | Aggregated Event Log |
|---|---|
| ![Activity frequency distribution — raw event log](/figures/raw_log_activity_frequency_distribution.png) | ![Activity frequency distribution — aggregated event log](/figures/aggregated_log_activity_frequency_distribution.png) |

*Figure: Activity frequency distributions before and after aggregation. Medication activities show significant reduction, while single-instance activities remain unchanged.*

#### Event Aggregation Strategy

The observed concurrent activities do not represent true parallelism but rather logging system artifacts where multiple related records (e.g., multiple medications, multiple diagnoses) are recorded at a single timestamp. To address this data quality issue while preserving genuine parallel activities, we implemented a nurse-aware aggregation strategy:

1. **Medicine reconciliation:** Events at the same timestamp are aggregated only when performed by the same nurse (identified by `reconciliation_nurse_id`). Events by different nurses at the same timestamp are preserved as separate activities, representing true parallelism.

2. **Medicine dispensations:** Events at the same timestamp are aggregated only when administered by the same nurse (identified by `administering_nurse_id`), using the same logic as medicine reconciliation.

3. **Discharge from the ED:** All diagnosis codes at the same timestamp are aggregated into a single discharge event, as multiple diagnoses represent a single clinical decision rather than parallel processes.

4. **Other activities:** No aggregation is performed for `Enter the ED`, `Triage in the ED`, and `Vital sign check`, as these typically occur as single events.

Aggregated attributes are preserved as JSON arrays to maintain full data fidelity. Each aggregated event is tagged with metadata (`aggregated_event_count`, `is_aggregated`) to enable traceability to the original raw data.

#### Aggregation Impact

The aggregation procedure reduced the event log from 25,115 events to 18,316 events (27.1% reduction) while maintaining all 1,820 cases. The aggregated log shows substantial reduction in self-concurrency: `Medicine reconciliation` self-concurrency decreased from 16,692 to 741, and `Discharge from the ED` self-concurrency was completely eliminated (from 2,948 to 0). The aggregated log now exhibits one discharge event per case, accurately reflecting the clinical reality that each patient receives a single discharge decision. This transformation ensures that subsequent process discovery and performance analysis reflect actual process behavior rather than data recording artifacts.

---

## 2. Performance Analysis

Following data preprocessing and aggregation, we conducted a comprehensive performance analysis across three analytical levels: case-level, activity-level, and resource-level. This multi-dimensional approach enables identification of process bottlenecks, resource utilization patterns, and opportunities for workflow optimization.

### 2.1 Key Performance Indicators

The table below presents the complete set of KPIs calculated in this study, organized by analytical level, along with their operational definitions and benchmark standards for emergency department processes.

**Key Performance Indicators and Emergency Department Benchmarks**

| KPI | Level | Definition | ED Standard |
|---|---|---|---|
| **Case-Level KPIs** | | | |
| Length of Stay (LOS) | Case | Total time from ED entry to discharge | < 4 hours (target); < 6 hours (acceptable) |
| Case Complexity | Case | Number of events per case | N/A (varies by acuity) |
| **Activity-Level KPIs** | | | |
| Cumulative Lead Time | Activity | Time from case start when activity occurs | N/A (process-specific) |
| **Domain-Specific KPIs** | | | |
| Door-to-Triage (DTT) | Process | Time from ED entry to triage completion | < 10 minutes |
| Door-to-Treatment (DTTreat) | Process | Time from ED entry to first treatment | < 60 minutes |
| Triage-to-Treatment (TTT) | Process | Time from triage to first treatment intervention | < 30 minutes |
| Treatment Duration | Process | Time from first treatment to discharge | N/A (varies by condition) |
| **Resource-Level KPIs** | | | |
| Nurse Workload | Resource | Events/cases handled per nurse | N/A (balanced distribution) |
| LOS by Acuity | Stratified | Length of stay stratified by clinical urgency | Level 1: immediate; Levels 2–3: < 4h; Levels 4–5: < 2h |

### 2.2 Case-Level Performance

#### Length of Stay Distribution

The overall length of stay distribution across all cases exhibits a right-skewed pattern with the following statistical characteristics:

- **Median LOS:** 5.0 hours (within acceptable ED standards)
- **Mean LOS:** 6.8 hours (exceeds 6-hour benchmark)
- **Standard deviation:** 7.1 hours (high variability)
- **Range:** 0.1–84.4 hours

The median LOS of 5.0 hours falls within the acceptable ED performance threshold (< 6 hours), suggesting that the majority of cases are processed efficiently. However, the mean LOS of 6.8 hours exceeds this benchmark, driven by a subset of complex cases requiring extended stays. The substantial standard deviation (7.1 hours) reflects heterogeneity in case complexity and clinical pathways. The distribution shows right skewness (mean > median), with most cases resolved within 10 hours but a long tail extending to 84.4 hours.

![Case-level length of stay distribution.](/figures/case_los_distribution.png)

*Figure: Case-level length of stay distribution. The distribution shows right skewness (mean > median), with most cases resolved within 10 hours but a long tail extending to 84.4 hours.*

#### Case Complexity Distribution

Case complexity, measured as the number of events per case, demonstrates the following distribution characteristics:

- **Median:** 8 events per case
- **Mean:** 10.1 events per case
- **Standard deviation:** 6.3 events
- **Range:** 3–73 events

The distribution reveals that most cases follow relatively simple pathways (median: 8 events), while a minority requires extensive interventions (maximum: 73 events). Cases in the upper tail likely correspond to high-acuity patients requiring multiple vital sign assessments, medications, and diagnostic procedures.

![Case complexity distribution showing the number of events per case.](/figures/case_length_distribution.png)

*Figure: Case complexity distribution showing the number of events per case. Higher event counts indicate more complex clinical pathways with multiple interventions.*

### 2.3 Activity-Level Performance

The cumulative lead time analysis reveals the temporal progression of activities throughout the care process. Activities are ordered chronologically by their mean occurrence time from case initiation:

- **Enter the ED & Triage in the ED:** 0.00h (case start, confirming proper temporal ordering)
- **Medicine reconciliation:** 1.99h mean (timely medication history assessment)
- **Vital sign check:** 5.18h mean (distributed throughout case trajectory)
- **Medicine dispensations:** 5.57h mean (treatment delivery mid-process)
- **Discharge from the ED:** 6.81h mean (final activity, matching overall LOS)

The substantial standard deviations and wide distributions reflect variability in clinical pathways based on patient acuity and treatment requirements.

![Cumulative lead time by activity.](/figures/comulative_lead_time.png)

*Figure: Cumulative lead time by activity. Left panel shows mean ± standard deviation with median markers; right panel displays distribution via box plots. Activities are ordered chronologically by mean occurrence time.*

### 2.4 Domain-Specific Performance

![Domain-specific ED performance indicators.](/figures/domain_specific_kpis.png)

*Figure: Domain-specific ED performance indicators. Top row: histograms for Door-to-Triage, Door-to-Treatment, Triage-to-Treatment, and Treatment Duration. Bottom left: KPI comparison via box plots. Bottom right: typical case timeline showing mean duration per phase.*

**Domain-Specific KPI Performance vs. Benchmarks**

| KPI | Mean | Median | Benchmark | Assessment |
|---|---|---|---|---|
| Door-to-Triage | 0.00h (0 min) | 0.00h (0 min) | < 10 min | **Acceptable** (could also be due to logging artifacts — immediate triage) |
| Door-to-Treatment | 1.60h (96 min) | 1.05h (63 min) | < 60 min | ⚠️ **Below standard** (60% over target) |
| Triage-to-Treatment | 1.60h (96 min) | 1.05h (63 min) | < 30 min | ⚠️ **Below standard** (220% over target) |
| Treatment Duration | 5.67h | 3.77h | N/A | Variable by condition (SD: 7.03h) |

#### Critical Findings

**Door-to-Triage Performance:** Exceptional performance with immediate triage (0.00h mean and median). All 1,820 cases received triage assessment simultaneously with ED entry, indicating efficient patient intake processes.

**Door-to-Treatment Delay:** Mean of 1.60 hours (96 minutes) significantly exceeds the 60-minute benchmark. The median of 1.05 hours (63 minutes) also surpasses the target, with high variability (SD: 1.98h). This represents a critical bottleneck in the care delivery pathway.

**Triage-to-Treatment Gap:** Mean of 1.60 hours (96 minutes) substantially exceeds the 30-minute standard. This 220% overage indicates delays between initial assessment and therapeutic intervention initiation, suggesting potential resource constraints or workflow inefficiencies.

**Treatment Duration:** Mean of 5.67 hours with substantial variability (SD: 7.03h) reflects case heterogeneity. The right-skewed distribution (mean > median) indicates a subset of cases requiring prolonged treatment.

**Process Composition:** The typical case timeline reveals that treatment duration constitutes 83% (5.67h/6.81h) of total LOS, while triage-to-treatment waiting time comprises 24% (1.60h/6.81h), highlighting the primary delay point in the care process.

### 2.5 Resource-Level Performance

#### Nurse Workload Distribution

Workload analysis across 44 nursing staff (3 medicine reconciliation nurses, 41 medicine administration nurses) reveals distinct patterns by job type.

**Medicine Reconciliation Nurses:**
- **Nurse 1:** 2,120 events across 1,021 cases (2.08 events/case)
- **Nurse 2:** 573 events across 471 cases (1.22 events/case)
- **Nurse 3:** 84 events across 82 cases (1.02 events/case)

The workload imbalance is pronounced: Nurse 1 handles 76% of all medicine reconciliation activities (2,120/2,777 events), suggesting either role specialization or inequitable task distribution. The higher events-per-case ratio (2.08) for Nurse 1 indicates handling of more complex medication histories.

**Medicine Administration Nurses:** All 41 administration nurses exhibit 1.00 events per case, indicating a standardized one-medication-administration-per-case pattern. The top nurse (Admin_1) handled 1,218 events, while workload distribution shows substantial variation (mean: 102 events, median: 4 events), with most nurses handling relatively few cases. This suggests either part-time staffing, shift-based rotation, or role specialization within the administration team.

![Nurse workload analysis.](/figures/nurse_workload_plot.png)

*Figure: Nurse workload analysis. Top left: all nurses ranked by events handled. Top right: top 15 administration nurses showing events vs. cases. Bottom left: workload distribution comparison by job type. Bottom right: efficiency analysis plotting cases handled vs. events per case.*

#### Length of Stay by Acuity Level

Stratification of length of stay by clinical acuity level reveals the following:

- **Level 1** (Most Urgent, *n* = 129): Mean 5.8h, Median 4.6h
- **Level 2** (*n* = 609): Mean 8.0h, Median 6.1h
- **Level 3** (*n* = 877): Mean 6.1h, Median 4.9h
- **Level 4** (*n* = 142): Mean 3.0h, Median 2.9h
- **Level 5** (Least Urgent, *n* = 7): Mean 2.7h, Median 1.5h

**Unexpected Pattern:** Level 2 patients exhibit the longest LOS (mean: 8.0h), exceeding even Level 1 patients (mean: 5.8h). This counterintuitive finding suggests Level 2 patients, while not requiring immediate resuscitation, may require extensive diagnostic workup or specialist consultation, resulting in prolonged stays. Level 1 patients may benefit from streamlined critical care protocols enabling faster disposition decisions.

A Kruskal-Wallis test confirms significant differences in LOS across acuity levels (*H* = 125.43, *p* < 0.001), validating acuity as a meaningful stratification variable. The analysis reveals a non-linear relationship with peak LOS at intermediate acuity levels.

![Length of stay stratified by acuity level.](/figures/los_by_acuity.png)

*Figure: Length of stay stratified by acuity level. Top row: box plots and violin plots. Bottom left: mean vs. median comparison. Bottom right: scatter plot with quadratic trend line and mean markers.*

### 2.6 Performance Summary

**Strengths:** Immediate triage (0.00h DTT) and acceptable median LOS (5.0h) demonstrate efficient patient intake processes.

**Critical Bottleneck:** Triage-to-treatment delay (mean: 1.60h) represents a 220% exceedance of the 30-minute standard, indicating the primary target for process improvement interventions.

**Resource Concerns:** Significant workload imbalance among reconciliation nurses, with one nurse handling 76% of all activities, suggests need for task redistribution.

**Unexpected Finding:** Level 2 acuity patients demonstrate the longest LOS (8.0h mean), suggesting diagnostic complexity exceeds that of Level 1 critical cases.

These findings indicate that while patient intake is efficient, post-triage workflow optimization is essential to meet ED performance benchmarks. The 96-minute average wait for treatment initiation represents the primary target for process improvement interventions.

---

## 3. Variant Analysis

Process variants represent unique sequences of activities observed across cases. We employed PM4Py's variant discovery algorithm to analyze pathway diversity in the aggregated event log.

![Comprehensive variant analysis.](/figures/variant_analysis.png)

*Figure: Comprehensive variant analysis. Top left: frequency of top 20 variants. Top right: power-law distribution test (log-log scale). Bottom left: Pareto chart with cumulative coverage. Bottom right: case distribution by variant length.*

### 3.1 Distribution Characteristics

The analysis reveals substantial process variability: 1,820 cases produce over 1,000 unique variants (variants-to-cases ratio: 0.55:1). The most frequent variant accounts for only 4.2% of cases (77 cases), with frequency declining rapidly thereafter. This fragmentation indicates adaptive pathways responding to individual patient needs rather than standardized protocols.

**Power-Law Distribution Test:** Log-log linear regression yields Zipf exponent α = 0.55 with R² = 0.802, confirming moderate-to-strong power-law behavior. The relatively low exponent indicates gradual frequency decline, supporting high process variability.

**Cumulative Coverage:** Achieving 80% case coverage requires 609 variants (60% of all unique variants), substantially higher than the 5–20 variants typical of standardized processes. The Pareto chart shows gradual accumulation beyond the top variants, demonstrating absence of dominant pathways.

**Variant Length:** Weighted mean of 10.1 activities per variant, with distribution ranging from 3 to 73 activities. The peak at 8–10 activities represents typical ED pathways (entry, triage, assessment, treatment, discharge), while the long tail captures complex cases requiring extensive interventions.

### 3.2 Implications and Limitations

The high variant count (> 1,000) highlights a critical limitation of sequential variant definitions: cases differing only in the order of independent activities (e.g., vital signs and medication reconciliation reversed) are treated as separate variants despite functional equivalence. This strict definition inflates variant counts in healthcare processes where activity order flexibility is clinically acceptable.

**Need for Relaxed Definitions:** Future analysis should employ trace clustering or hierarchical abstraction to identify functionally similar pathways. Approaches accounting for activity order flexibility, optional activities, and activity abstraction would better capture underlying process structure while accommodating necessary clinical adaptability.

The exceptional process heterogeneity reflects inherent ED complexity, where clinical pathways must adapt to diverse patient presentations and acuities. Traditional process standardization approaches requiring rigid pathway compliance may be inappropriate for emergency care workflows.

---

## 4. Process Discovery and Conformance Checking

### 4.1 Methodology

Process discovery aims to construct process models that accurately represent observed behavior while maintaining interpretability. We employed three discovery algorithms with systematic parameter tuning:

1. **Inductive Miner (IM):** Guarantees sound models through process tree discovery with noise filtering (tested noise thresholds: 0.0–0.8)
2. **Heuristics Miner (HM):** Handles infrequent behavior and parallelism through dependency graph construction (tested dependency thresholds: 0.3–0.9, AND thresholds: 0.3–0.8)
3. **Alpha Miner:** Baseline algorithm assuming complete and noise-free logs

Model quality was assessed through three complementary metrics:
- **F1 Score:** Harmonic mean of fitness (log replay capability) and precision (behavioral specificity): F1 = 2 · (fitness × precision) / (fitness + precision)
- **Generalization:** Model's ability to handle unseen but similar behavior
- **Simplicity:** Structural parsimony (inversely related to model complexity)

To identify optimal models balancing these competing objectives, we employed Pareto-based multi-objective optimization. A model is Pareto-optimal (Pareto Front, Rank 1) if no other model strictly dominates it across all three metrics simultaneously.

### 4.2 Discovery Results

A comprehensive parameter search evaluated 27 model configurations across the three algorithms. The table below presents all Pareto-optimal models identified through multi-objective ranking.

**Pareto Front Models (Rank 1) — Non-Dominated Solutions**

| Model | Fitness | Precision | F1 | Gen | Simplicity | Places | Trans | Arcs |
|---|---|---|---|---|---|---|---|---|
| **IM_noise_0.3** | **0.967** | **0.902** | **0.933** | **0.974** | **0.744** | 13 | 16 | 34 |
| IM_noise_0.5 | 0.909 | 0.822 | 0.863 | 0.901 | 0.750 | 11 | 13 | 28 |
| IM_noise_0.6 | 0.909 | 0.822 | 0.863 | 0.901 | 0.750 | 11 | 13 | 28 |
| IM_noise_0.2 | 0.981 | 0.757 | 0.855 | 0.975 | 0.700 | 13 | 15 | 34 |
| IM_noise_0.1 | 0.993 | 0.720 | 0.835 | 0.975 | 0.700 | 13 | 15 | 34 |
| IM_noise_0.7 | 0.887 | 0.780 | 0.830 | 0.899 | 0.806 | 11 | 14 | 28 |
| IM_noise_0.8 | 0.887 | 0.780 | 0.830 | 0.899 | 0.806 | 11 | 14 | 28 |
| Alpha Miner | 0.867 | 0.761 | 0.810 | 0.980 | **1.000** | 4 | 6 | 7 |

**Key Findings:**
- **Best Overall Model:** Inductive Miner with noise threshold 0.3 achieves the highest F1 score (0.933) while maintaining strong generalization (0.974) and acceptable simplicity (0.744).
- **Algorithm Dominance:** Inductive Miner variants constitute 7 of 8 Pareto-optimal models, demonstrating superior balance across quality dimensions for this high-variability log.
- **Alpha Miner Limitations:** While achieving perfect simplicity (1.0) with minimal structure (4 places, 6 transitions), Alpha Miner exhibits the lowest F1 score (0.810), indicating inability to capture process complexity.
- **Heuristics Miner:** All configurations ranked below Pareto front (Rank 2+), suggesting suboptimal trade-offs for this specific process despite extensive parameter tuning.

### 4.3 Multi-Objective Trade-off Analysis

![Pareto-based multi-objective analysis.](/figures/pareto_analysis.png)

*Figure: Pareto-based multi-objective analysis. Top panels: 2D projections showing F1 vs. Generalization (left) and F1 vs. Simplicity (right) with Pareto front marked by stars. Bottom left: 3D Pareto space. Bottom right: best scores by algorithm.*

The 2D projections reveal inherent trade-offs: models with higher F1 scores (better fitness-precision balance) tend toward lower simplicity (more complex structures needed to capture process variability). Conversely, Alpha Miner's maximal simplicity comes at the cost of reduced F1 performance. Inductive Miner variants dominate the Pareto front, occupying diverse positions that balance the three objectives differently.

The algorithm comparison demonstrates Heuristics Miner achieves competitive F1 scores (best: 0.89) and generalization (best: 1.0) but fails to reach the Pareto front due to lower simplicity scores, indicating models with unnecessary structural complexity for the achieved quality.

### 4.4 Parameter Sensitivity Analysis

![Parameter sensitivity analysis.](/figures/parameter_sensitivity.png)

*Figure: Parameter sensitivity analysis. Left: Inductive Miner performance across noise thresholds. Right: Heuristics Miner F1 scores across dependency and AND threshold combinations.*

**Inductive Miner Sensitivity:** F1 score peaks at noise threshold 0.3 (0.933), declining at both extremes. Lower thresholds (0.0–0.2) retain excessive infrequent behavior, reducing precision. Higher thresholds (0.5–0.8) over-generalize, sacrificing fitness. Generalization remains stable (> 0.89) across all thresholds, while simplicity increases monotonically with noise filtering as models become more compact.

**Heuristics Miner Sensitivity:** Optimal F1 performance occurs at dependency threshold 0.3 with AND threshold 0.8 (F1 = 0.888). Lower dependency thresholds (0.3) capture more behavioral nuances but risk overfitting. The AND threshold shows minimal impact on F1 scores when dependency is held constant, suggesting dependency threshold is the primary tuning parameter for this algorithm on this log.

### 4.5 Process Model Interpretation

#### Optimal Model Structure

![Optimal process model discovered by Inductive Miner (noise threshold 0.3).](/figures/petri_net_1_IM_noise_0_3.png)

*Figure: Optimal process model discovered by Inductive Miner (noise threshold 0.3). The model captures parallel execution of medication activities and vital sign monitoring, with choice constructs for optional pathways. F1 = 0.933, Generalization = 0.974, Simplicity = 0.744.*

The Inductive Miner model (noise threshold 0.3) reveals key process patterns:

1. **Sequential Core:** Mandatory sequence of *Enter the ED* → *Triage in the ED* → *Discharge from the ED* establishes the fundamental pathway structure.

2. **Parallel Activities:** The model captures concurrency between *Vital sign check* and medication activities (*Medicine reconciliation*, *Medicine dispensations*), reflected by parallel gateway structures. This aligns with clinical reality where monitoring and treatment occur simultaneously.

3. **Optional Pathways:** Choice constructs accommodate cases where certain medication activities are skipped (e.g., patients requiring only reconciliation or only dispensation), reflecting variable treatment protocols.

#### Alpha Miner Deficiencies

![Alpha Miner process model demonstrating algorithm limitations.](/figures/alpha_miner_petri_net.png)

*Figure: Alpha Miner process model demonstrating algorithm limitations. The overly simplified structure (4 places, 6 transitions) treats all medication activities and vital signs as optional branches from the same point, failing to capture sequential dependencies and parallel patterns. F1 = 0.810, Generalization = 0.980, Simplicity = 1.000.*

The Alpha Miner model demonstrates critical limitations for complex healthcare processes:

- **Over-simplification:** All medication activities and vital signs branch directly from triage as independent, mutually exclusive options, ignoring temporal dependencies and parallel execution patterns.
- **Incomplete Flow:** The model suggests patients can proceed directly from triage to discharge via any single activity, contradicting the observed requirement for multiple interventions in most cases.
- **Missing Concurrency:** Parallel medication and monitoring activities are represented as sequential alternatives, fundamentally misrepresenting actual clinical workflows.

While Alpha Miner's perfect simplicity (1.0) reflects minimal structural complexity, the resulting model fails to provide actionable process insights, achieving the lowest F1 score (0.810) among all tested configurations.

### 4.6 Model Selection Recommendation

The Pareto analysis identifies **Inductive Miner with noise threshold 0.3** as the recommended model for emergency department process analysis. This model achieves:

- **Highest F1 score** (0.933) on the Pareto front, indicating optimal fitness-precision balance
- **Strong generalization** (0.974), ensuring robustness to unseen cases
- **Acceptable simplicity** (0.744), maintaining interpretability while capturing essential process complexity

The model successfully balances accuracy, precision, and parsimony, making it suitable for both descriptive process analysis and prescriptive improvement identification. The noise threshold of 0.3 effectively filters artifacts from the high-variability log (> 1,000 variants) while preserving clinically significant process patterns.

---

## 5. Feature-Based Variant Analysis

To address the high sequential variant count (> 1,000 variants), we relax the variant definition from exact activity sequences to pattern-based clinical profiles. Each case is encoded as a 12-dimensional binary vector representing vital sign thresholds (fever, tachycardia, tachypnea, hypoxemia, hypertension), transport mode (ambulance, walk-in, helicopter), and clinical features (medication received, high acuity). Cases are clustered using K-Medoids with Hamming distance, yielding three distinct patient cohorts.

### 5.1 Clustering Results and Characterization

![Cluster coverage analysis showing distribution, sizes with medoid IDs, and feature composition.](/figures/cluster_coverage.png)

*Figure: Cluster coverage analysis showing distribution (Panel A), sizes with medoid IDs (Panel B), and feature composition (Panel C).*

![Feature prevalence by cluster.](/figures/feature_prevalence_heatmap.png)

*Figure: Feature prevalence by cluster. Cell values represent fraction of cases exhibiting each feature.*

![Top 20 pattern-based variants.](/figures/top_20_variants.png)

*Figure: Top 20 pattern-based variants covering 58.1% of cases (1,057/1,820). Total unique patterns: 238.*

**Pattern-based variant analysis yields:**
- 238 unique patterns (76% reduction from > 1,000 sequential variants)
- Top 20 patterns cover 58.1% of cases (vs. 20% for sequential variants)
- Three clinically distinct clusters identified

**Cluster Profiles:**

**Cluster 0 (46.3%, n = 843) — Routine Walk-in Patients:** Low vital sign abnormalities, moderate pain (40%), high walk-in arrival (97%). Stable presentations requiring standard pharmacological care.

**Cluster 1 (16.5%, n = 301) — Cardiac/Hypertensive Emergencies:** Universal tachycardia (100%), high hypertension (SBP 87%, DBP 75%), high pain (73%). Time-sensitive cardiovascular conditions requiring intensive monitoring.

**Cluster 2 (37.1%, n = 676) — Pain-Dominant Ambulatory Cases:** High pain prevalence (86%), dominant walk-in arrival (97%), moderate hypertension, no tachycardia (0%). Analgesia-focused care for musculoskeletal or chronic pain presentations.

### 5.2 Process Improvement Recommendations

**Cluster 0 — Throughput Optimization (46.3% of cases)**
Target: Reduce LOS from 6.8h to < 4h
- Implement nurse-led fast-track pathway for stable walk-in patients
- Deploy standing medication orders to bypass physician bottleneck
- Create dedicated discharge planning zone for routine cases

**Cluster 1 — Time-Sensitive Cardiac Pathway (16.5% of cases)**
Target: Reduce door-to-treatment from 1.60h to < 1h (60-minute benchmark)
- Triage-triggered automatic ECG and troponin orders for tachycardic + hypertensive patients
- Pre-allocate dedicated treatment bay for cardiac presentations
- Implement parallel processing: vital signs + medication reconciliation concurrent with diagnostic workup
- Direct cardiology consultation pathway bypassing general ED queue

**Cluster 2 — Early Analgesia Protocol (37.1% of cases)**
Target: Reduce triage-to-treatment from 1.60h to < 30 minutes for analgesic administration
- Standing analgesic orders at triage for pain scores > 5
- Nurse-administered pain protocol without physician approval
- Dedicated pain management zone with parallel care streams

### 5.3 Expected Impact

Targeted interventions for Clusters 1 and 2 (53.6% of cases) directly address the critical triage-to-treatment bottleneck (current: 1.60h, 220% over 30-minute benchmark). Fast-tracking Cluster 0 (46.3% of cases) reduces congestion for time-sensitive presentations. Pattern-based clustering enables predictive triage classification for proactive resource allocation rather than reactive management.

**Variant Analysis Comparison**

| Metric | Sequential | Pattern-Based |
|---|---|---|
| Total unique variants | > 1,000 | 238 |
| Top 20 coverage | 20% | 58.1% |
| Variants for 80% coverage | 609 (60%) | 95 (40%) |
| Actionability | Low | High |

---

## 6. Knowledge Uplift Trail

**Knowledge Uplift Trail for Emergency Department Process Mining Analysis**

| Step | Input | Transformation / Analysis | Output | Knowledge Type |
|---|---|---|---|---|
| **1 — Data Collection** | Raw CSV file with 25,115 events | Load event log, rename columns to PM4Py standard format, convert timestamps, validate structure | Structured event log: 1,820 cases, 6 activities, 25,115 events | World Knowledge (raw observations of ED processes) |
| **2 — Missing Data Analysis** | Structured event log | Calculate case-level missing rates, apply case-aware coverage, identify NMAR mechanism | Missing data report: acuity missing 3.08% (listwise deletion), rhythm 94.2% missing (excluded) | Epistemic Knowledge (discovered NMAR missingness reflects clinical workflows) |
| **3 — Temporal Validation** | Event log with timestamps | Detect backward timestamps, verify precedence, identify concurrent activities | No temporal anomalies; high concurrency detected (16,692 medicine reconciliation events at same timestamp) | Epistemic Knowledge (discovered logging artifacts from batch recording system) |
| **4 — Event Aggregation** | Raw log with concurrent events | Apply nurse-aware aggregation by (case, timestamp, activity, nurse_id) | Aggregated log: 18,316 events (27.1% reduction), one discharge per case | World Knowledge (corrected representation removing artifacts) |
| **5 — Performance KPI Calculation** | Aggregated log, ED benchmarks | Calculate case-, activity-, domain-, and resource-level KPIs against benchmarks | Mean LOS 6.8h, DTTreat 1.60h (60% over target), TTT 1.60h (220% over target, critical bottleneck) | Epistemic Knowledge (measured performance gaps) |
| **6 — Sequential Variant Discovery** | Aggregated log | Test 3 algorithms, 27 configurations; Pareto ranking | Best: IM noise=0.3 (F1=0.933, Gen=0.974, Simp=0.744) | Epistemic Knowledge (discovered optimal model) |
| **7 — Sequential Variant Analysis** | Aggregated log | Extract activity sequences, calculate frequency, power-law fit | 1,057 unique variants, Zipf exponent α=0.55, R²=0.802 | Epistemic Knowledge (discovered inadequacy of sequential definition) |
| **8 — Feature-Based Variant Analysis** | Aggregated log, clinical thresholds | Encode cases as 12-dimensional binary vectors, K-Medoids clustering (Hamming distance) | 238 pattern-based variants (76% reduction); three clusters | Epistemic Knowledge (discovered clinically meaningful patient cohorts) |
| **9 — Cluster Characterization** | Feature prevalence matrix | Calculate feature prevalence per cluster, interpret clinical profiles | C0=stable routine (97% walk-in), C1=cardiac emergencies (100% tachycardia), C2=pain-dominant (86% pain) | Conceptual Knowledge (understood distinct patient archetypes) |
| **10 — Performance-Cluster Integration** | Performance metrics, cluster profiles | Map performance gaps to cluster characteristics, compare against benchmarks | Cluster-specific gap analysis and bottleneck identification per cohort | Conceptual Knowledge (root cause understanding) |
| **11 — Process Improvement Recommendations** | Cluster-specific gaps, process model | Design targeted interventions per cluster, estimate expected impact | Intervention portfolio for 100% of cases: C0 LOS→4h, C1 DTTreat→1h, C2 TTT→0.5h | Conceptual Knowledge → Human Artefact (prescriptive redesign recommendations) |

---

## 7. Conclusion

This process mining study analyzed 1,820 emergency department cases to identify performance bottlenecks and develop targeted improvement strategies. The analysis progressed through data quality enhancement, performance measurement, process discovery, and feature-based patient stratification.

### Key Findings

**Data Quality:** The raw event log contained 27.1% redundant events due to batch recording artifacts. Nurse-aware aggregation reduced 25,115 events to 18,316 while preserving genuine parallelism, yielding clinically accurate process representations.

**Performance Gaps:** Three critical findings emerged:
- **Triage-to-treatment delay:** Mean 1.60h (96 minutes) exceeds 30-minute benchmark by 220%, representing the primary bottleneck
- **Door-to-treatment time:** 60% over target, indicating delayed therapeutic intervention
- **Resource imbalance:** One nurse handles 76% of medication reconciliations; Level 2 acuity patients experience longest LOS (8.0h)

**Process Complexity:** Traditional variant analysis revealed extreme heterogeneity: over 1,000 unique sequential variants with poor coverage (609 variants for 80%). This indicated that strict activity sequence definitions inadequately represent healthcare process flexibility.

**Patient Clusters:** Feature-based clustering reduced variant space by 76% (to 238 patterns) while improving top-20 coverage from 20% to 58.1%. Three clinically distinct clusters:
- **Cluster 0 (46.3%):** Routine walk-in patients with stable vitals — candidates for fast-track pathway
- **Cluster 1 (16.5%):** Cardiac emergencies with tachycardia and hypertension — require immediate cardiac protocol
- **Cluster 2 (37.1%):** Pain-dominant ambulatory cases — need early medical intervention

Expected impact: Targeted interventions address the critical triage-to-treatment gap (affecting 53.6% of cases in Clusters 1–2) while optimizing throughput for routine cases, aligning ED performance with organizational benchmarks.

This study demonstrates the value of integrating clinical domain knowledge into variant analysis through pattern-based feature engineering. By transcending strict sequential definitions, the approach enables actionable patient stratification that supports evidence-based process redesign.
