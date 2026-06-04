# Mapping t2 Motivation Categories to SEM Variables
## Analytical Memo — Impact Function Derivation

**Purpose:** This memo documents the reasoning behind the mapping of
participant-reported motivation categories (t2 responses) to SEM knowledge
states (K) and motivation-based strategies (M). It is intended to inform
the model section of the manuscript.

**Data:** Experiment 1, n = 362. After viewing a speaker use a precise or
approximate numerical expression in a high- or low-precision context,
participants provided open-ended explanations of why the speaker chose that
wording. These were coded into binary motivation categories (t2 columns).

---

## 1. The Inferential Direction

The mapping follows the inferential direction of the SEM itself. In the
model, the listener observes an utterance and reasons backwards to infer
the speaker's knowledge state (k) and motivation-based strategy (m). The
t2 motivation categories reflect exactly this reasoning: they are
listener-generated explanations of the speaker's choice, expressed in
lay terms.

The correct approach is therefore to ask: *given that a listener infers
strategy m (or knowledge state k), which lay-language motives would they
plausibly report?* This is the reverse of fitting motives to strategies
post-hoc; it starts from the strategy definition and asks what its
natural verbal expression is in participant language.

---

## 2. Linking Strategies to Experimental Conditions

Each SEM variable produces a characteristic pattern of utterance choices
across the 2×2 design (form: precise/approx × context: highPr/lowPr):

| SEM variable | Produces utterances in | Interpretation |
|---|---|---|
| **k_p** (knows exact) | precise / both contexts | Can truthfully say precise |
| **k_np** (doesn't know) | approx / both contexts | Can only say approx |
| **m_max** (always precise) | precise / both contexts | Maximally precise regardless of context |
| **m_sit** (situationally appropriate) | precise / highPr AND approx / lowPr | Calibrates to contextual demands |
| **m_min** (always approx) | approx / both contexts | Minimally precise regardless of context |

Note that k_p and m_max share the same observable conditions (precise in
both contexts), as do k_np and m_min (approx in both contexts). This means
their motive profiles overlap in the data — what separates them empirically
is the *specific motive* the listener reports, not the condition they were
in.

---

## 3. Motive Profiles per Condition

Table 1 shows the percentage of participants in each condition who checked
each motive. This reveals which motives are characteristic of each
form×context combination.

**Table 1: P(motive checked | condition) — % of participants in condition**

| Motive | n | prc/HP | prc/LP | apx/HP | apx/LP |
|---|---|---|---|---|---|
| notKnow | 94 | 0.0% | 0.0% | 64.1% | 37.6% |
| realWorld | 53 | 38.9% | 6.9% | 9.8% | 3.2% |
| needed | 40 | 23.3% | 5.7% | 3.3% | 11.8% |
| knewExact | 38 | 16.7% | 26.4% | 0.0% | 0.0% |
| fussyPerson | 29 | 7.8% | 24.1% | 0.0% | 1.1% |
| infoNotAvail | 29 | 0.0% | 0.0% | 13.0% | 18.3% |
| purpose | 23 | 12.2% | 6.9% | 1.1% | 5.4% |
| usualApx | 20 | 0.0% | 0.0% | 7.6% | 14.0% |
| infoAvail | 17 | 7.8% | 11.5% | 0.0% | 0.0% |
| easySpeaker | 9 | 0.0% | 0.0% | 2.2% | 7.5% |
| presentation | 9 | 2.2% | 4.6% | 2.2% | 1.1% |
| emphasis | 8 | 0.0% | 6.9% | 0.0% | 2.2% |
| helpful | 7 | 4.4% | 2.3% | 1.1% | 0.0% |
| cautious | 6 | 0.0% | 0.0% | 4.3% | 2.2% |

Condition sizes: prc/HP n=90, prc/LP n=87, apx/HP n=92, apx/LP n=93.

The table reveals a clear partition: `notKnow`, `infoNotAvail`, `usualApx`,
`easySpeaker`, and `cautious` appear exclusively in approximate conditions;
`knewExact`, `fussyPerson`, `infoAvail` appear exclusively in precise
conditions. `needed`, `realWorld`, and `purpose` appear in both, with
`needed` and `realWorld` strongly concentrated in precise/highPr —
the canonical m_sit cell.

---

## 4. Enrichment Analysis

Table 2 shows how strongly each motive is concentrated in the conditions
associated with each SEM variable group, expressed as an enrichment ratio
(% in strategy-relevant conditions / % in all other conditions).

**Table 2: Motive enrichment by strategy-relevant condition group**

Condition groups:
- **k_p / m_max**: precise/highPr + precise/lowPr
- **m_sit**: precise/highPr + approx/lowPr
- **k_np / m_min**: approx/highPr + approx/lowPr

| Motive | n | % total | k_p/m_max % | ratio | m_sit % | ratio | k_np/m_min % | ratio |
|---|---|---|---|---|---|---|---|---|
| notKnow | 94 | 26.0% | 0.0% | 0.0× | 19.1% | 0.6× | 50.8% | ∞ |
| realWorld | 53 | 14.6% | 23.2% | 3.6× | 20.8% | 2.5× | 6.5% | 0.3× |
| needed | 40 | 11.0% | 14.7% | 1.9× | 17.5% | 3.9× | 7.6% | 0.5× |
| knewExact | 38 | 10.5% | 21.5% | ∞ | 8.2% | 0.6× | 0.0% | 0.0× |
| fussyPerson | 29 | 8.0% | 15.8% | 29.3× | 4.4% | 0.4× | 0.5% | 0.0× |
| infoNotAvail | 29 | 8.0% | 0.0% | 0.0× | 9.3% | 1.4× | 15.7% | ∞ |
| purpose | 23 | 6.4% | 9.6% | 3.0× | 8.7% | 2.2× | 3.2% | 0.3× |
| usualApx | 20 | 5.5% | 0.0% | 0.0× | 7.1% | 1.8× | 10.8% | ∞ |
| infoAvail | 17 | 4.7% | 9.6% | ∞ | 3.8% | 0.7× | 0.0% | 0.0× |
| easySpeaker | 9 | 2.5% | 0.0% | 0.0× | 3.8% | 3.4× | 4.9% | ∞ |
| presentation | 9 | 2.5% | 3.4% | 2.1× | 1.6% | 0.5× | 1.6% | 0.5× |
| emphasis | 8 | 2.2% | 3.4% | 3.1× | 1.1% | 0.3× | 1.1% | 0.3× |
| helpful | 7 | 1.9% | 3.4% | 6.3× | 2.2% | 1.3× | 0.5% | 0.2× |
| cautious | 6 | 1.7% | 0.0% | 0.0× | 1.1% | 0.5× | 3.2% | ∞ |

∞ = motive appears exclusively in this condition group; 0.0× = motive
never appears in this condition group.

---

## 5. Correlation with Social Attribute Ratings

Table 3 shows point-biserial correlations between each motivation category
(binary) and each of the six social attribute ratings (1–7 Likert scale),
expressed as significance levels with direction.

Cell format: direction (+/−) and significance level.
Significance: \*\*\* p<.001, \*\* p<.01, \* p<.05, ns = not significant.

**Table 3: t2 motivation category × social attribute ratings (Experiment 1)**

| Motive | n | % | comp | knowl | w_prep | helpf | likea | pedan |
|---|---|---|---|---|---|---|---|---|
| notKnow | 94 | 26.0% | −\*\*\* | −\*\*\* | −\*\*\* | −\*\*\* | ns | −\*\*\* |
| realWorld | 53 | 14.6% | ns | ns | +\*\*\* | +\* | ns | ns |
| needed | 40 | 11.0% | +\*\* | +\* | +\* | +\* | ns | ns |
| knewExact | 38 | 10.5% | +\*\*\* | +\*\*\* | +\*\* | +\*\* | ns | +\* |
| fussyPerson | 29 | 8.0% | +\* | ns | +\* | +\* | −\* | +\*\*\* |
| infoNotAvail | 29 | 8.0% | ns | ns | ns | ns | ns | ns |
| purpose | 23 | 6.4% | +\* | +\* | +\* | +\* | +\*\*\* | ns |
| usualApx | 20 | 5.5% | ns | ns | −\* | −\* | ns | ns |
| infoAvail | 17 | 4.7% | ns | ns | +\* | ns | ns | +\* |
| easySpeaker | 9 | 2.5% | −\*\* | −\*\* | −\* | −\* | ns | ns |
| presentation | 9 | 2.5% | ns | ns | ns | ns | ns | ns |
| emphasis | 8 | 2.2% | ns | ns | ns | ns | ns | ns |
| helpful | 7 | 1.9% | ns | ns | ns | ns | ns | ns |
| cautious | 6 | 1.7% | ns | ns | ns | ns | ns | ns |

---

## 6. Derived Mapping

Combining the condition profiles (Table 1), enrichment analysis (Table 2),
and social attribute correlations (Table 3), the following mapping emerges.

### Mapping rationale per SEM variable

**k_p (speaker knows exact value):**
Maps to `knewExact` and `infoAvail`. Both motives appear exclusively in
precise conditions (enrichment ratio ∞ for k_p/m_max group; 0.0× for
approx group). `knewExact` directly expresses the k_p construct ("the
speaker knew the exact value"). `infoAvail` ("precise information was
available to the speaker") is conceptually equivalent — availability of
precise information is the necessary condition for k_p. Both show
consistent positive effects on the competence cluster in Table 3.

**k_np (speaker lacks precise knowledge):**
Maps to `notKnow` and `infoNotAvail`. Both appear exclusively in
approximate conditions (∞ enrichment for k_np/m_min group). `notKnow`
directly expresses k_np. `infoNotAvail` ("precise information was not
available") is the objective correlate of the same construct — the
speaker approximates because they could not have been precise.
`notKnow` shows strong negative effects on the competence cluster
and on pedantry (Table 3), consistent with the k_np impact function.
`infoNotAvail` shows no significant effects, likely because it
describes an external circumstance rather than speaker ability,
attenuating the social attribution.

**m_max (always maximally precise):**
Maps to `fussyPerson`. This motive is 29× enriched in precise conditions
and appears almost exclusively in the precise/lowPr cell — the diagnostic
condition for m_max, where precision exceeds contextual demands.
`fussyPerson` ("the speaker is a detail-oriented/pedantic person by nature")
captures the defining feature of m_max: precision as a stable personality
disposition rather than a contextual response. It is the only motive with
a strong pedantry signal (+\*\*\*) and a negative likeability effect (−\*),
consistent with the theoretical characterization of m_max.

**m_sit (situationally appropriate):**
Maps to `needed`, `realWorld`, and `purpose`. All three are enriched in
m_sit-relevant conditions (3.9×, 2.5×, and 2.2× respectively).
`needed` ("the level of precision was required by the situation") is the
most direct expression of m_sit — situational demands explaining the
precision choice. Its highest prevalence in precise/highPr (23.3%) and
notable presence in approx/lowPr (11.8%) matches the m_sit profile
exactly. `realWorld` ("real-world consequences made precision important")
captures the same contextual logic: the stakes of the situation justify
the speaker's choice. Its concentration in precise/highPr (38.9%) is the
highest of any motive in any single cell, consistent with m_sit's
defining role in high-precision contexts. `purpose` ("the speaker was
deliberate/purposeful") reflects intentional calibration to context,
which is the cognitive counterpart of situational appropriateness.

**m_min (always minimally precise):**
Maps to `usualApx` and `easySpeaker`. Both appear exclusively in
approximate conditions (∞ enrichment). `usualApx` ("the speaker usually
uses approximate expressions") describes approximation as a stable
behavioral habit — the lay counterpart of the formal construct of a
speaker who always chooses the approximate form regardless of context.
`easySpeaker` ("it was easier for the speaker") describes effort
minimization — the motivational substrate of m_min. Both motives show
negative effects on the competence-adjacent attributes (well_prepared,
helpful) in Table 3, consistent with the m_min impact function.

### Summary mapping table

| SEM variable | Definition | t2 proxies | Enrichment | Justification |
|---|---|---|---|---|
| **k_p** | Knows exact value | `knewExact`, `infoAvail` | ∞ in precise | Direct expressions of having precise knowledge |
| **k_np** | Lacks precise knowledge | `notKnow`, `infoNotAvail` | ∞ in approx | Direct expressions of lacking precise knowledge |
| **m_max** | Always maximally precise | `fussyPerson` | 29× in precise | Precision as stable personality trait; diagnostic in prc/lowPr |
| **m_sit** | Situationally appropriate | `needed`, `realWorld`, `purpose` | 2.2–3.9× in m_sit | Situational demands and deliberate contextual calibration |
| **m_min** | Always approximate | `usualApx`, `easySpeaker` | ∞ in approx | Habit-based and effort-based approximation |

### Motives not mapped

Four motives with n ≥ 5 are not assigned to any SEM variable:

- **`presentation`** (n=9): Distributed across all conditions; no
  systematic enrichment pattern. Likely reflects scenario-specific
  reasoning unrelated to the knowledge/motivation constructs.
- **`emphasis`** (n=8): Concentrated in precise/lowPr (6.9%) but
  n too small and conceptually ambiguous (emphasis as rhetorical
  device does not map cleanly to any SEM strategy).
- **`helpful`** (n=7): Concentrated in precise/highPr (4.4%) but
  n too small for reliable estimation.
- **`cautious`** (n=6): Appears exclusively in approximate conditions,
  suggesting proximity to k_np (epistemic hedging). However, caution
  as a communicative stance is conceptually distinct from lacking
  knowledge, and the small n does not support inclusion.

---

## 7. Notes on the Derivation Procedure

The impact function values (φ_K, φ_M ∈ {−1, 0, +1}) are derived from
Table 3 using the following procedure:

1. For each SEM variable, pool the t-test results across all assigned
   proxies using a majority vote (sum > 0 → +1; sum < 0 → −1; sum = 0 → 0)
2. Map social attributes directly to SEM attributes:
   - a_comp ← `competent` (direct, no aggregation)
   - a_like ← `likeable` (direct, no aggregation)
   - a_ped  ← `pedantic` (direct, no aggregation)
3. Apply significance threshold p < .01

The direct attribute mapping (step 2) was chosen over aggregation across
related attributes (knowledgeable, well_prepared, helpful) to avoid
importing theoretical assumptions about which attributes constitute
a unified construct. The SEM models three attributes; the derivation
tests exactly those three.

The threshold p < .01 was chosen as the primary derivation criterion.
Lookup tables for all thresholds (p < .001, .01, .05, .10) are stored in
`data/cs1_experiment/processed/impact_tables_cs1.json` for use in
threshold-sensitivity robustness checks.

---

*Generated: April 2026. Data: experiment1.csv (n=362).*
*Code: `sem/impact_derivation.py`, `scripts/derive_impact_tables.py`.*
