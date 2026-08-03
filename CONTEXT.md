# AI Recruiter Pipeline

Candidate scoring and ranking for recruiting: a job description and a candidate pool go
in, a ranked shortlist comes out. This glossary pins the vocabulary the scoring pipeline
and its evaluation are described in.

## Language

### Scoring

**Component**:
A single weighted scoring signal in the fusion sum (e.g. title, skill, similarity,
industry), each on a 0–1 scale, combined by `calculate_total_score`.
_Avoid_: feature, factor, dimension

**Retriever** (Stage 1):
The full Component pipeline scoring every candidate to produce the initial ranking, from
which the Head is selected.
_Avoid_: first pass, recall stage

**Reranker** (Stage 2):
The cross-encoder that re-scores only the Head and reorders it. Deliberately a stage, not
a Component.
_Avoid_: re-scorer, second-pass component

**Head** (top-K):
The top-K candidates by Stage-1 score (K = 50) that the Reranker is allowed to reorder;
its membership is frozen once selected.
_Avoid_: top slice, candidate pool

**Shortlist**:
The final top-30 deliverable handed to the recruiter — a subset of the reordered Head.
_Avoid_: results, output, final list

### Evaluation

**Judge panel**:
The two independent frontier LLMs (Claude Opus 4.8 + GPT-5.5) that blind-score candidates
against a rubric derived from the real posting.
_Avoid_: the LLM, the grader, the model

**Judge grade**:
A frozen, per-candidate 0–100 score produced by the Judge panel; a silver
(LLM-generated, not human) evaluation label.
_Avoid_: gold label, fit score, relevance label

**Section grade**:
A Judge's bounded score for one named rubric section. The Judge grade is the
additive total of all Section grades.
_Avoid_: component score, feature score

**Judged cohort**:
The frozen set of candidates that receives Judge grades in one adjudication run.
_Avoid_: candidate panel, judge pool

**Silver Judge-grade anchor**:
An evaluation case whose candidate labels are Judge grades rather than human
assessments. It can support regression checks but is not gold ground truth.
_Avoid_: gold anchor, gold case
