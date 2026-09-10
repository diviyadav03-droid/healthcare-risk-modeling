# U.S. County Health Outcomes: Drivers, Access, and Telehealth Market Strategy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diviyadav03-droid/healthcare-risk-modeling/blob/main/notebooks/county_health_outcomes_analysis.ipynb)

A data analytics case study examining the behavioral, socioeconomic, and healthcare-access factors associated with
county-level health outcomes across the United States, developed as a market-entry feasibility study for
GetBetterHealth.com, a telehealth provider evaluating where to prioritize geographic expansion.

## Project Overview

This project analyzes **County Health Rankings & Roadmaps 2024** data covering **3,143 U.S. counties** and **88
variables**, spanning health outcomes, health behaviors, clinical care, and social, economic, and environmental
factors. The three primary outcomes studied are:

- Life expectancy
- Poor physical health days
- Poor mental health days

## Business Questions

1. Which behavioral, socioeconomic, and access-related factors are most strongly associated with poor health
   outcomes at the county level?
2. Is healthcare access (primary care physician supply) associated with outcomes independently of income?
3. Can counties be segmented into distinct health-risk archetypes to prioritize a market-entry strategy?
4. Which counties represent the highest-priority opportunity for a telehealth expansion strategy?

## Methods

- Exploratory data analysis across the three outcome variables
- Pearson correlation analysis
- Multivariate linear regression
- Random forest regression with 5-fold cross-validation and feature-importance analysis
- Healthcare-access analysis with income-controlled comparisons
- K-means clustering for county segmentation
- Rule-based telehealth opportunity screening

## Key Findings

- **Behavioral factors matter most for physical health burden.** Adult smoking and physical inactivity show the
  highest predictive importance for poor physical health days in the random forest model.
- **Strong model performance.** The random forest model for physical health burden achieved approximately **0.81
  cross-validated R²** (5-fold cross-validation).
- **Four distinct county archetypes** were identified via K-means clustering, each with a different profile of
  health outcomes, behaviors, and socioeconomic conditions.
- **The highest-burden archetype** averaged approximately **72.1 years** of life expectancy and was heavily
  concentrated in Southern states.
- **Healthcare access shows an association beyond income differences.** Primary care physician supply was compared
  across income levels, and access-related differences in outcomes persisted across income groups.
- **Telehealth opportunity screening** identified **477 counties** as potential priority markets — 265 flagged for
  physical-health-driven need and 212 for mental-health-driven need — where healthcare-access gaps and health
  burden overlap. Approximately **79%** of high-burden counties are concentrated in Southern states.

> These findings represent statistical associations from observational, cross-sectional county-level data and do
> not establish causal relationships.

## Strategic Recommendations

A rule-based opportunity screen identified 477 counties as potential telehealth priority markets where
healthcare-access gaps and health burdens overlap, with a strong concentration in Southern states. These counties
represent a reasonable starting point for prioritizing a telehealth expansion strategy, combining elevated health
burden with limited existing access to in-person care. As with all findings in this project, this prioritization
reflects statistical association, not a causal guarantee of outcomes, and should inform — not replace — further
business and clinical due diligence.

## Data

**Source:** [County Health Rankings & Roadmaps, 2024](https://www.countyhealthrankings.org/health-data), a program
of the University of Wisconsin Population Health Institute in collaboration with the Robert Wood Johnson
Foundation.

**The raw dataset is not included in this repository.** To run the notebook, obtain `analytic_data2024_case.csv`
directly from the official source above and place it at `data/analytic_data2024_case.csv`. Redistribution terms
for the raw file have not been independently verified, so it is left to users to obtain it themselves.

## Repository Structure

```text
healthcare-risk-modeling/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── county_health_outcomes_analysis.ipynb
│
├── data/
│   └── analytic_data2024_case.csv        # not included — see "Data" above
│
├── src/
│   └── GBH_Analysis.py
│
├── presentation/
│   └── Strategic_Health_Insights_County_Health_Outcomes.pptx
│
└── visualizations/                       # populated when the notebook is run
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Running the Notebook

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the 2024 County Health Rankings dataset from the official source.
4. Place `analytic_data2024_case.csv` in `data/`.
5. Open `notebooks/county_health_outcomes_analysis.ipynb`.
6. Run the notebook from beginning to end.
7. Generated visualizations are saved to `visualizations/`.

## Transferable Risk Analytics Skills

Although this project focuses on healthcare data, the analytical framework demonstrates transferable risk-analytics
capabilities, including predictive modeling, risk segmentation, feature-importance analysis, clustering,
comparative analysis, and data-driven prioritization — skills directly applicable to risk analytics, model risk,
quantitative analytics, and business analytics roles.

## Limitations

- **Observational, cross-sectional data.** Results reflect statistical associations, not causal relationships.
- **County-level analysis.** Findings describe county-level patterns and should not be interpreted as predictions
  about individual people.
- **Associations, not causation.** No claim in this project should be read as establishing that any single factor
  causes a given health outcome.
- **Data and variable dependency.** Results are dependent on the variables available in the County Health Rankings
  dataset and may not capture all relevant drivers of health outcomes.

## Author

**Divya Yadav** — [LinkedIn](https://linkedin.com/in/divya-yadav-ncsu) · [GitHub](https://github.com/diviyadav03-droid)
