from core.pipeline import run_pipeline


def test_llm_generated_jd_pipeline_ranks_candidate_top_3():
    """
    Tests the evaluation pipeline by checking if a candidate, for whom a synthetic
    job description was generated, ranks among the top 3 matches.

    This test case assesses the "reverse" of the primary pipeline:
    1. A synthetic job description is created by an LLM based on a specific candidate's profile (C477 in this case).
    2. The pipeline is run with this synthetic job description.
    3. The test asserts that the original candidate (whose profile was used to generate the JD)
       is ranked within the top 3 results returned by the pipeline.

    This helps validate that the matching and scoring logic is effective and can
    identify a highly relevant candidate when the job description is tailored to them.
    """
    # pick a known candidate
    candidate_id = "C477"

    # synthetic jd generated for above candidate using llm
    synthetic_jd = """# Financial AI Analyst - FinTech & Data Analytics

## Company Overview
We are a leading financial technology company specializing in AI-powered solutions for accounting, financial analysis, and business intelligence. Our mission is to transform traditional financial processes through innovative machine learning applications and data-driven insights.

## Position Summary
We are seeking a Financial AI Analyst to bridge the gap between financial expertise and artificial intelligence. This role involves developing AI models for financial data analysis, automating accounting processes, and creating intelligent solutions for financial reporting and analysis. The ideal candidate will combine strong financial acumen with emerging AI capabilities to revolutionize how financial data is processed and analyzed.

## Key Responsibilities
- **Financial AI Model Development**: Design and implement machine learning models for financial forecasting, fraud detection, and risk assessment
- **Data Analysis & Financial Insights**: Analyze large financial datasets to identify patterns, trends, and anomalies using advanced analytics
- **Automated Financial Processes**: Develop AI solutions to streamline accounts payable, accounts receivable, and general ledger processes
- **Financial Reporting Automation**: Create intelligent systems for generating financial reports, balance sheets, and trial balances
- **Cross-Functional Collaboration**: Work with finance, accounting, and technology teams to implement AI-driven financial solutions
- **Model Training & Deployment**: Train machine learning models on financial data and deploy them in production environments
- **Documentation & Compliance**: Maintain comprehensive documentation of AI models and ensure compliance with financial regulations
- **Business Intelligence**: Develop predictive models for sales analysis, cash flow forecasting, and financial planning

## Required Qualifications
- **Education**: Master's degree in Commerce (M.Com) or Bachelor's degree in Commerce (B.Com) with strong academic background
- **Financial Experience**: 3+ years of experience in accounting, bookkeeping, or financial analysis
- **Technical Skills**: Proficiency in data entry, financial software, and analytical processes
- **Software Proficiency**: Advanced Excel skills, ERP systems, QuickBooks, MYOB, and MS Office applications
- **Languages**: Fluency in English and Hindi preferred for client communication

## Technical Skills
- **AI/ML Frameworks**: TensorFlow, PyTorch, Scikit-learn
- **Programming**: Python, R, or Java for financial modeling and analysis
- **Financial Software**: QuickBooks, MYOB, ERP systems, and accounting software
- **Data Analysis**: Excel, PowerPoint, Outlook, and MIS reporting tools
- **Financial Processes**: General ledger, accounts payable/receivable, payroll, bank reconciliation, invoicing

## Preferred Experience
- Experience with financial data analysis and sales analysis
- Background in cash management and fixed assets accounting
- Knowledge of accrual accounting and financial statement preparation
- Experience with purchasing processes and vendor management
- Supervisory or management experience in financial operations
- Understanding of legal and compliance requirements in finance

## Key Competencies
- **Analytical Skills**: Strong ability to analyze complex financial data and identify meaningful patterns
- **Problem-Solving**: Innovative approach to automating traditional financial processes
- **Communication**: Excellent written and verbal communication skills for cross-functional collaboration
- **Team Collaboration**: Proven ability to work effectively with diverse teams
- **Detail-Oriented**: High accuracy in financial data processing and model development
- **Adaptability**: Willingness to learn new AI technologies and apply them to financial use cases

## What We Offer
- Opportunity to be at the forefront of AI innovation in financial services
- Comprehensive training in machine learning and AI applications
- Career advancement opportunities in the rapidly growing FinTech sector
- Competitive compensation package with performance-based incentives
- Collaborative work environment with emphasis on continuous learning
- Exposure to cutting-edge AI technologies and financial systems

## Work Environment
This is a hybrid role combining traditional financial analysis with innovative AI development. The position requires both analytical thinking for financial processes and technical aptitude for AI model development. Remote work options available with flexible scheduling.

---

*We are committed to creating an inclusive workplace that values diverse perspectives and backgrounds in both finance and technology.*
"""

    df_with_scores = run_pipeline(jd_text=synthetic_jd)

    top_3_candidates = df_with_scores.sort_values("total_score", ascending=False).head(3)

    assert candidate_id in top_3_candidates['candidate_id'].values, \
        f"Candidate {candidate_id} not found in top 3 for generated JD"
