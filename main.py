import argparse
import logging
import pandas as pd
from core.pipeline import run_pipeline

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="AI Recruiter Pipeline")
    parser.add_argument(
        "--resumes",
        type=str,
        default="./data/resume_data.csv",
        help="Path to the resume CSV file."
    )
    parser.add_argument(
        "--jd",
        type=str,
        default="./jd/sample_jd_01.txt",
        help="Path to the job description text file."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of top candidates to return."
    )
    args = parser.parse_args()

    logging.info("Running pipeline with:")
    logging.info(f"  Resumes: {args.resumes}")
    logging.info(f"  Job Description: {args.jd}")
    logging.info(f"  Top N: {args.top_n}")

    top_candidates = run_pipeline(
        jd_text=None,
        top_n=args.top_n,
        resume_csv_path=args.resumes,
        jd_file_path=args.jd
    )

    logging.info("\n--- Top Candidates ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    print(top_candidates)

if __name__ == "__main__":
    main()
