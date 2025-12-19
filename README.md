# MBO_shared
This repository contains code for our manuscript under review entitled "Episodic memory formation is theta-rhythmic". 

Steps:
1) First run MBO_data_cleaning_helper.Rmd, which provides helper functions for data cleaning
2) Then run MBO_data_cleaning.Rmd, which: 
  - Organizes data, compute memory accuracy, and save cleaned data
  - Applies glm's at the subject level to provide additional data cleaning
  - Save timeseries data for AR1_control analysis
3) Then run python script in Code/AR1_control directory, which:
  - Runs AR1_control analysis, and save the output to Data/AR1_control
4) Next, run MBO_data_analysis_helper.Rmd, which loads helper functions for all main analyses
5) Finally, run MBO_data_analysis.Rmd, which:
  - Runs all the analyses reported in the paper, and saves figures and tables in Results directory
