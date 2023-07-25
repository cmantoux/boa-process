# Code for the application to an urban tree bases data set

- All the codes can be run in Python 3.x
- The datasets can be found in the folders "donnees_occurence_2009_2018" and "donnees_occurence_2014_2018". Each csv file corresponds to one of the pairs listed in Table A.2. 
- The files ls.txt and ls_2014_2018.txt contain the list of the files in these two folders.

1. **Estimation - Species monitored in 2009-2018**:
Run the code "analyse_pas_2009_2018_v4.py". The result of the estimation will be recorded in the folder "output_analysis_pa_2009_2018"

2. **Estimation - Species monitored in 2014-2018**:
Run the code "analyse_pas_2014_2018_v4.py". The result of the estimation will be recorded in the folder "output_analysis_pa_2014_2018"

3. **Summary of estimation results**:
Run the code "creation_fusion_csvs.py". Produces a csv file with the estimater GER and LER for each pair species/portion of street. 

4. **Computation of the SMDs**:
Run the code "comparaison_distr_p_ext_v2.py".
