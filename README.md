# End-to-End Imputation and Prediction via Adaptive Frequency Learning In Irregular Time-series data
DyFAIP (Dynamic Frequency-Aware Network for Imputation and Prediction), a novel deep dynamic memory neural network framework that jointly optimizes missing data imputation and downstream prediction in an end-to-end manner.


1*) How to run the script on Beijing Air Quality Multi-Site dataset:

    -Unzipp the dataset from the data folder 
    -output_path : File to save subject's data

python subjects_features_extractor.py -table "OUTPUTEVENTS,LABEVENTS,CHARTEVENTS,INPUTEVENTS_CV,INPUTEVENTS_MV" -output_path "NEW_FILE"

2*) Extract Subject's events data from files(table):
   
      -output_path : Filename where subject's data have been saved!
      -subject_path : File to save subject's events data!
      -inputs: Save inputevents data to file INPUTEVENTS!
    
python subject_data.py -output_path "NEW_FILE" -subject_path "subjects" -inputs "INPUTEVENTS" -charts "CHARTEVENTS" -labs "LABEVENTS" -outputs "outputevents"

 3*) Generate Subject's timeseries data
 
      -path_dir:  Main directory where all the data have been saved!
    
  python subject_timeseries.py -path_dir "NEW_FILE" -subject_path "icustays" -output_path "subjects_timeseries" -cohort_path "EPISODES_SUBJECTS" -hrs_data_min 0 -hrs_data_max 47

Overall Script's time execution : 7:41:26

Link to Subjects medical notes
https://drive.google.com/drive/folders/1vO8JToP8HfnVSl6TpfiwAMdvdow7b9bz?usp=drive_link
