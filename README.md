# textual_analysis_patents
**To visualize attributes significance:**
1. download the folder sig-visualize
2. from this directory type in ```python -m SimpleHTTPServer ``` in your terminal
3. navigate to http://localhost:8000/ in your browser
4. to visualize a different sample, modify "NUM_CATEGORY" and "json_fname" accordingly, around line 30 in index.html, details can be found in comments 

Note: 
1. sig-visualize contains a pre-trained model with 4 classifiers, and there is no need to train a network to run the visualization
2. to load data successfully, you may need to change the loading path in sig-visualize/vis_prepareData.py in the "prepare_data" section
3. vis_prepareData.py outputs a json file containing texts, label, prediction with probability for each class, and attributes significance scores for one selected sample. If you would like to test a different sample, the easiest way is to modify in the "interpret" section, for example, change index for y_test and y_pred
