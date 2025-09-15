# Spotify Music Classification Project

<h3>Capstone Project for CSCI-UA 473 Fundamentals of Machine Learning</h3>

This project uses data from 50,000 randomly selected songs to predict the genre each song belongs in using classification methods.

<h3><b>Data Overview</b></h3>
<p>This project contains one dataset: <b>musicData.csv</b>. It contains the following columns for the 50,000 songs:</p>
<ol>
  <li>Unique Spotify ID of each song</li>
  <li>Artist Name</li>
  <li>Song Name</li>
  <li>Popularity of the Music (Percentage of users that know about it, 0-99%)</li>
  <li>Acousticness Level (0-1)</li>
  <li>Danceability Level (0-1)</li>
  <li>Duration of the Music (milliseconds)</li>
  <li>Energy Level (0-1)</li>
  <li>Instrumentality Level (0-1)</li>
  <li>Key of the Song (musical notation)</li>
  <li>Liveness Level (0-1)</li>
  <li>Loudness (in dB)</li>
  <li>Mode of the Song</li>
  <li>Speechiness Level (0-1)</li>
  <li>Tempo (in beats)</li>
  <li>Obtained Date</li>
  <li>Valence Level (0-1)</li>
  <li>Music Genre</li>
</ol>

<h3>Key Steps</h3>
<ol>
  <li><b>Data Preprocessing:</b></li>
  <ul>
    <li>Loaded and cleaned the dataset by filtering out irrelevant data, handling missing data, and managing outliers.</li>
    <li>Encoded the Music Genre variable by extracting 500 rows for each genre to determine how many and what genres there are.</li>
    <li>Ran stratified train/test splits such that 500 rows per genre from the encode are in the test set and the remaining 4500 rows are in the training set, excluding information about the artist and track.</li>
    <li>Set PCA to 0.95 for enough variance without serious overfitting and binarized the output test variable for multi-class classification.</li>
  </ul>
  <li><b>Building and Running 3 Classification Models:</b></li>
  <ul>
    <li>Ran 3 classification models: Random Forest, Decision Trees, and AdaBoost; to evaluate and compare each model's performance on the dataset.</li>
    <li>Used micro-average AUC scores of each model to treat all variables equally.</li>
  </ul>
  <li><b>Data Visualization:</b></li>
  <ul>
    <li>Generated an AUROC plot of each model's micro-average AUROC curves to compare the performances visually.</li>
    <li>Used Dimensionality Reduction Clustering to illustrate the distribution of each genre.</li>
    <li>Created a Confusion Matrix for the best model to show the number of correct and incorrect predictions.</li>
  </ul>
</ol>

<h3>Tools and Modules Used</h3>
<ul>
  <li><b>Python:</b> Main Programming Language</li>
  <li><b>Spyder:</b> Software for Code Development and Analysis</li>
  <li><b>Jupyter Lab:</b> Software for Code Development as a Notebook Version</li>
  <li><b>Microsoft Word:</b> Write the Report Highlighting Findings and Insights</li>
  <li><b>Pandas:</b> Data Manipulation and Preprocessing</li>
  <li><b>NumPy:</b> Numerical Operations</li>
  <li><b>Random:</b> Seed the Random Number Generator</li>
  <li><b>Matplotlib:</b> Data Visualization</li>
  <li><b>Scikit-learn:</b> Machine Learning Models</li>
</ul>

<h3>Key Findings</h3>
<ul>
  <li>The Micro-Average AUC scores for each model are: Random Forest – 0.897, Decision Tree – 0.658, AdaBoost – 0.824. The predictive performance for the Random Forest and AdaBoost models are strong while Decision Tree is moderate. But the Random Forest Model has a better predictive performance.</li>
  <li>For clustering, most genres form tight clusters between -2.5 and 2.5 along PC2 with a few outliers. Along PC1, Classical Music forms a tight cluster that extends to -6 while other genres have narrower clusters between -2 and 2. This clustering shows how PCA captures distinct audio feature differences like tempo and energy, but the success depends on the ability for non-linear dimensionality reduction models to figure out genres that overlap in 2D space.</li>
  <li>In the Random Forest Confusion Matrix, the model correctly classifies 2541 of 5000 samples. While just over half of the test sample is predicted correctly, the micro-averaged AUC score is still high. AUC measures the model's ability to rank correct genres higher than incorrect ones, not only exact matches. In a multi-class problem, a model can perform well in terms of the AUC score, even if its top-1 prediction is occasionally wrong, so Random Forest still captures the structure of data effectively and remains a strong predictive model.</li>
</ul>

<h3>Project Deliverables</h3>
<p>This project contains three deliverable files.</p>
<ol>
  <li><b>CapstoneProjectReport.pdf:</b> The project report that contains the full answers to the questions including visuals and results along with an introductory paragraph about data preprocessing.</li>
  <li><b>Capstone.py:</b> The Python code file that produced the data analysis and visuals. Done through Spyder.</li>
  <li><b>Capstone.ipynb:</b> The Jupyter Notebook version of the above file that also contains the values and visuals.</li>
</ol>

<h3>Important Notes</h3>
<ul>
  <li><b>RNG Seeding:</b> To ensure reproducibility, the code begins by seeding the random number generator with a unique number. This ensures that the train/test splits and classification models are keyed consistently to this number.</li>
  <li><b>Row Count and Missing Data:</b> There were a total of 50,005 rows in the CSV file, 5 of which have NaNs for missing data while the remaining 50,000 rows had "?" for any missing values. These were turned to NaNs, but not filtered out. Instead, I filled them in as either the median value of the column for numerical variables or the most frequent categorical input for categorical variables after grouping them by genre.</li>
</ul>

<h3>Contact</h3>
For any inquires or suggestions, don't hesitate to reach out to <a href="mailto:andrewyan32@gmail.com">Andrew Yan</a>.
