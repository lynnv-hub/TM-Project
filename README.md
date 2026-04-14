## How different news outlets frame elections globally

## Abstract
A max 150-word description of the project question or idea, goals, dataset used. What story you would like to tell and why? What's the motivation behind your project?

## Research questions
A list of research questions you would like to address during the project. 

How do different major US news outlets evaluate selected elections globally? 
How consistent are outlet specific patterns when looking at different countries? 
How do the profiles of media outlets match their expected political alignment?
How do GDELT sentiment scores compare to indepentently calculated ones?

## Dataset
List the dataset(s) you want to use, and some ideas on how do you expect to get, manage, process and enrich it/them. Show you've read the docs and are familiar with some examples, and you've a clear idea on what to expect. Discuss data size and format if relevant.

We use the GDELT data set. It includes major news outlets and it is possible to filter out for reactions to certain elections (e.g. pre election week + post election week) as it has thematic encoding. This data already has a sentimental score, but this could also be done ourselves and then compared to the GDELT scores.
In practice articles about a specific election can be filtered out by using a time window, actors and key words (e.g. elections, president, parliament).
We would use the US presidential elections 2020 and 2024, the German elections 2025, Polish elections 2019 and 2023, French parliamentary elections 2024. All those elections have rather clear outcomes and involved debates about populism and polarization.
The newsoutlets are main US outlets with a known political leaning: NYT (more left), foxnews(right), CNN (more left) and AP (neutral). 
The data can be extracted via BigQuery as csv files and then analyzed. 
Possible problems could be different amount of articles of the outlets per elections (normalization as possible solution). Additionally it has to be checked if there are no articles multiple times and if most articles about the topic are actually included.  


## A tentative list of milestones for the project
Add here a sketch of your planning for the coming weeks. Please mention who does what.


### Data Collection and Cleaning
The first important step is to extract the correct data for the different elections via BigQuery. The datasets then have to be cleaned up (missing values, duplicates etc.).  
### Analysation and Profiling
We will use the already existing sentiment score by GDELT and additionally build our own sentiment analysis (BERT model). Then we will compute the scores per outlet and elections and compare the two different results, to build profiles of the different outlets.   
### Analyse results
An important step is to analyse and find reasons for diverging scores. Additionaly it has to be analysed if the profiles of outlets match the expected political leaning.

## Documentation
This can be added as the project unfolds. You should describe, in particular, what your repo contains and how to reproduce your results.
