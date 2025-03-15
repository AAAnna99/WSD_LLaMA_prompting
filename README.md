======================================================================================
	TECNICHE DI PROMPTING DI LARGE LANGUAGE MODEL PER IL WORD SENSE DISAMBIGUATION
======================================================================================

This package contains datasets, scripts, and results from my Master's Thesis in Computer Science from the University of Bologna.

The repository is organized as follows:
- The Dataset folder contains all the datasets used for this project:

	WSD_Unified_Evaluation_Datasets folder contains five Word Sense Disambiguation evaluation datasets.

	SemCor folder contains a sense-annotated corpora used for extracting example sentences for words with the same meaning as the words in datasets.

	Each dataset consists of a data file [dataset].data.xml and a gold file [dataset].gold.key.txt. 
	All senses are annotated with WordNet 3.0.

- The Scripts folder contains the code developed for LLaMA2-7b Chat and LLaMA3.1-8b Instruct models.
	The LLaMA2 folder contains code for ZeroShot Prompting developed with different prompt templates.
	The LLaMA3 folder contains code for different prompting techniques tested for the entire Unified Evaluation Framework.
	
- scorer.java for calculating ratings from the [gold-file] [data-file] comparison.

- The Results folder contains the predictions generated by the model, divided by dataset and prompting technique.



### Python virtual environment to manage dependencies: 
To create the virtual environment:
`python -m venv venv_name`


To activate the virtual environment:

On Windows:
`venv_name\Scripts\activate`

On macOS and Linux:
`source venv_name/bin/activate`

### Install packages
`pip install transformers accelerate bitsandbytes nltk lxml huggingface-hub`


###  Login to Hugging Face
Access to LLaMA models is possible upon authorization through the Hugging Face platform, using an authentication token issued upon request for use.


### SCORER
To use the scorer compile:
`$ javac Scorer.java`

To evaluate the system: 
`java Scorer [gold-standard] [system-output]`

Example of usage:
`$ java Scorer semeval2013/semeval2013.gold.key.txt semeval2013/output.key`


=======================================================================================================================
REFERENCE PAPER
Alessandro Raganato, José Camacho-Collados and Roberto Navigli. 
Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison
In Proceedings of European Chapter of the Association for Computational Linguistics (EACL), 
Valencia, Spain, April 3-7, 2017. 
