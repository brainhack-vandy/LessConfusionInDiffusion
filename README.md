# LessConfusionInDiffusion
The goal of “Less confusion in diffusion” is to develop a VLM (vison language model)-based tool that identifies image issues in diffusion weighted images (DWI) and recommends solutions. 

Project description:
## What are you doing, for whom, and why?
The goal of “Less confusion in diffusion” is to develop a LLM-based tool that identifies image issues (eddy current distortions, significant motion, poor resolution, insufficient number of b-vector directions, missing slices, top of brain not in FOV) in diffusion weighted images (DWI) and recommends solutions. Working with DWI can be tricky (especially if you don’t have a diffusion imaging expert on call!) and there are a wide range of distortions, artifacts, and noise that need to be corrected. The proposed tool is designed for people getting started in DWI. This project is the first step toward a tool that gives advice based on data acquisition and quality. 

## What makes your project special and exciting?
Contributors to this project will gain experience in 1) using HuggingFace, 2) fine-tuning LLMs, 3) debugging LLMs, and 4) creating a user interface. 

## Where to find key resources?
Diffusion imaging: https://radiopaedia.org/articles/diffusion-weighted-imaging-2?lang=us
Common issues and solutions: https://pubmed.ncbi.nlm.nih.gov/33533094/ 

## Goals for Brainhack Global
1. Get a hugging face account
2. Dataset curation: There is a directory of diffusion image slices. Have an expert create their associated text labels (what should the completion look like?).
3. Data visualization: How many samples are there of each type? What do the responses look like? This is the time to look at the data and understand what each of the expected cases are.  
4. Model selection: what hugging face models are appropriate for this image to text task?
Test run model: Run model as-is on a test dataset (hopefully provided by hugging face project). At this stage, we need to make sure the model will act as expected (inputs, outputs). 
5. Dataloading: Set up dataloader to get slices and labels from directory and properly interface with current model. 
6. Test: Try model on a few samples and observe the behaviour!
7. Documentation: Input/Output description, open problems, how to use

8. Advanced: Joint embedding of text and image inputs (ex. “here is my image and I have a b-vector file with 100 directions and b-values ranging 0-2000”)
9. Extra 1: Improve on response quality (more conversational, more information provided). 
10. Extra 2: User interface: Set up a local server that can take an image slice as input, and provide a response. 
11. Extra 3: Upload model to Hugging Face!

## Skills
Must haves:    
* Proficient in python 
* Proficient in pytorch
* Basic knowledge of medical images (they have headers and metadata)
* Working knowledge of machine learning principles (training, inference, data loaders)
* Able to push/pull from github
* Access to GPUs 
* Prefered: Experience with diffusion weighted MRI
  
## Onboarding documentation
* Add your name to CONTRIBUTING.md by committing to the repo
* Get a hugging face account https://huggingface.co/ 
* Basics of Diffusion weighted MRI modality: https://radiopaedia.org/articles/diffusion-weighted-imaging-2?lang=us
* Common issues and solutions with these images: https://pubmed.ncbi.nlm.nih.gov/33533094/ 
* Downloading a model from hugging face: https://huggingface.co/docs/hub/models-downloading 

## What will participants learn?
* How to get a model from hugging face
* Experience fine-tuning a language model 
* Experience working with medical images 
* Experience with image-to-text tasks

## Data to use
Data is currently in NIFTI format here: https://vanderbilt.box.com/s/v50gfkqzirr2pp05dgf9rs45sum3lq8h 

## Number of collaborators
4-6

## Credit to collaborators
Name listed in ReadMe (make sure you added your name to contributions) and co-authorship if there is any resulting publication or conference proceeding.


