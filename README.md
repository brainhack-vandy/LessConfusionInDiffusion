# LessConfusionInDiffusion
The goal of “Less confusion in diffusion” is to develop a VLM-based tool that identifies image issues (eddy current distortions, significant motion, poor resolution, insufficient number of b-vector directions, missing slices, top of brain not in FOV) in diffusion weighted images (DWI) and recommends solutions. Working with DWI can be tricky (especially if you don’t have a diffusion imaging expert on call!) and there are a wide range of distortions, artifacts, and noise that need to be corrected. The proposed tool is designed for people getting started in DWI. This project is the first step toward a tool that gives advice based on data acquisition and quality. 

Overall, the proposed workflow is: A user pastes/loads a png screenshot of their B0-image into our user interface. The VLM vectorizes this image, and provides a completion to the user. This completion will include: the problems detected in the image ("diagnosis") and how to solve those problems with toolboxes/code ("cure"). 

## Table of Contents
- [Onboarding Documentation](#onboarding)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Access](#data-access)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Goals for Brainhack](#goals-for-brainhack-global)


## Onborading Documentation
* Add your name to CONTRIBUTING.md by committing to the repo
* Get a hugging face account https://huggingface.co/ 
* Basics of Diffusion weighted MRI modality: https://radiopaedia.org/articles/diffusion-weighted-imaging-2?lang=us
* Common issues and solutions with these images: https://pubmed.ncbi.nlm.nih.gov/33533094/ 
* Downloading a model from hugging face: https://huggingface.co/docs/hub/models-downloading 

## Prerequisites

List all system requirements and dependencies needed before installing the project:

- Python 3.8+
- PyTorch

## Installation

Step-by-step installation instructions:

```bash
# Clone the repository
git clone https://github.com/username/project-name.git
```

## Data Access
I curated a few different options for us to play with. We have the samples in three forms: original DWI, .npy slices, and .png screenshots of those slices. Each sample/subject corresponds to a data entry in the table 'labels.csv'. This file contains the lookup ID that matches the .png or .npy file name, the problems with the image, and the solutions to those problems. 

Download from our Box folder:
```
URL: [https://vanderbilt.box.com/s/v50gfkqzirr2pp05dgf9rs45sum3lq8h]
Download these example subjects to /data folder to get started:

sub-001
sub-002
sub-003
```

## Project Structure
The project will be structured as follows:
```
project/
├── src/              # Source code (if any)
├── data/             # Data files (put your downloaded images and table here)
└── scripts/          # Utility scripts
```

## Contributing

Instructions for potential contributors:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Follow coding standards

## License

This project is licensed under [license type]. The data used in this project is covered under the original data sharing agreement of [repository name].

## Goals for Brainhack Global
1. Get a hugging face account
2. Dataset curation: There is a directory of diffusion image slices. Have an expert create their associated text labels (what should the completion look like?). Note: This step was completed 1/23 by Nancy Newlin.
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


