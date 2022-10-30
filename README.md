# Semantic Segmentation of Satellite Imagery

## Group 8: Stephen Ebrahim and Ebram Mekhail

## CS 301 - 103

This branch contains Milestone 1 of the project.

## Milestone 1: Environment Preparation

1. Initially, we created a repository for the project on Github.
2. Secondly, we made sure that Python was in a relatively new version, in ourcase since we had python using Homebrew, we updated it through that: `brew upgrade python3`
3. Then we set up poetry by installing it through the following command: `curl -sSL https://install.python-poetry.org | python3 -`
4. To ensure that poetry was available gloably, we also had to include its path the in the bash configuration file. We included this in ~/.zshrc `export PATH="/Users/stephen/.local/bin:$PATH"`
5. Then, we added the necessary dependencies using `poetry add [name of dependency]`. We did this for python, tensorflow, pandas, numpy, matplotlib, and, **NNI**. Furthermore, to ensure that everything was up to date using poetry, we executed the following command: `poetry update package`
6. Now, we test that everything was working properly by setting up the sample project of NNI. First, we open the poetry's shell by running `poetry shell`. Now that we are in the environment of poetry, we can then use installed packages/dependencies.
7. We install the NNI hello project by executing `nnictl hello`. After this is installed, we can then run the main file by `python3 nni_hello_hyp/main.py`. If this successfully executes, then we will be able to see the NNI UI in the local broswer and it will look like following.

8. That is it, everything has been successfully setup with the use of poetry!
