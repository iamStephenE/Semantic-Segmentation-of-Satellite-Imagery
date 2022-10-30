# CS 301 - 103: Semantic Segmentation of Satellite Imagery

## Group 8: Stephen Ebrahim and Ebram Mekhail

This branch contains Milestone 1 of the project.

## Milestone 1: Environment Preparation

Note: These Commands were run on macos

1. Initially, we created a repository for the project on GitHub.
2. Secondly, we made sure that Python was in a relatively new version, in our case since we had python using Homebrew, we updated it through that: `brew upgrade python3`
3. Then we set up poetry by installing it through the following command: `curl -sSL https://install.python-poetry.org | python3 -`
4. To ensure that poetry was available globally, we also had to include its path the in the bash configuration file. We included this in ~/.zshrc `export PATH="/Users/<your_user_name>/.local/bin:$PATH"` 
5. Then, we added the necessary dependencies using `poetry add [name of dependency]`. We did this for python, tensorflow, pandas, numpy, matplotlib, and, **NNI**. Furthermore, to ensure that everything was up to date using poetry, we executed the following command: `poetry update package`
6. Now, we tested that everything was working properly by setting up the sample project of NNI. First, we open the poetry shell by running `poetry shell`. Now that we are in the environment of poetry, we can then use installed packages/dependencies. To install the packages you have to first do `poetry build` then `poetry lock`.
7. We install the NNI hello project by executing `nnictl hello`. After this is installed, we can then run the main file by `python3 nni_hello_hyp/main.py`. If this successfully executes, then we will be able to see the NNI UI in the local browser and it will look like the following.

![NNI_UI](https://user-images.githubusercontent.com/66531257/198859714-087b5673-840c-4146-b70c-d85baba48779.png)

8. That is it, everything has been successfully set up with the use of poetry!
