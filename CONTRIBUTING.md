# How to contribute to super-pixels?

**Note**: We use VSCode for managing our project. Hence, we strongly recommend that you do the same. You can check the settings for the workspace in `.vscode` directory. However, you can also choose to work without VSCode :) Our commands are flexible enough to work with 

1. Fork the [repository](https://github.com/abheesht17/super-pixels) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone git@github.com:<your Github handle>/super-pixels.git
	cd super-pixels
	git remote add upstream https://github.com/abheesht17/super-pixels.git
	```

3. Create a new branch to hold your development changes:

	```bash
	git checkout -b a-descriptive-name-for-my-changes
	```

	**DO NOT** work on the `master` branch. Use `-` (hyphens) for separating words in branch names. Avoid uppercase branch names.

4. Set up a development environment by running the following command in a virtual environment:

	```bash
	make env
	```

5. Develop the features on your branch. Properly document your code. You want to provide enough information for someone to be able to make tweaks to your code and make it even better :). We use Google style documentation for our files - See [this link](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for more details.

6. Test your code- To test your code locally, activate the environment using the following command:
	
	```bash
	source superpixels-env/bin/activate
	```
	Otherwise, if you are using a Colab notebook, install all the requirements using:

	```bash
	pip install -r requirements.txt
	```

	**Note**: If you have to add any new packages, please ensure the environment is activated so that all the packages get added when you run `make final` (see next step).

7. If you have installed any new packages in your environment, please run the following:

    ```bash
    make final
    ```

	If you have not added any new packages, you can choose to just format your code using black and isort with the following command:

	```bash
	make style
	```

	**Note**: We only format the code in the `src` directory or right outside

8. Use the following command to check for any linting issues:

	```python
	make quality
	```
	**Note**: We ignore "unused imports" and "unable to detect undefined names" issue. We do this because we use a `configmapper` to map the classes, which uses the imports indirectly and hence leads to these issue. Therefore, try not to have unused imports and `import *` as much as possible because we don't check for them :)

	**Note**: We also ignore "line break before binary operator" because `black` often tends to produce such lines.

	**Note**: VSCode settings do not ignore any such warnings, so you are aware of these issues.

9. Once you're happy with your changes, add your changes and make a commit to record your changes locally:

	```bash
	git add <your changed files>
	git commit
	```

10. If you are publishing you branch for the first time, perform a rebase on the upstream branch as follows:

    ```bash
    git fetch upstream
	git rebase upstream/master
    ``` 
    If the branch already exists on your remote origin, then you can merge before pushing your changes:

	```bash
	git fetch upstream
	git merge upstream/master
    ```

11. Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

12. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review. If your PR has the potential to solve an issue, mention it in the comments.