# Drafting a new development build
Take the following steps to draft a new development build without issuing a new release:

1. Create a pull request to the master branch with your *SMALL* changes.
2. Increment the *DEVBUILD* string in `setup.py` and run `python setup.py develop` to update the `blues/version.py` and `devtools/conda-recipe/meta.yaml` files.
3. Commit and push the changes to 'setup.py', `blues/version.py` and `devtools/conda-recipe/meta.yaml`
4. Review and merge your changes to master.
5. Travis will build and upload the anaconda package to the `mobleylab` channel with the `dev` label and can be installed with the command: `conda install -c mobleylab/label/dev blues`

# Issuing a Release
Take the following steps to issue a new (stable) release candidate.

1. Create a new branch/PR with the name `release-X.X.X` using the appropriate release version number.
2. Update the `README.md` with the release version number and a short statement on the changes.
3. In `setup.py` make *DEVBUILD = 0* and *ISRELEASED = TRUE* and run `python setup.py develop` to update the `blues/version.py` and `devtools/conda-recipe/meta.yaml` files.
4. In `devtools/conda-recipe/conda_upload.sh` replace the `-l dev` label with `-l main`
5. Commit and push the changes to `README.md`, `setup.py`, `blues/version.py`, `devtools/conda-recipe/meta.yaml`, and `devtools/conda-recipe/conda_upload.sh`.
6. Review and merge your changes to master.
7. Travis will build and upload the anaconda package to the `mobleylab` channel with the `main` label and can be installed with the command: `conda install -c mobleylab blues`
7. Draft a new release on the repository page with update notes, detailing the PRs incorporated in this release.
8. A new zendo DOI should be automatically issued and updated on the main repository page.
