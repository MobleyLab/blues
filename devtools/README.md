# Issuing a Release
After merging changes from a pull request to the master branch, take the following steps to issue a new release.

1. Create a new branch/PR to make the changes in version numbers.
2. Update the `README.md` with the version number and a short statement on the changes.
3. Update the version numbers in `setup.py` and the `meta.yaml` file.
4. Run `setup.py` to update the `blues/version.py` file accordingly.
5. Commit and push the changes to `README.md`, `setup.py`, `blues/version.py` and the `meta.yaml`.
6. Merge your changes to master.
7. Draft a new release on the repository page with update notes and to generate a new zendo DOI.

# Updating the conda package
After issuing a new release, the conda package needs to be updated on the channel.

1. Clone the repository or from a **clean** directory, navigate to the top `blues/`.
2. Build the python3.5 package with `conda build --python 3.5 .`
3. Build the python2.7 package with `conda build --python 2.7 .`
4. Upload the packages to the [MobleyLab](https://anaconda.org/mobleylab/blues/files) anaconda channel
