# Setup on GreenPlanet and TSCC
Login to the remote clusters and then do the following:

## Install Anaconda
*While in your HOME directory*

Download the Anaconda installation file
> wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh

Install Anaconda (this may take 15-30mins)
> bash Anaconda3-4.3.1-Linux-x86_64.sh -b

When it asks you to add Anaconda to your bash shell PATH, select **YES**.

Check that Anaconda installed properly by running the command `python`. It's output should look something like:
```
Python 3.6.0 |Anaconda 4.3.0 (64-bit)| (default, Dec 23 2016, 12:22:00)
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Install some extra packages into a virtual environment
All our software needs to be run on **Python 3.5** so we will create a virtual environment and install some extra packages we will need down the line.

Create a virtual environment called `py35` with:
>conda create -n py35 python=3.5

Activate the virtual environment with:
>source activate py35

The terminal will look like something below to indicate you're in the virtual environment:
```
[limn1@tscc-login2 ~]$ source activate py35  
(py35) [limn1@tscc-login2 ~]$
```

Now, install the extra packages we will need down the line (this could take another 15mins):
```bash
conda install numpy matplotlib
conda install nb_conda -c conda-forge
conda install -c omnia -c omnia/label/dev -c mobleylab openmm==7.1.0rc1 openmoltools==0.7.5 ambermini==16.16.0 smarty==0.1.4 parmed==2.7.1 alchemy==1.2.3 pdbfixer==1.4 smirnoff99frosst==1.0.5 yank==0.15.2
```

## Install OpenEye toolkits
*While still in your py35 virtualenv*

Make a license folder in your **HOME** directory:
> mkdir licenses

Your path should look like `/home/USERNAME/licenses`.

Upload/copy the `oe_license.txt` file into the `licenses` folder.

Add to your `~/.bash_profile` the following line:
> export OE_LICENSE='$HOME/licenses/oe_license.txt"'

Install the OpenEye toolkits with:
> pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits

Verify the installation with:
> oecheminfo.py

The output should look something like:
```
Installed OEChem version: 2.1.1 platform: linux-g++4.x-x64 built: 20170210

Examples: /home/limn1/anaconda3/envs/dev/lib/python3.5/site-packages/openeye/examples
Doc Examples: /home/limn1/anaconda3/envs/dev/lib/python3.5/site-packages/openeye/docexamples

code| ext           | description                      |read? |write?
----+---------------+----------------------------------+------+------
  1 | smi           | Canonical stereo SMILES          | yes  | yes
  2 | mdl,mol,rxn   | MDL Mol                          | yes  | yes
  3 | pdb,ent       | PDB                              | yes  | yes
  4 | mol2,syb      | Tripos MOL2                      | yes  | yes
  5 | usm           | Non-Canonical non-stereo SMILES  | yes  | yes
  6 | ism,isosmi    | Canonical stereo SMILES          | yes  | yes
  7 | mol2h         | MOL2 with H                      | yes  | yes
  8 | sdf,sd        | MDL SDF                          | yes  | yes
  9 | can           | Canonical non-stereo SMILES      | yes  | yes
 10 | mf            | Molecular Formula                | no   | yes
 11 | xyz           | XYZ                              | yes  | yes
 12 | fasta,seq     | FASTA                            | yes  | yes
 13 | mopac,pac     | MOPAC                            | no   | yes
 14 | oeb           | OEBinary v2                      | yes  | yes
 15 | dat,mmd,mmod  | Macromodel                       | yes  | yes
 16 | sln           | Tripos SLN                       | no   | yes
 17 | rdf,rd        | MDL RDF                          | yes  | no
 18 | cdx           | ChemDraw CDX                     | yes  | yes
 19 | skc           | MDL ISIS Sketch File             | yes  | no
 20 | inchi         | IUPAC InChI                      | no   | yes
 21 | inchikey      | IUPAC InChI Key                  | no   | yes
 22 | csv           | Comma Separated Values           | yes  | yes
 23 | json          | JavaScript Object Notation       | yes  | yes
----+---------------+----------------------------------+------+------
```

# [Recommended] Install Bash on Windows (Locally)
*On your local Windows laptop*

Follow the guide linked [here](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide)

If the BASH terminal guide works and you can successfully use BASH commands (i.e `cd`, `ls`). Do let me know, this is a pretty big deal and will help other people who aren't on MacOS/Linux. Now, try repeating the steps above to see if we can get Anaconda/OpenEye installed on your local machine too.

If you run into too many issues along the way and it's become a headache, don't worry about it. I'm not requiring you do this, but this may help alleviate some issues when running on a Windows machine and prevent problems when trying to translate between running on different OS. This feature is still in BETA and I have not tried it out myself. So I'm really hoping it works out on your machine. This will replace PuTTY so that your terminal can more closely mimic the command terminal found on Linux/MacOS machines or when you login to remote clusters.
