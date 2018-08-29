Introduction
============
.. figure:: ../images/blues.png
   :scale: 15 %
   :target: https://github.com/MobleyLab/blues

   BLUES is a python package that takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes.


Github
------
.. role:: raw-html(raw)
   :format: html

:raw-html:`<a class="github-button" href="https://github.com/mobleylab/blues" data-size="large" data-show-count="false" aria-label="Star mobleylab/blues on GitHub">Github</a><script async defer src="https://buttons.github.io/buttons.js"></script>` Check out our Github repository.

:raw-html:`<a class="github-button" href="https://github.com/mobleylab/blues/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="false" aria-label="Issue mobleylab/blues on GitHub">Issue</a><script async defer src="https://buttons.github.io/buttons.js"></script>` If you have any problems or suggestions through our issue tracker.

:raw-html:`<a class="github-button" href="https://github.com/mobleylab/blues/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork mobleylab/blues on GitHub">Fork</a><script async defer src="https://buttons.github.io/buttons.js"></script>` To contribute to our code, please fork our repository and open a Pull Request.

.. image:: https://travis-ci.org/MobleyLab/blues.svg?branch=master
   :target: https://travis-ci.org/MobleyLab/blues

.. image:: https://readthedocs.org/projects/mobleylab-blues/badge/?version=master
   :target: https://mobleylab-blues.readthedocs.io/en/master/?badge=master

.. image:: https://anaconda.org/mobleylab/blues/badges/version.svg
   :target: https://anaconda.org/mobleylab/blues

.. image:: https://zenodo.org/badge/62096511.svg
   :target: https://zenodo.org/badge/latestdoi/62096511


Publication
-----------
.. image:: ../images/jp-2017-11820n_0015.gif
   :target: https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b11820

.. rubric:: Binding Modes of Ligands Using Enhanced Sampling (BLUES): Rapid Decorrelation of Ligand Binding Modes via Nonequilibrium Candidate Monte Carl

Samuel C. Gill, Nathan M. Lim, Patrick B. Grinaway, Ariën S. Rustenburg, Josh Fass, Gregory A. Ross, John D. Chodera , and David L. Mobley

*The Journal of Physical Chemistry B* **2018** *122* (21), 5579-5598

**DOI:** `10.1021/acs.jpcb.7b11820 <https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b11820>`_

Publication Date (Web): February 27, 2018

.. rubric:: Abstract

.. epigraph::
   Accurately predicting protein–ligand binding affinities and binding modes is a major goal in computational chemistry, but even the prediction of ligand binding modes in proteins poses major challenges. Here, we focus on solving the binding mode prediction problem for rigid fragments. That is, we focus on computing the dominant placement, conformation, and orientations of a relatively rigid, fragment-like ligand in a receptor, and the populations of the multiple binding modes which may be relevant. This problem is important in its own right, but is even more timely given the recent success of alchemical free energy calculations. Alchemical calculations are increasingly used to predict binding free energies of ligands to receptors. However, the accuracy of these calculations is dependent on proper sampling of the relevant ligand binding modes. Unfortunately, ligand binding modes may often be uncertain, hard to predict, and/or slow to interconvert on simulation time scales, so proper sampling with current techniques can require prohibitively long simulations. We need new methods which dramatically improve sampling of ligand binding modes. Here, we develop and apply a nonequilibrium candidate Monte Carlo (NCMC) method to improve sampling of ligand binding modes. In this technique, the ligand is rotated and subsequently allowed to relax in its new position through alchemical perturbation before accepting or rejecting the rotation and relaxation as a nonequilibrium Monte Carlo move. When applied to a T4 lysozyme model binding system, this NCMC method shows over 2 orders of magnitude improvement in binding mode sampling efficiency compared to a brute force molecular dynamics simulation. This is a first step toward applying this methodology to pharmaceutically relevant binding of fragments and, eventually, drug-like molecules. We are making this approach available via our new Binding modes of ligands using enhanced sampling (BLUES) package which is freely available on GitHub.

Theory
------
Suggested readings:

.. [1] `Best Practices for Foundations in Molecular Simulations : v1.0. Efrem Braun, Justin Gilmer, Heather B. Mayes, David L. Mobley, Jacob I. Monroe, Samarjeet Prasad, Daniel M. Zuckerman.  <https://github.com/MobleyLab/basic_simulation_training/blob/master/paper/basic_training.pdf>`_
.. [2] `Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. Jerome P. Nilmeier, Gavin E. Crooks, David D. L. Minh, and John D. Chodera. PNAS November 8, 2011. 108 (45) E1009-E1018 <http://www.pnas.org/content/108/45/E1009>`_
