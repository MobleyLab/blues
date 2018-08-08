Introduction
============

``chemper`` contains a variety of tools that will be useful in
automating the process of chemical perception for the new
SMIRKS Native Open Force Field (`SMIRNOFF <https://github.com/openforcefield/openforcefield>`_)
format as a part of the `Open Force Field Consortium <http://openforcefield.org>`_ :ref:`[1] <refs>`.

This idea originated from the tools `SMARTY and SMIRKY <https://github.com/openforcefield/smarty>`_
which were designed to use an automated monte carlo algorithm to
sample the chemical perception used in existing force fields.
SMARTY samples SMARTS patterns corresponding to traditional atom
types and was tested compared to the `parm99/parm@frosst <http://www.ccl.net/cca/data/parm_at_Frosst/>`_
force field. SMIRKY is an extension of SMARTY created to sample SMIRKS
patterns corresponding to SMIRNOFF parameter types
(nonbonded, bond, angle, and proper and improper torsions).
SMIRKY was tested against our own `smirnoff99Frosst <https://github.com/openforcefield/smirnoff99Frosst>`_

One of the most important lessons learned while testing SMARTY
and SMIRKY is that the combinatorial problem in SMIRKS space is
very large. These tools currently use very naive moves in SMIRKS
space chosing atom or bond decorators to add to a pattern at
complete random. This wastes a signficant amount of time making
chemically insensible moves. One of the take away conclusions
on that project was that future chemical perception sampling
tools would need to take atom and bond information from input
molecules in order to be feasible :ref:`[2] <refs>`.

We developed ``chemper`` based on the knowledge of the SMARTY
project outcomes. The goal here is to take clustered molecular
subgraphs and generate SMIRKS patterns. These tools will use
information stored in the atoms and bonds of a molecule to drive
choices about SMIRKS decorators. Then will automatically
generate reasonable SMIRKS patterns matching clustering of
molecular subgraphs.

Currently, ``chemper`` provides modules for generating SMIRKS patterns from molecule objects and specified atoms.
The final product will be capable of clustering molecular fragments based on reference data and
then assigning SMIRKS patterns for each of those clusters.
See :doc:`installation`.

Contributors
------------

* `Caitlin Bannan (UCI) <https://github.com/bannanc>`_
* `David L. Mobley (UCI) <https://github.com/davidlmobley>`_

Acknowledgments
---------------

CCB is funded by a fellowship from
`The Molecular Sciences Software Institute <http://molssi.org/>`_
under NSF grant ACI-1547580.

.. _refs:

References
----------

1. D. Mobley et al. bioRxiv 2018, 286542. `doi.org/10.1101/286542 <http://doi.org/10.1101/286542>`_
2. C. Zanette and C.C. Bannan et al. chemRxiv 2018, `doi.org/10.26434/chemrxiv.6230627.v1 <https://doi.org/10.26434/chemrxiv.6230627.v1>`_
