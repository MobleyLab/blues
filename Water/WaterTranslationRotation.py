class WaterTranslationMove(Move):
    """ Move that translates a random water within a specified radius of the protein's
    center of mass to another point within that radius
    Parameters
    ----------
    structure:
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        water_name: str, optional, default='WAT'
            Residue name of the waters in the system.
        radius: float*unit compatible with simtk.unit.nanometers, optional, default=2.0*unit.nanometers
            Defines the radius within the protein center of mass to choose a water
            and the radius in which to randomly translate that water.
    """

    def __init__(self, structure, water_name='WAT', radius=20.*unit.nanometers):
        #initialize self attributes
        self.radius = radius #
        self.water_name = water_name
        self.water_residues = [] #contains indices of the atoms of the waters
        self.protein_atoms = [] #contains indices of the atoms in the protein residues
        self.before_ncmc_check = True

        #go through the topology and identify water and protein residues
        residues = structure.topology.residues()
        #looks for residues with water_name ("WAT")
        for res in residues:
            if res.name == self.water_name: #checks if the name of the residue is 'WAT'
                water_mol = [] #list of each waters atom indices
                for atom in res.atoms():
                    water_mol.append(atom.index) #append the index of each of the atoms of the water residue
                self.water_residues.append(water_mol)#append the water atom indices as a self attribute (above)

        residues = structure.topology.residues() #Iterates over all Residues in the Topology.

        #looks for residues with a 'CA' atom name to identify protein residues
        for res in residues:
            atom_names = [] #list of the name of each atom in protein residues
            atom_index = [] #list of the index of each atom in protein residues
            for atom in res.atoms(): #for each atom in the residues
                atom_names.append(atom.name) #add the atom name to the list above
                atom_index.append(atom.index) #add the index of that atom
            if 'CA' in atom_names: #'CA' denotes protein residue
                self.protein_atoms = self.protein_atoms+atom_index #prot_atoms contains indices of the proteins atoms?

        #set more self attributes
        #self.atom_indices is used to define the alchemically treated region
        #of the system
        #in this case the first water in the system
        self.atom_indices = self.water_residues[0] #the atom indices of the first water, this is the alchemical water
        #Topology: describes the organization of atoms into residues, bonds, etc
        #The python object `parmed.Structure` contains all the topology information.
        #It’s assumed to be the topology of the entire system.
        #By providing it the specific atom indices, you get the topology info based on given indices subset.
        self.topology_protein = structure[self.protein_atoms].topology #typology info based on indices of the atoms in protein residues
        self.topology_water = structure[self.atom_indices].topology  #typology info based on the indices of the first waters atom indices in the system
        self.water_mass = self.getMasses(self.topology_water)#Provides the mass of the specified waters atom
        self.protein_mass = self.getMasses(self.topology_protein)#Provides the mass of each of the proteins atoms

        #attributes, random rotational
        self.masses = []

    def _random_sphere_point(self, radius):
        """function to generate a uniform random point
        in a sphere of a specified radius.
        Used to randomly translate the water molecule
        Parameters
        ----------
        radius: float
            Defines the radius of the sphere in which a point
            will be uniformly randomly generated.
        """
        #numpy.random.random: Returns random floats in the half-open interval [0.0, 1.0).
        #numpy.random.uniform: Draw samples from a uniform distribution.
        #Samples are uniformly distributed over the half-open interval [low, high)
        #(includes low, but excludes high). In other words, any value within the
        #given interval is equally likely to be drawn by uniform.

        #If you draw a line from the origin to the point, the first parameter is
        #length of this line, represented by r.
        # r is not evenly distributed (ie there's an unequal chance for any of the numbers to be picked).
        #Larger numbers are favored here, which is good, because there's more area in a sphere farther out.
        #Note: r doesn't have to have even distribution, but the end result does (sphere_point). This is because of how a sphere is. Analogy: a circular auditorium
        #there's fewer seats in the inner rows than in the outer rows. If you were to pick a completely random seat in the whole arena with an even probability, you'd want to pick
        #the rows further away more often. And so, there distribution over which you picked rows wouldn't be even, but the overall distribtion for picking seats would be even.
        #This is because there's more options, if you picked the seats in the center all the time there's a higher chance/frequency you'd pick the same one.

        r = radius * ( np.random.random()**(1./3.) )  #r (radius) = specified radius * cubed root of a random number between 0.00 and 0.99999
        phi = np.random.uniform(0,2*np.pi) #restriction of phi (or azimuth angle) is set from 0 to 2pi. random.uniform allows the values to be chosen w/ an equal probability
        costheta = np.random.uniform(-1,1) #restriction set from -1 to 1
        u = np.random.random() #I don't see this getting used anywhere...
        theta = np.arccos(costheta) #calculate theta, the angle between r and Z axis
        x = np.sin(theta) * np.cos(phi) #x,y,and z are cartesian coordinates
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        sphere_point = np.array([x, y, z]) * r
        return sphere_point #sphere_point = a random point with an even distribution

        #Notes for above: Cartesian coordinates are [x,y,z] coordinates, and is how you define a point in 3 dimensional space.
        #If you picked a random point inside of a cube, it's straight forward (ie pick a random x value inside the width, and a
        #random y value inside the height, and a random z value inside of the depth). So the boundaries of a cube/box are straight forward.
        #With a sphere, this isn't the case. To define a sphere's boundaries you do r^2 = x^2 + y^2 +z^2. This makes it harder to choose a random
        #cartesian coordinates inside of a sphere.
        #A spherical coord is defined by two angles and a radius. 1 angle in the x/y plane theta, some angle phi between the x/y plane and the z-axis, and a radius
        #So, a spherical coord is defined by theta, phi and r
        #Using trig, we can go back and forth between spherical and Cartesian
        #It's easier to pick a random theta, random phi and random r, and then find a point in a spherical coordinate and convert it back to cartesian coordinates
        #If it works for spherical, it'll work for cartesian.


    def getMasses(self, topology):
        """Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        """
        masses = unit.Quantity(np.zeros([int(topology.getNumAtoms()),1],np.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass #gets the mass of the atom, adds to list (along with index)
        return masses

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array
        Parameters
        ----------
        positions: parmed.Structure
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        print('masses', masses)
        print('type', type(masses))
        print('type2', type(masses[0]))
        print('masses[0]', masses[0]/ unit.dalton * unit.dalton) #first atoms mass / dalton*dalton, why?
        print('dir', dir(masses)) #prints list of atom masses
        #print('value', positions.value_in_unit(positions.unit))
        print(positions) #positons: A list of 3-element Quantity tuples of dimension length representing the atomic positions for every atom in the system.
        coordinates = np.asarray(positions._value, np.float32) #gives the value of atomic positions as an array
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

    def beforeMove(self, nc_context):
        """
        Temporary function (until multiple alchemical regions are supported),
        which is performed at the beginning of a ncmc iteration. Selects
        a random water within self.radius of the protein's center of mass
        and switches the positions and velocities with the alchemical water
        defined by self.atom_indices, effecitvely duplicating mulitple
        alchemical region support.
        Parameters
        ----------
        nc_context: simtk.openmm Context object
            The context which corresponds to the NCMC simulation.
        """
        start_state = nc_context.getState(getPositions=True, getVelocities=True)
        start_pos = start_state.getPositions(asNumpy=True) #gets starting positions
        print('start_pos', start_pos[self.atom_indices[0]]) #prints starting position of the first water atom

        start_vel = start_state.getVelocities(asNumpy=True) #gets starting velocities
        switch_pos = np.copy(start_pos)*start_pos.unit #starting position (a shallow copy) is * by start_pos.unit to retain units
        switch_vel = np.copy(start_vel)*start_vel.unit #starting vel (a shallow copy) is * by start_pos.unit to retain units
        print('switch_pos', switch_pos)
        #Get the center of mass of the protein, need position (which eventually gets converted to coordinates) and masses
        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms], #passes in a copy of the protein atoms starting position
                            masses = self.protein_mass) #passes in list of the proteins atoms masses

        #pick random water within the sphere radius
        dist_boolean = 0
        #TODO use random.shuffle to pick random particles (limits upper bound)
        while dist_boolean == 0:
            #water_choice = np.random.choice(water_residues)
            water_index = np.random.choice(range(len(self.water_residues))) #chooses a random water number (based on the range of the length of the list containing the water atoms indices)
            water_choice = self.water_residues[water_index] #pass the random water number into wat_res to get the indices of its atoms.
            #We now have the the indices of the random waters atoms
            oxygen_pos = start_pos[water_choice[0]] # pass the first atom indice of the random water, get the starting positions of that waters atoms
            #get the distance between the randomly chosen water and the proteins center of mass
            #np.linalg.norm(x - y)) will give you Euclidean distance between the vectors x and y (ie "ordinary" straight-line distance between two points in Euclidean space).
            water_distance = np.linalg.norm(oxygen_pos._value - prot_com._value)
            print('water_distance', water_distance)
            #If the waters distance is <= to the specified radius
            if water_distance <= (self.radius.value_in_unit(unit.nanometers)):
                dist_boolean = 1
            print('water_choice', water_choice)
        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            #set indices of the alchemical waters atoms equal to the indices of the starting positions of the random waters atoms
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            #do the same for velocity
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
            #set indices of the randomly chosen waters atom equal to alchemical waters atom indices. Same w/ velocity
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]

        print('after_switch', switch_pos[self.atom_indices[0]]) #prints the new indices of the alchemical water
        nc_context.setPositions(switch_pos)
        nc_context.setVelocities(switch_vel)

        return nc_context

    def calculateProperties(self):
        """Function to quickly calculate available properties."""
        self.masses, self.totalmass = self.getMasses(self.topology)
        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)

    def move(self, context):
        """
        This function is called by the blues.MoveEngine object during a simulation.
        Translates the alchemical water randomly within a sphere of self.radius.
        """
        #get the position of the system from the context
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        protein_pos = before_move_pos[self.protein_atoms] #gets the positions from the indices of the atoms in the protein residues in relation to the system
        water_pos = before_move_pos[self.atom_indices] # gets positions of the alchemical waters atoms

        #find the center of mass and the displacement
        prot_com = self.getCenterOfMass(positions=protein_pos, masses=self.protein_mass) #gets protein COM
        water_com = self.getCenterOfMass(positions=water_pos, masses=self.water_mass) #gets protein COM

        sphere_displacement = self._random_sphere_point(self.radius) #gets a uniform random point in a sphere of a specified radius
        movePos = np.copy(before_move_pos)*before_move_pos.unit #makes a copy of the position of the system from the context

        print('movePos', movePos[self.atom_indices]) #gets positions of the alchemical waters atoms from the context
        print('center of mass', prot_com) #prints the protein COM
        print('Water coord', self.atom_indices) #prints alchemical waters atoms indices

        #first atom in the water molecule (which is Oxygen) was used to measure the distance
        #water_dist = movePos[self.atom_indices[0]] - prot_com] #estimate distance of the water from the proteins com.
        water_dist = movePos[water_com - prot_com] #here, distance of the alch. water com from the protein com

        #TODO: make water within radius selection correctly handle PBC
        print('water_dist._value', np.linalg.norm(water_dist._value)) #prints alch. waters numerical euclidean distance
        print('self.radius._value', self.radius._value) #prints numerical value of radius

        #if the alchemical water is within the radius, translate it
        if np.linalg.norm(water_dist._value) <= self.radius._value: #see if euc. distance of alch. water is within defined radius
            for index, resnum in enumerate(self.atom_indices):
                # positions of the the alch. water atoms - distance of the alch. water from protein com + sphere displacement
                movePos[resnum] = movePos[resnum] - water_dist + sphere_displacement #new positions of the alch water
                print('before', before_move_pos[resnum])
                print('after', movePos[resnum])
            context.setPositions(movePos) #Sets the positions of particles


        #Perform a random rotation about the alch. waters center of mass
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        self.positions = positions[self.atom_indices] #get positions of the alch water from context
        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)
        reduced_pos = self.positions - self.center_of_mass

        #Define random rotational move on the water
        rand_quat = mdtraj.utils.uniform_quaternion()
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rot_move = np.dot(reduced_pos, rand_rotation_matrix) * positions.unit + self.center_of_mass

        #Update alch. water positions in nc_sim
        for index, atomidx in enumerate(self.atom_indices):
            positions[atomidx] = rot_move[index]
        context.setPositions(positions)
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices]
        return context
