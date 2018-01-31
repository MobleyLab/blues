
proc range {num args} {
    switch [llength $args] {
        0 {
            set from 0
            set to $num
            set step 1
        }
        1 {
            set from $num
            set to $args
            set step 1
        }
        2 {
            set from $num
            lassign $args to step
        }
    }
    set result [list ]
    for {set i $from} {$i < $to} {set i [expr {$i + $step}]} {
        lappend result $i
    }
    return $result
}

proc save_viewpoint {} {
   global viewpoints
   if [info exists viewpoints] {unset viewpoints}
   # get the current matricies
   foreach mol [molinfo list] {
      set viewpoints($mol) [molinfo $mol get {
	center_matrix rotate_matrix scale_matrix global_matrix}]
   }
}

proc restore_viewpoint {} {
   global viewpoints
   foreach mol [molinfo list] {
      puts "Trying $mol"
      if [info exists viewpoints($mol)] {
         molinfo $mol set {center_matrix rotate_matrix scale_matrix
	   global_matrix} $viewpoints($mol)
      }
   }
}

proc take_picture {args} {
  global take_picture

  # when called with no parameter, render the image
  if {$args == {}} {
    set f [format $take_picture(format) $take_picture(frame)]
    # take 1 out of every modulo images
    if { [expr $take_picture(frame) % $take_picture(modulo)] == 0 } {
      render $take_picture(method) $f
      # call any unix command, if specified
      if { $take_picture(exec) != {} } {
        set f [format $take_picture(exec) $f $f $f $f $f $f $f $f $f $f]
        eval "exec $f"
       }
    }
    # increase the count by one
    incr take_picture(frame)
    return
  }
  lassign $args arg1 arg2
  # reset the options to their initial stat
  # (remember to delete the files yourself
  if {$arg1 == "reset"} {
    set take_picture(frame)  0
    set take_picture(format) "./animate.%04d.rgb"
    set take_picture(method) snapshot
    set take_picture(modulo) 1
    set take_picture(exec)    {}
    return
  }
  # set one of the parameters
  if [info exists take_picture($arg1)] {
    if { [llength $args] == 1} {
      return "$arg1 is $take_picture($arg1)"
    }
    set take_picture($arg1) $arg2
    return
  }
  # otherwise, there was an error
  error {take_picture: [ | reset | frame | format  | \
  method  | modulo ]}
}
# to complete the initialization, this must be the first function
# called.  Do so automatically.
take_picture reset

proc get_colormap {n_clusters data} {
    #Desired color mapping:
        #0 red 1 orange 2 yellow 3 green 4 lime 5 cyan 6 blue 7 purple
    # VMD colormap
        #0  1  1    3   2    4    3   7  4  12  5  10  6  0   7   11
    #set colormap [string map {0 1 1 3 2 4 3 7 4 12 5 10 6 0 7 11} $data]
    #set colors [string map {0 blue 1 red 2 gray 3 orange 4 yellow 5 tan 6 silver 7 green 8 white 9 pink 10 cyan 11 purple '12' lime 13 mauve 14 ochre 15 iceblue 16 black 17 'yellow2' 18 'yellow3' 19 'green2' 20 'green3' 21 'cyan2' 22 'cyan3' 23 'blue2' 24 'blue3' 25 violet 26 'violet2' 27 magenta 28 'magenta2' 29 'red2' 30 'red3' 21 'orange2' 33 'orange3'} $colormap]
    switch $n_clusters {
        2 {
            #Red / Green
            set colormap [string map {0 1 1 7} $data]
        }
        3 {
            #Red / Green / Blue
            set colormap [string map {0 1 1 7 2 23} $data]
        }
        4 {
            #Red / Yellow / Green / Blue
            set colormap [string map {0 1 1 4 2 7 3 23} $data]
        }
        5 {
            #Red / Yellow / Green / Blue / Purple
            set colormap [string map {0 1 1 4 2 7 3 23 4 11} $data]
        }
        6 {
            #Red / Yellow / Green / Cyan / Blue / Purple
            set colormap [string map {0 1 1 4 2 7 3 10 4 23 5 11} $data]
        }
        7 {
            #Red / Orange / Yellow / Green / Cyan / Blue / Purple
            set colormap [string map {0 1 1 3 2 4 3 7 4 10 5 23 6 11} $data]
        }
        8 {
            #Red / Orange / Yellow / Green / Lime / Cyan / Blue / Purple
            set colormap [string map {0 1 1 3 2 4 3 7 4 12 5 10 6 23 7 11} $data]
        }
        9 {
            #Red / Orange / Yellow / Green / Lime / Cyan / Blue / Purple / Violet
            set colormap [string map {0 1 1 3 2 4 3 7 4 12 5 10 6 23 7 11 8 25} $data]
        }
    }
    return $colormap
}

proc get_n_clusters {testlist} {
    foreach el $testlist {
        set array_tmp($el) {}
    }
    set unique [array names array_tmp]
    return [llength $unique]
}

proc set_viewpoint {} {
    set viewpoints([molinfo top]) {{{1 0 0 -28.8648} {0 1 0 -30.1644} {0 0 1 -34.5486} {0 0 0 1}} {{-0.565023 0.755831 -0.33085 0} {0.502784 -0.00251855 -0.864407 0} {-0.65418 -0.654758 -0.378597 0} {0 0 0 1}} {{0.0872075 0 0 0} {0 0.0872075 0 0} {0 0 0.0872075 0} {0 0 0 1}} {{1 0 0 0.8} {0 1 0 -0.62} {0 0 1 0} {0 0 0 1}}}
    lappend viewplist [molinfo top]
    foreach v $viewplist {
      molinfo $v set {center_matrix rotate_matrix scale_matrix global_matrix} $viewpoints($v)
    }
}

proc loadtraj {pdb dcd jobname sta fin} {
    #Load cms file/trajectories
    mol new ${pdb} type {pdb} first $sta last $fin step 1 waitfor 1
    mol addfile ${dcd} type {dcd} first $sta last $fin step 1 waitfor all
    animate delete beg 0 end 0 skip 0 0
    set repmolid [molinfo top get id]
    mol rename $repmolid ${jobname}
    # delete all representations
    set numrep [molinfo $repmolid get numreps]
    for {set i 0} {$i < $numrep} {incr i} {
        mol delrep $i $repmolid
    }
    # Add rep of protein
    mol color {ColorID 15}
    mol representation NewCartoon
    mol selection protein
    mol material AOEdgy
    mol addrep $repmolid
    # Add rep of ligand
    mol representation Licorice
    mol selection {resname LIG and (not name "H.*")}
    #{segname "HET.*"}
    mol material AOEdgy
    mol color Name
    mol addrep $repmolid

    mol smoothrep top 0 2
    mol smoothrep top 1 2
}


proc play_animation {} {
    global labels
    #Initialize output for writing RMSD
    set f [open ${labels} r]
    set data [split [string trim [read $f]]]
    close $f

    #Select trajectory we are comparing and loop over frames
    set num_frames [molinfo top get numframes]
    set n_clusters [get_n_clusters $data]
    set cmap [get_colormap ${n_clusters} $data]
    set colors [string map {0 blue 1 red 2 gray 3 orange 4 yellow 5 tan 6 silver 7 green 8 white 9 pink 10 cyan 11 purple '12' lime 13 mauve 14 ochre 15 iceblue 16 black 17 'yellow2' 18 'yellow3' 19 'green2' 20 'green3' 21 'cyan2' 22 'cyan3' 23 'blue2' 24 'blue3' 25 violet 26 'violet2' 27 magenta 28 'magenta2' 29 'red2' 30 'red3' 21 'orange2' 33 'orange3'} $cmap]

    foreach i [range ${num_frames}] idx $data cm $cmap co $colors {
      puts "Frame=${i} Cluster=${idx} ColorID=${cm} : ${co}"
      animate goto ${i}
      color Name C $cm;
      display update
      take_picture 
    }
}

set relpath ../blues/analysis/tests/data
set pdb "${relpath}/run03-centered.pdb"
set dcd "${relpath}/run03-centered.dcd"
set labels "${relpath}/run03-labels.txt"

loadtraj ${pdb} ${dcd} run03 0 2001
set_viewpoint

#play_animation
