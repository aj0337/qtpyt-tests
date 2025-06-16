#!/usr/local/Cellar/gnuplot/6.0.0/bin/gnuplot
#
#    
#    	G N U P L O T
#    	Version 6.0 patchlevel 0    last modified 2023-12-09 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2023
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
set terminal epslatex standalone color colourtext size 10.0 cm, 15.0 cm font ',8pt'
set output "trans.tex"

set multiplot

set grid

xmin=-4
xmax=3
set xlabel '$E-E_F$ [eV]'
set xrange [xmin:xmax]
set xtics 1
set format x '\small $%g$'

ymin=1e-5
ymax=1e1
set ylabel '$T(E)$' offset 1.0,0.0
set logscale y
set yrange [ymin:ymax]
set format y '\small $10^{%T}$'

set key samplen 1.5 spacing 1.5 top center reverse Left


E0=0.1
set origin 0.0,0.65
set size 1.0,0.32
set label at graph 0.03,0.1 '\small LCAO' 
p 'ET_lcao.bench' u 1:2 w l dt 1 lw 7 lc rgb "#CCCCCC" t '\small QTPYT',\
  'ET_lcao.dat'   u 1:2 w l dt 1 lw 3 lc rgb "#DEDE78" t '\small ASE'
unset label

set origin 0.0,0.325
set size 1.0,0.32
set label at graph 0.03,0.1 '\small LO-$p_z$' 
p 'ET_lcao.bench' u 1:2 w l dt 1 lw 7 lc rgb "#CCCCCC" notit,\
  'ET_pz.bench'   u 1:2 w l dt 1 lw 7 lc rgb "#D09AC6" t '\small QTPYT',\
  'ET_pz.dat'     u 1:2 w l dt 1 lw 3 lc rgb "#8B008B" t '\small ASE'
unset label

set origin 0.0,0.0
set size 1.0,0.32
set label at graph 0.03,0.1 '\small LO-$p_z\!+\!d$' 
p 'ET_lcao.bench'             u 1:2 w l dt 1 lw 7 lc rgb "#CCCCCC" notit,\
  'ET_pzd.bench'              u 1:2 w l dt 1 lw 7 lc rgb "#FF9999" t '\small QTPYT',\
  'ET_lcao.dat'               u 1:2 w l dt 1 lw 7 lc rgb "#DEDE78" t '\small $\Xi=0$',\
  'threshold0.001/ET_pzd.dat' u 1:2 w l dt 1 lw 2 lc rgb "#ACFA70" t '\small $\Xi=0.001$',\
  'threshold0.010/ET_pzd.dat' u 1:2 w l dt 2 lw 2 lc rgb "#68AE4C" t '\small $\Xi=0.010$',\
  'threshold0.050/ET_pzd.dat' u 1:2 w l dt 3 lw 2 lc rgb "#2E6828" t '\small $\Xi=0.050$',\
  'threshold0.075/ET_pzd.dat' u 1:2 w l dt 6 lw 2 lc rgb "#012801" t '\small $\Xi=0.075$',\
  'threshold0.100/ET_pzd.dat' u 1:2 w l dt 1 lw 2 lc rgb "#000000" t '\small $\Xi=0.100$'
unset label


unset multiplot


#    EOF
