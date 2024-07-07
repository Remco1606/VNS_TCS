: $Id: xtra.mod,v 1.4 2014/08/18 23:15:25 ted Exp ted $
: 2018/05/20 Modified by Aman Aberra 

NEURON {
	SUFFIX xtra
	RANGE rx1, rx2, rx3, rx4, rx5, rx6: (es = max amplitude of the potential)		
	RANGE x, y, z, type, order
	GLOBAL is1, is2, is3, is4, is5, is6: (stim = active electrode, amp = stimulation amplitude)
	POINTER ex 
}

PARAMETER {	
	rx1 = 0 (ohm) : mV/mA
	rx2 = 0 (ohm) : mV/mA
	rx3 = 0 (ohm) : mV/mA
	rx4 = 0 (ohm) : mV/mA
	rx5 = 0 (ohm) : mV/mA
	rx6 = 0 (ohm) : mV/mA
	x = 0 (1) : spatial coords
	y = 0 (1)
	z = 0 (1)		
	type = 0 (1) : numbering system for morphological category of section - unassigned is 0
	order = 0 (1) : order of branch/collateral. 
}

ASSIGNED {
	v (millivolts)
	ex (millivolts)
	is1 (miliamps)
	is2 (miliamps)
	is3 (miliamps)
	is4 (miliamps)
	is5 (miliamps)
	is6 (miliamps)
	area (micron2)
}

INITIAL {
	ex = is1*rx1+is2*rx2+is3*rx3+is4*rx4+is5*rx5+is6*rx6
}


BEFORE BREAKPOINT { : before each cy' = f(y,t) setup
  	ex = is1*rx1+is2*rx2+is3*rx3+is4*rx4+is5*rx5+is6*rx6
}

