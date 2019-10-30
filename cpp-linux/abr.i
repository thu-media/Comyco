/********************************************************
 * Swig module description file for wrapping a C++ class.
 * Generate by saying "swig -python -shadow number.i".   
 * The C module is generated in file number_wrap.c; here,
 * module 'number' refers to the number.py shadow class.
 ********************************************************/
%module envcpp
%include "std_vector.i"
%include "std_string.i"

namespace std {
   %template(vectori) vector<int>;
   %template(vectord) vector<double>;
   %template(vectors) vector<string>;
};

// This tells SWIG to treat an double * argument with name 'OutDouble' as
// an output value.  

%typemap(argout) double *OUTPUT {
	$result = sv_newmortal();
	sv_setnv($result, *$input);
	argvi++;                     /* Increment return count -- important! */
}

// We don't care what the input value is. Ignore, but set to a temporary variable

%typemap(in,numinputs=0) double *OUTPUT(double junk) {
	$1 = &junk;
}

%{
#include "env.hh"
%}

%include "env.hh"