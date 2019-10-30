swig -c++ -python abr.i
g++ abr_wrap.cxx env.cpp -fPIC -shared -I /usr/include/python2.7/ -L /usr/lib/python2.7 -o _envcpp.so -O4
