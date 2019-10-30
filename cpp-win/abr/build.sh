swig -c++ -python abr.i
g++ abr_wrap.cxx env.cpp -fPIC -shared -I /usr/include/python2.7/ -L /usr/lib/python2.7 -o ../x64/Release/_envcpp.so -O4
cp envcpp.py ../x64/Release/