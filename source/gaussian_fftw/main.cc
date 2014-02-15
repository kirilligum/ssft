#include <vector>
#include <complex>
#include "Array.h"
#include "fftw++.h"

// instal fftw and 
// Compile with:
// g++ -std=c++11 -I fftw++-1.13 -fopenmp main.cc fftw++-1.13/fftw++.cc -lfftw3 -lfftw3_omp; ./a.out

using namespace std;
using namespace Array;
using namespace fftwpp;

int main()
{
  fftw::maxthreads=get_max_threads();
  
  unsigned int n=700;
  const double dx=0.01;
  //unsigned int np=n;
  unsigned int np=n/2+1;
  size_t align=sizeof(Complex);
  
  array1<double> s(n,align);
  array1<Complex> fs(np,align);
  array1<double> gfs(n,align);
  
  rcfft1d Forward(n,s,fs);
  crfft1d Backward(n,fs,gfs);
  
  for(unsigned int i=0; i < n; i++) {
    double x = i-0.5*n;
    s[i]=exp(-pow(x*dx,2));;
  }

  ofstream ofs("s.dat");
  for (size_t i = 0; i < n; ++i) {
    ofs << i*dt << "  " << s[i] << "\n";
  }
	
  Forward.fft(s,fs);

  ofstream offs("fs.dat");
  for (size_t i = 0; i < np; ++i) {
    offs << i*dt << "  " << real(fs[i]) << "  " << imag(fs[i]) << "\n";
  }
	
  
  //cout << fs << endl;
  
  Backward.fftNormalized(fs,gfs);
  
  ofstream ofgfs("gfs.dat");
  for (size_t i = 0; i < n; ++i) {
    ofgfs << i*dt << "  " << gfs[i] << "\n";
  }
	
  //cout << s << endl;
}
