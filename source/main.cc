#include <limits>
#include <vector>
#include <complex>
//#include "gnuplot_i.hpp"
#include "Array.h"
#include "fftw++.h"

// instal fftw and 
// Compile with:
// g++ -std=c++11 -I fftw++-1.13 -fopenmp main.cc fftw++-1.13/fftw++.cc -lfftw3 -lfftw3_omp; ./a.out

using namespace std;
using namespace Array;
using namespace fftwpp;

double u(double x) {
  return x*x;
}
int main()
{
  fftw::maxthreads=get_max_threads();
  const Complex ii(0.0,1.0);
  const double invhb = 1.0;
  const double invm = 1.0;
  
  unsigned int n=1400;
  const double dx=0.01;
  const double dt=0.000001;
  unsigned int np=n/2+1;
  size_t align=sizeof(Complex);
  
  array1<Complex> psix(n,align);
  array1<Complex> fs(n,align);
  array1<Complex> gfs(n,align);
  
  fft1d Forward(n,-1,psix,fs);
  fft1d Backward(n,1,fs,gfs);

  double x0 = 4;
  
  for(unsigned int i=0; i < n; i++) {
    double x = (i-0.5*n)*dx;
    x+=x0;
    psix[i]=exp(-pow(x,2));
  }

  ofstream ofs("psix.dat");
  for (size_t i = 0; i < n; ++i) {
    double x = (i-0.5*n)*dx;
    ofs << x << "  " << real(psix[i]) << "  " << imag(psix[i]) << "\n";
  }

  int tn=10;
  //// following http://hep.physics.indiana.edu/~hgevans/p411-p610/material/08_pde_iv/schrodinger.html
  array1<Complex> ax(n,align); ///> a(x)=e^(-i*u*dt/(2*hb))*s(x,t)
  array1<Complex> bp(n,align); ///> B(p) ≡ e^( −i p2 Δt / 2m) A(p)
  for (size_t l = 0; l < 100; ++l) {
    for (size_t j = 0; j < tn; ++j) {
      /// step 1
      for (size_t i = 0; i < n; ++i) {
        double x = (i-0.5*n)*dx;
        ax[i]=exp(-ii*u(x)*dt*0.5*invhb)*psix[i];///> a(x)=e^(-i*u*dt/(2*hb))*s(x,t)
      }
      /// step 2
      Forward.fft(ax,fs);
      /// step 3
      for (size_t i = 0; i < n; ++i) {
        //double p=i;
        double dw = 2*3.1415926/n;
        double p=i*dw;
        //double p=(i-n/2)*dw;
        bp[i]=exp(-ii*p*p*dt*0.5*invm)*fs[i];///> B(p) ≡ e^( −i p2 Δt / 2m) A(p)
      }
      /// step 4
      Backward.fftNormalized(bp,gfs);
      /// step 5
      for (size_t i = 0; i < n; ++i) {
        double x = (i-0.5*n)*dx;
        psix[i]=exp(-ii*u(x)*dt*0.5*invhb)*gfs[i];///>psi(x, t+Δt)  =  e^(−i U Δt / 2ℏ) B(x)
      }
    }
    vector<double> xv,psir;
    for (size_t i = 0; i < n; ++i) {
      xv.push_back((i-0.5*n)*dx);
      psir.push_back(real(psix[i]));
    }
    //Gnuplot g("lines");
    //g.plot_xy(xv,psir);
    //std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );
  }
  
  ofstream offs("fs.dat");
  for (size_t i = 0; i < n; ++i) {
    double p=i*dx;
    offs << p << "  " << real(fs[i]) << "  " << imag(fs[i]) << "\n";
  }

  ofstream ofgfs("psixt.dat");
  for (size_t i = 0; i < n; ++i) {
    double x = (i-0.5*n)*dx;
    ofgfs << x << "  " << real(psix[i]) << "  " << imag(psix[i]) << "\n";
  }
}
