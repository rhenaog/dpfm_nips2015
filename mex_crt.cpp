//
// make: mexsh -largeArrayDims -I/Users/rhenao/Work/ext/eigen-eigen-ffa86ffb5570 mex_crt.cpp
//
#include "mex.h"
#include <random>
// #include <Eigen/Dense>

// using namespace Eigen;

// random stuff
std::mt19937 Engine;
std::uniform_real_distribution<double> randu( 0.0, 1.0 );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwIndex k, n, i;
	
	// set seed
	if( nrhs == 1 ){
		Engine.seed( (int) mxGetScalar( prhs[0] ) );
		// randu.reset( (int) mxGetScalar( prhs[0] ) );
		return;
	}
	
	// sizes
	mwSize K = mxGetM( prhs[0] );
	mwSize N = mxGetN( prhs[0] );
	// data
	double *x = mxGetPr( prhs[0] );
	double *r = mxGetPr( prhs[1] );
	// output
	plhs[0] = mxCreateDoubleMatrix( K, 1, mxREAL );
	double *L = mxGetPr( plhs[0] );
	
	for( k = 0; k < K; k++ )
		for( n = 0; n < N; n++ )
			for( i = 0; i < x[k + n*K]; i++ )
				L[k] += (double)( randu( Engine ) <= r[k]/( r[k] + i ) );
}