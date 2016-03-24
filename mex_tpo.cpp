//
// make: mexsh -largeArrayDims -I/Users/rhenao/Work/ext/eigen-eigen-ffa86ffb5570 mex_tpo.cpp
//
#include "mex.h"
#include <random>
#include <Eigen/Dense>

#define EIGEN_DONT_PARALLELIZE

using namespace Eigen;

// random stuff
std::mt19937 Engine;
std::uniform_real_distribution<double> randu( 0.0, 1.0 );

double tpo_one( const double lambda )
{
	double m, u;
	
	std::poisson_distribution<> randp( lambda );
	
	if( lambda < 1 ){
		m = randp( Engine );
		u = randu( Engine );
		while( u > 1/( m + 1 ) ){
			m = randp( Engine );
			u = randu( Engine );
		}
		m++;
	}
	else{
		m = randp( Engine );
		while( m < 0 )
			m = randp( Engine );
	}
	
	return m;
}

/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// sizes
	mwSize M = mxGetM( prhs[0] );
	
	// set seed
	if( ( nrhs == 1 ) && ( M == 1 ) ){
	// if( ( nrhs == 1 ) && mxIsScalar( prhs[0] ) ){
		Engine.seed( (int) mxGetScalar( prhs[0] ) );
		// randu.reset( (int) mxGetScalar( prhs[0] ) );
		return;
	}
	
	// in
	Map<MatrixXd> x( mxGetPr( prhs[0] ), M, 1 );
	// out
	plhs[0] = mxCreateDoubleMatrix( M, 1, mxREAL );
	Map<MatrixXd> z( mxGetPr( plhs[0] ), M, 1 );
	
	// #pragma omp parallel for
	for( mwIndex i = 0; i < M; i++ ){
		z(i) = tpo_one( x(i) );
	}
}
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *pr, *pr_out;
	mwIndex i, j, k, ix, *jc, *ir, *jc_out, *ir_out;
	mwIndex out;
	
	// set seed
	if( nrhs == 1 ){
		Engine.seed( (int) mxGetScalar( prhs[0] ) );
		// randu.reset( (int) mxGetScalar( prhs[0] ) );
		return;
	}
	
	// sizes
	mwSize M = mxGetM( prhs[0] );
	mwSize N = mxGetN( prhs[0] );
	mwSize K = mxGetN( prhs[1] );
	// data
	pr =  mxGetPr( prhs[0] );
	jc = mxGetJc( prhs[0] );
	ir = mxGetIr( prhs[0] );
	// loadings and scores
	Map<MatrixXd> Phi( mxGetPr( prhs[1] ), M, K );
	Map<MatrixXd> Theta( mxGetPr( prhs[2] ), K, N );
	
	// output counts
	plhs[0] = mxCreateSparse( M, N, mxGetNzmax( prhs[0] ), mxREAL );
	pr_out = mxGetPr( plhs[0] );
	jc_out = mxGetJc( plhs[0] );
	ir_out = mxGetIr( plhs[0] );
	
	ix = 0;
	double tmp;
	jc_out[0] = jc[0];
	for( i = 0; i < N; i++ ){
		jc_out[i + 1] = jc[i + 1];
		for( j = 0; j < ( jc[i + 1] - jc[i] ); j++ ){
			tmp = Phi.row( ir[ix] )*Theta.col( i );
			
			pr_out[ix] = tpo_one( tmp );
			ir_out[ix] = ir[ix];
			
			ix++;
		}
	}
}
