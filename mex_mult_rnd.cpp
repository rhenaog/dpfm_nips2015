//
// make: mexsh -largeArrayDims -I/Users/rhenao/Work/ext/eigen-eigen-ffa86ffb5570 mex_mult_rnd.cpp
//
#include "mex.h"
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

// random stuff
std::mt19937 Engine;
std::uniform_real_distribution<double> randu( 0.0, 1.0 );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// double *pr;
	mwIndex i, j, k;//ix , *jc, *ir;
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
	double *pr =  mxGetPr( prhs[0] );
	mwIndex *jc = mxGetJc( prhs[0] );
	mwIndex *ir = mxGetIr( prhs[0] );
	// loadings and scores
	Map<MatrixXd> Phi( mxGetPr( prhs[1] ), M, K );
	Map<MatrixXd> Theta( mxGetPr( prhs[2] ), K, N );
	mxLogical *tix = mxGetLogicals( prhs[3] );
	
	// output counts
	plhs[0] = mxCreateDoubleMatrix( M, K, mxREAL );
	Map<MatrixXd> x_mk( mxGetPr( plhs[0] ), M, K );
	plhs[1] = mxCreateDoubleMatrix( K, N, mxREAL );
	Map<MatrixXd> x_kn( mxGetPr( plhs[1] ), K, N );
	
	mwIndex ix = 0;
	ArrayXd tmp;
	for( i = 0; i < N; i++ ){
		for( j = 0; j < ( jc[i + 1] - jc[i] ); j++ ){
			tmp = Phi.row( ir[ix] ).array()*Theta.col( i ).transpose().array();
			std::partial_sum( tmp.data(), tmp.data() + K, tmp.data() );
			for( k = 0; k < pr[ix]; k++ ){
				out = ( tmp < randu( Engine )*tmp(K - 1) ).count();
				x_mk(ir[ix],out) += tix[i];
				x_kn(out,i)++;
			}
			ix++;
		}
	}
}

/*
std::mt19937 Engine;
std::uniform_real_distribution<> randu( 0.0, 1.0 );

plhs[0] = mxCreateDoubleMatrix( 1000000, 1, mxREAL );
out = mxGetPr( plhs[0] );

ArrayXd map_tmp;
for( int i = 0; i < 1000000; i++ ){
	map_tmp = map;
	std::partial_sum( map_tmp.data(), map_tmp.data() + K, map_tmp.data() );
	out[i] = ( map_tmp < randu( Engine )*map_tmp(K - 1) ).count() + 1;
}
*/

/*
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *out, *par;
	double *map;
	
	map = mxGetPr( prhs[0] );
	std::vector<double> map_( map, map + 32 );
	
	std::mt19937 Engine;
	std::discrete_distribution<> dist( map_.begin(), map_.end() );
	
	plhs[0] = mxCreateDoubleMatrix( 1000000, 1, mxREAL );
	out = mxGetPr( plhs[0] );
	
	//map_[0] = 45;
	//map_[1] = 15;
	
	std::discrete_distribution<>::param_type pp;
	
	for( int i = 0; i < 1000000; i++ ){
		pp = { map_.begin(), map_.end() };
		dist.param( pp );
		out[i] = dist(Engine);
	}
	
}
*/