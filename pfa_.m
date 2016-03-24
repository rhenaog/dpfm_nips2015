classdef pfa_ < handle
	properties( Access = public )
		% sizes
		M
		N
		N_tr
		N_te
		N_ho
		K
		C
		
		% data
		test_idx
		% hold_idx
		
		% parameters
		Phi
		Theta
		Z
		eta
		rk
		gamma
		pn
		Pi
		
		Psi
		
		% hyperparameters
		h_eta = struct( 'a', 8 )
		h_rk = struct( 'a', 32, 'b', 1 )
		h_gamma = struct( 'a', 1, 'b', 1 )
		h_pn = struct( 'a', 0.5 )
		h_pi = struct( 'a', 1 )
		
		% options
		do_Z = false
		do_rk = true
		do_pn = false
		share_theta = true
		
		do_test
		do_hold
		do_class
		
		sopts = struct( 'nsamples', 250, 'thin', 1, 'bnin', 250, 'verb', 50 )
		% ropts = struct( 'maxit', 1000, 'thin', 1, 'maxtime', 10000, 'verb', 25 )
	end
	properties( Access = private )
		% llik aux
		Phi_Theta
		count = 1
		M_tr
		M_te
		M_ho_o
		M_ho_e
		SX_tr
		SX_te
		SX_ho
	end
	methods( Access = public )
		function this = pfa_( X, y, Xh, varargin )
			%
			% constructor
			%
			
			[ this.M this.N ] = size( X );
			
			% check if a classifier is needed
			if ~isempty( y )
				this.do_class = true;
				this.C = size( y, 1 );
				
				if this.N ~= size( y, 2 )
					error( 'pfa :: sizes of X and y do not match' )
				end
			else
				this.do_class = false;
			end
			
			% check additional arguments first
			this.match_args( varargin{:} );
			
			% handle test set
			if ~isempty( this.test_idx )
				this.do_test = true;
				this.N_te = sum( this.test_idx );
				this.N_tr = this.N - this.N_te;
				if ~isempty( Xh )
					this.do_hold = true;
					this.N_ho = size( Xh, 2 );
					
					if this.N_te ~= this.N_ho
						error( 'pfa :: sizes of X_te and X_ho do not match' )
					end
				else
					this.do_hold = false;
					this.N_ho = 0;
				end
			else
				this.do_test = false;
				this.do_hold = false;
				this.N_te = 0;
				this.N_ho = 0;
				this.N_tr = this.N;
				this.test_idx = false( 1, this.N );
			end
			
			% init
			% this.init( X, double( y ), Xh );
			
			% build mex if not available
			
			% set external seed
			% mex_mult_rnd( 0 );
		end
		function init( this, X, ~, Xh )
			%
			%
			%
			
			this.Psi = sparse( double( X > 0 ) );
			
			this.Phi = rand( this.M, this.K );
			this.Theta = ( 1/this.K )*ones( this.K, this.N );
			
			this.eta = this.h_eta.a/this.K;
			this.rk = ( this.h_rk.a/this.K )*ones( this.K, 1 );
			this.gamma = this.h_gamma.a/this.h_gamma.b;
			this.pn = this.h_pn.a*ones( 1, this.N );
			this.Pi = ( this.h_pi.a/this.K )*ones( this.K, 1 );
			
			this.Phi_Theta = zeros( this.M, this.N );
			M_all = X > 0;
			this.M_tr = bsxfun( @and, M_all, ~this.test_idx );
			this.SX_tr = sum( X(this.M_tr) );
			if this.do_test
				this.M_te = bsxfun( @and, M_all, this.test_idx );
				this.SX_te = sum( X(this.M_te) );
			end
			if this.do_hold
				this.M_ho_o = Xh > 0;
				this.M_ho_e = [ false( this.M, this.N_tr ) this.M_ho_o ];
				this.SX_ho = sum( Xh(this.M_ho_o) );
			end
			
			if this.do_class
				% this.A = randn( this.M, this.K + this.bias );
				% this.A = ( y*y' + 1e-6*eye( this.M ) )\( ( y - 0.5 )*this.H' );
				% this.EAA = this.A.^2;
				% this.Sa = ones( this.M, this.K + this.bias );
			end
        end
		function [ llik ] = sstep( this, X, y, Xh, Pi, niter )
			%
			%
			%
			
			% loop it
			if nargin > 5
				for i=1:niter
					[ llik ] = this.sstep( X, y, Xh, Pi );
				end
				
				return
			end
			
			% training set
			tix = ~this.test_idx;
			
			this.Psi = mex_tpo( X, this.Phi, this.Theta );
			
			% resample latent counts
			[ x_mk x_kn ] = mex_mult_rnd( this.Psi, this.Phi, this.Theta, tix );
			
			if this.do_Z
				% resample Z
				this.resample_Z( x_kn, Pi );
				
				% resample Pi
				if isvector( Pi )
					Znz = sum( this.Z(:,tix), 2 );
					this.Pi = betarnd( this.h_pi.a/this.K + Znz, this.h_pi.a - this.h_pi.a/this.K + this.N_tr - Znz );
				end
				
				% resample pn
				if this.do_pn
					
				end
				
				% resample rk
				if this.do_rk
					Lk = mex_crt( x_kn(:,tix), this.rk );
					sumbpi = sum( bsxfun( @times, this.Z(:,tix), log( max( 1 - this.pn(tix), realmin ) ) ), 2 );
					this.rk = ( this.h_rk.b - sumbpi )./randg( this.gamma + Lk );
				end
				
				% resample Theta
				if this.share_theta
					this.Theta = bsxfun( @rdivide, this.rk.*sum( this.Z, 2 ) + sum( x_kn, 2 ), 1./this.pn );
				else
					this.Theta = bsxfun( @rdivide, randg( bsxfun( @times, this.rk, this.Z ) + x_kn ), 1./this.pn );
				end
			else
				% resample Theta
				if this.share_theta
					this.Theta = bsxfun( @rdivide, this.rk + sum( x_kn, 2 ), 1./this.pn );
				else
					this.Theta = bsxfun( @rdivide, randg( bsxfun( @plus, this.rk, x_kn ) ), 1./this.pn );
				end
			end
			
			% resample Phi
			this.Phi = randg( this.eta + x_mk );
			this.Phi = bsxfun( @rdivide, this.Phi, 0.5 + sum( this.Theta, 2 )' );
			
			% likelihood
			if nargout > 0
				this.Phi_Theta = this.Phi_Theta + this.Phi*this.Theta;
				this.count = this.count + 1;
				tmp = bsxfun( @rdivide, this.Phi_Theta/this.count, sum( this.Phi_Theta, 1 )/this.count );
				llik(1) = sum( X(this.M_tr).*log( tmp(this.M_tr) ) )/this.SX_tr;
				if this.do_test
					llik(2) = sum( X(this.M_te).*log( tmp(this.M_te) ) )/this.SX_te;
				end
				if this.do_hold
					llik(3) = sum( Xh(this.M_ho_o).*log( tmp(this.M_ho_e) ) )/this.SX_ho;
				end
				
				if this.do_class
					
				end
			end
		end
        function res = srun( this, X, y, Xh, varargin )
			%
			%
			%
			
			% set inference options
			this.match_struct_args( 'sopts', varargin{:} );
			
			% traces
			ns = floor( this.sopts.nsamples/this.sopts.thin );
			res.tr.llik = zeros( 1 + this.do_test + this.do_hold, ns );
			% res.tr.err_tr = zeros( 1, ns );
			if this.do_class
				% res.tr.lly_tr = zeros( 1, ns );
				% res.tr.lly_te = zeros( 1, ns );
				% res.tr.acc_tr = zeros( this.M, ns );
				% res.tr.acc_te = zeros( this.M, ns );
			end
			
			% tix = ~this.test_idx;
			
			% simmaries
			res.Phi = this.Phi;
			res.Theta = this.Theta;
			res.Z = zeros( this.K, this.N );
			res.rk = this.rk;
			
			% loop
			ss = 1;
			for s=-this.sopts.bnin:this.sopts.nsamples
				if s == 50 - this.sopts.bnin
					this.do_Z = true;
				end
				
				if s > 0
					llik = this.sstep( X, y, Xh, this.Pi );
				else
					this.sstep( X, y, Xh, this.Pi );
				end
				
				% traces/summaries
				if ( s > 0 ) && ( mod( s, this.sopts.thin ) == 0 )
					% traces
					res.tr.llik(:,ss) = llik; 
					
					% averages
					res.Phi = res.Phi + this.Phi;
					res.Theta = res.Theta + this.Theta;
					res.Z = res.Z + this.Z;
					res.rk = res.rk + this.rk;
					
					% classification accuracy
					if this.do_class
						%{
						ll = lly + pp - qq;
						res.tr.lly_tr(:,ss) = mean( ll(tix) );
						
						e = vauc( y, this );
						res.tr.acc_tr(:,ss) = e(:,1);
						if this.do_test
							res.tr.lly_te(:,ss) = mean( ll(~tix) );
							res.tr.acc_te(:,ss) = e(:,2);
						end
						%}
					end
					
					ss = ss + 1;
				end
					
				% verbosity
				if mod( s, this.sopts.verb ) == 0
					fprintf( 'it: %d\n', s )
				end
			end
			
			fprintf( 'lik: %g\n', res.tr.llik(:,end) )
			
			% collect results
			res.Phi = res.Phi/ns;
			res.Theta = res.Theta/ns;
			res.Z = res.Z/ns;
			res.rk = res.rk/ns;
		end
	end
	methods( Access = private )
		function match_args( this, varargin )
			%
			% matches arguments for constructor
			%
		
			if isempty( varargin )
				return
			end
			if mod( numel( varargin ), 2 ) ~= 0
				error( '%s :: inconsistent number of input parameters', class( this ) )
			end
		
			for i=1:numel( varargin )/2
				% if ismember( varargin{2*i - 1}, this.fields )
				if ~isempty( this.findprop( varargin{2*i - 1} ) )
					this.(varargin{2*i - 1}) = varargin{2*i};
				else
					error( '%s :: wrong argument name: %s\n', class( this ), varargin{2*i - 1} );
				end
			end
		end
		function match_struct_args( this, sname, varargin )
			%
			% mathces fields form an structure in the object
			%
		
			if isempty( varargin )
				return
			end
			if mod( numel( varargin ), 2 ) ~= 0
				error( '%s :: inconsistent number of input parameters', class( this ) )
			end
		
			for i=1:numel( varargin )/2
				if isfield( this.(sname), varargin{2*i - 1} )
					this.(sname).(varargin{2*i - 1}) = varargin{2*i};
				else
					error( '%s :: wrong argument name: %s\n', class( this ), varargin{2*i - 1} );
				end
			end
		end
		function resample_Z( this, x_kn, Pi )
			%
			%
			%
			
			lix = x_kn == 0;
			[ rix cix ] = find( x_kn == 0 );
			p1 = Pi(rix).*( ( 1 - this.pn(cix)' ).^this.rk(rix) );
			p0 = 1 - Pi(rix);
			this.Z = ones( size( x_kn ) );
			this.Z(lix) = ( p1./( p1 + p0 ) ) > rand( size( rix ) );
		end
	end
	methods( Access = private, Static )
		function [ x_mk x_kn ] = mult_rand( X, Phi, Theta )
			%
			%
			%
			
			function x = mrand( n, cp )
				%
				%
				%
				
				x = sum( bsxfun( @gt, rand( n, 1 )*cp(end), cp ), 2 ) + 1;
				x = sparse( x, 1, 1, numel( cp ), 1 );
			end
			
			M_ = size( X, 1 );
			[ K_ N_ ] = size( Theta );
			
			x_mk = zeros( M_, K_ );
			x_kn = zeros( K_, N_ );
			
			for n=1:N_
				inz = find( X(:,n) )';
				map = bsxfun( @times, Phi(inz,:), Theta(:,n)' ); % M x K
				map = cumsum( map, 2 );
				x_km = zeros( K_, numel( inz ) );
				for m=1:numel( inz )
					x_km(:,m) = x_km(:,m) + mrand( X(inz(m),n), map(m,:) );
				end
				
				x_kn(:,n) = sum( x_km, 2 );
				x_mk(inz,:) = x_mk(inz,:) + x_km';
			end
		end
		function m = tpo( lambda )
			%
			%
			%
			
			function m_ = tpo_one( lambda_ )
				if lambda_ < 1
					m_ = poissrnd( lambda_ ); u = rand();
					while u > 1/( m_ + 1 )
						m_ = poissrnd( lambda_ ); u = rand();
					end
					m_ = m_ + 1;
				else
					m_ = poissrnd( lambda_ );
					while m_ == 0
						m_ = poissrnd( lambda_ );
					end
				end
			end
			
			%m = zeros( size( lambda ) );
			%for n=1:numel( lambda )
			%	m(n) = tpo_one( lambda(n) );
			%end
			m = arrayfun( @tpo_one, lambda );
		end
	end
end
