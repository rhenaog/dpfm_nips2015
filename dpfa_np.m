classdef dpfa_np < handle
	properties( Access = public )
		% sizes
		M
		N
		N_tr
		N_te
		N_ho
		K
		C
		L
		
		% data
		test_idx
		
		% options
		do_test
		do_hold
		do_class
		
		iopts = struct( 'nsamples', 100, 'thin', 1, 'bnin', 100, 'verb', 50 )
		sopts = struct( 'nsamples', 250, 'thin', 1, 'bnin', 250, 'verb', 50 )
		
		% containers
		bl
	end
	properties( Access = private )
		
	end
	methods( Access = public )
		function this = dpfa_np( X, y, Xh, varargin )
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
			
			this.L = numel( this.K );
			
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
			
			% build mex if not available
			
			% set external seed
			mex_tpo( 0 );
			mex_mult_rnd( 0 );
			mex_crt( 0 );
		end
		function set_base( this, X, y, Xh, varargin )
			%
			%
			%
			
			% create base layer
			this.bl{1} = pfa_np( X, y, Xh, 'K', this.K(1), 'test_idx', this.test_idx );
			
			% set parameters
			% this.match_obj_args( 'bl', varargin{:} );
		end
		function res = init( this, X, y, Xh, varargin )
			%
			%
			%
			
			% set initialization options
			this.match_struct_args( 'iopts', varargin{:} );
			
			if isscalar( this.iopts.bnin ) && isvector( this.K )
				this.iopts.bnin = this.iopts.bnin*ones( 1, numel( this.K ) );
				this.iopts.nsamples = this.iopts.nsamples*ones( 1, numel( this.K ) );
			end
			
			tic();
			
			% initialize base layer
			this.bl{1}.init( X, y, Xh );
			res.bl{1} = this.bl{1}.srun( X, y, Xh, 'bnin', this.iopts.bnin(1), 'nsamples', this.iopts.nsamples(1) );
			
			% initialize sbn layers
			for k=2:this.L
				Z = sparse( this.bl{k-1}.Z );
				
				this.bl{k} = pfa_( Z, [], [], 'K', this.K(k), 'test_idx', this.test_idx, 'share_theta', false );
				this.bl{k}.init( Z, [], [] );
				
				res.bl{k} = this.bl{k}.srun( Z, [], [], 'bnin', this.iopts.bnin(1), 'nsamples', this.iopts.nsamples(1) );
			end
			
			this.bl{1}.reset_llik();
			
			if this.do_class
				% [ acc_tr acc_te ] = this.get_acc();
				% fprintf( 'acc tr: %s\n', sprintf( '%1.3f ', acc_tr ) )
				% fprintf( 'acc te: %s\n', sprintf( '%1.3f ', acc_te ) )
				% [ auc_tr auc_te ] = this.get_auc();
				% fprintf( 'auc tr: %s\n', sprintf( '%1.3f ', auc_tr ) )
				% fprintf( 'auc te: %s\n', sprintf( '%1.3f ', auc_te ) )
			end
			
			res.time = toc();
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
			
			% simmaries
			[ res.Phi res.Theta res.Z res.rk ] = deal( cell( this.L, 1 ) );
			for k=1:this.L
				res.Phi{k} = this.bl{k}.Phi;
				res.Theta{k} = this.bl{k}.Theta;
				res.Z{k} = zeros( this.K(k), this.N );
				res.rk{k} = this.bl{k}.rk;
			end
			
			tic();
			
			% loop
			ss = 1;
			for s=-this.sopts.bnin:this.sopts.nsamples
				% base layer
				Pi = 1 - exp( -this.bl{2}.Phi*this.bl{2}.Theta );
				if s > 0
					llik = this.bl{1}.sstep( X, y, Xh, Pi );
				else
					this.bl{1}.sstep( X, y, Xh, Pi );
				end
				
				% other layers
				for k=2:this.L
					Z = sparse( this.bl{k-1}.Z );
					if k == this.L
						Pi = this.bl{k}.Pi;
					else
						Pi = 1 - exp( -this.bl{k+1}.Psi );
					end
					this.bl{k}.sstep( Z, [], [], Pi );
				end
				
				% traces/summaries
				if ( s > 0 ) && ( mod( s, this.sopts.thin ) == 0 )
					% traces
					res.tr.llik(:,ss) = llik; 
					
					% averages
					for k=1:this.L
						res.Phi{k} = res.Phi{k} + this.bl{k}.Phi;
						res.Theta{k} = res.Theta{k} + this.bl{k}.Theta;
						res.Z{k} = res.Z{k} + this.bl{k}.Z;
						res.rk{k} = res.rk{k} + this.bl{k}.rk;
					end
					
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
			
			res.time = toc();
			
			for k=1:this.L
				res.Phi{k} = res.Phi{k}/ns;
				res.Theta{k} = res.Theta{k}/ns;
				res.Z{k} = res.Z{k}/ns;
				res.rk{k} = res.rk{k}/ns;
			end
			
			% fprintf( 'lik: %g lik: %g\n', res.tr.lik_tr(end), res.tr.lik_tr(end) )
			% if this.do_test
			% 	fprintf( 'lik: %g lik: %g\n', res.tr.lik_te(end), res.tr.lik_te(end) )
			% end
			
			if this.do_class
				% [ acc_tr acc_te ] = this.get_acc();
				% fprintf( 'err tr: %s\n', sprintf( '%1.3f ', acc_tr ) )
				% fprintf( 'err te: %s\n', sprintf( '%1.3f ', acc_te ) )
				% [ auc_tr auc_te ] = this.get_auc();
				% fprintf( 'auc tr: %s\n', sprintf( '%1.3f ', auc_tr ) )
				% fprintf( 'auc te: %s\n', sprintf( '%1.3f ', auc_te ) )
			end
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
		function match_obj_args( this, sname, varargin )
			%
			% mathces fields form an object in the object
			%
		
			if isempty( varargin )
				return
			end
			if mod( numel( varargin ), 2 ) ~= 0
				error( '%s :: inconsistent number of input parameters', class( this ) )
			end
		
			for i=1:numel( varargin )/2
				if isprop( this.(sname), varargin{2*i - 1} )
					this.(sname).(varargin{2*i - 1}) = varargin{2*i};
				else
					error( '%s :: wrong argument name: %s\n', class( this ), varargin{2*i - 1} );
				end
			end
		end
	end
	methods( Access = private, Static )
		
	end
end
