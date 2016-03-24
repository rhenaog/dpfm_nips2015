function exp_dpfa_gibbs_reuters()
%
%
%

close all
clc

rng( 0 )

% load data
rd = load( './data/rcv2_data.mat' );

% subset
X = rd.wordsTrain;
% y = logical( rd.labelsTrain' );
Xt = rd.wordsHeldout;
Xh = rd.wordsTest;
% yt = logical( rd.labelsTest' );
vocabulary = rd.vocabulary; %#ok
clear rd

X = X(:,1:100000);

test_idx = [ false( 1, size( X, 2 ) ) true( 1, size( Xt, 2 ) ) ];

% 2 layers
oo = dpfa_np( [ X Xt ], [], Xh, 'K', [ 128 64 ], 'test_idx', test_idx );
oo.set_base( [ X Xt ], [], Xh );
res_init = oo.init( [ X Xt ], [], Xh, 'bnin', 150, 'nsamples', 150 ); %#ok
res_out = oo.srun( [ X Xt ], [], Xh, 'bnin', 300, 'nsamples', 300 );

% show topics
print_topics( './results/res_reuters_topics.html', vocabulary, res_out.Phi{1}, 10, 0 );

% plot graph
plot_graph( res_out, vocabulary, [ 65 50 ] )
fsave( 'res_reuters_graph', './tex/images/', 'eps' )

end
