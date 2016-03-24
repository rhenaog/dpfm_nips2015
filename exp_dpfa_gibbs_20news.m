function exp_dpfa_gibbs_20news()
%
%
%

close all
clc

rng( 0 )

% load data
rd = load( './data/20news_data' );

% subset
ix = ismember( rd.labelsTrain, 1:20 );
X = rd.wordsTrain(:,ix);
% y = sparse( double( nominal( rd.labelsTrain(ix) ) ), 1:sum( ix ), true );
ix = ismember( rd.labelsTest, 1:20 );
Xt = rd.wordsHeldout(:,ix);
Xh = rd.wordsTest(:,ix);
% yt = sparse( double( nominal( rd.labelsTest(ix) ) ), 1:sum( ix ), true );

test_idx = [ false( 1, size( X, 2 ) ) true( 1, size( Xt, 2 ) ) ];% false( 1, size( Xh, 2 ) ) ];
% hold_idx = [ false( 1, size( X, 2 ) ) false( 1, size( Xt, 2 ) ) true( 1, size( Xh, 2 ) ) ];

oo = dpfa_np( [ X Xt ], [], Xh, 'K', [ 128 64 ], 'test_idx', test_idx );
oo.set_base( [ X Xt ], [], Xh );
res_init = oo.init( [ X Xt ], [], Xh, 'bnin', 100, 'nsamples', 100 ); %#ok
res_out = oo.srun( [ X Xt ], [], Xh, 'bnin', 600, 'nsamples', 600 );

% plot graph
plot_graph( res_out, rd.vocabulary, [ 65 50 ] )
fsave( 'res_20news_graph', './tex/images/', 'eps' )

% plot topics
plot_topic( res_out, rd.vocabulary, 13, 600, 'res_20news' )
plot_topic( res_out, rd.vocabulary, 51, 1000, 'res_20news' )
plot_topic( res_out, rd.vocabulary, 11, 1000, 'res_20news' )
plot_topic( res_out, rd.vocabulary, 28, 1000, 'res_20news' )

print_topics( './results/res_20news_topics.html', rd.vocabulary, res_out.Phi{1}, 10, 0 );

end
