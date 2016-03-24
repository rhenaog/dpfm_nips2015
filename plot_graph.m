function plot_graph( res, vocabulary, node_size, fs )
%
%
%

if nargin < 4
	fs = 11;
end

thr = 0.2;
% L1_size = sum( cumsum( sort( res.Phi{1}, 'descend') ) < thr );

[ ~, iz ] = sort( res.Phi{1}, 'descend' ); 

Phi2 = bsxfun( @rdivide, res.Phi{2}, sum( res.Phi{2} ) );
% L2_size = sum( cumsum( sort( Phi2, 'descend') ) < thr );

[ ~, IX ] = sort( Phi2, 'descend' );
for i=1:size( Phi2, 2 )
	Phi2(IX(4+1:end,i),i) = 0;
end

ix = find( any( Phi2, 2 ) );
Phi2 = Phi2(ix,:);

L1 = arrayfun( @(x) sprintf( 'T%d', x ), 1:size( Phi2, 1 ), 'uniformoutput', false )';
L2 = arrayfun( @(x) sprintf( 'M%d', x ), 1:size( Phi2, 2 ), 'uniformoutput', false )';
L = [ L1; L2 ];

A = [ zeros( size( Phi2, 1 ) ) Phi2; zeros( size( Phi2, 2 ), size( Phi2, 1 ) + size( Phi2, 2 ) ) ] > 0;

bgo = biograph( A, L, 'ShowArrows', 'off', 'LayoutType', 'radial', 'NodeAutoSize', 'off' );
nha = bgo.Nodes;
nL1 = nha(1:size( Phi2, 1 ));
nL2 = nha(size( Phi2, 1 )+1:end);

set( nL1, 'color', [ 0.9 0.9 0.9 ], 'linecolor', [ 0 0 0 ], 'size', node_size, 'fontsize', fs )
set( nL2, 'Shape', 'circle', 'color', [ 12 147 209 ]/255, 'size', [ 12 12 ] )

for i=1:numel( nL1 )
	str = sprintf( '%s\n', vocabulary{iz(1:4,ix(i))} );
	nL1(i).Label = str(1:end-1);
end
bgo.ShowTextInNodes = 'Label';

fo = biograph.bggui( bgo );
fh = get( fo.biograph.hgAxes, 'Parent' );
set( fh, 'HandleVisibility', 'on' )

end
