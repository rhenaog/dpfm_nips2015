function plot_topic( res, vocabulary, iy, width, base )
%
%
%

if nargin < 3
	for i=1:size( res.Phi{2}, 2 )
		plot_topic( res, vocabulary, i )
	end
	
	return
end

[ ~, iz ] = sort( res.Phi{1}, 'descend' ); 

phi = res.Phi{2}(:,iy);

[ ~, ix ] = sort( phi, 'descend' );

fg = figure; fg.Position(3:4) = [ width 300 ];
hold on
stem( ix(6:end), phi(ix(6:end)), 'color', [ 0.7 0.7 0.7 ], 'linewidth', 2, 'marker', 'none' );
stem( ix(1:5), phi(ix(1:5)), 'color', 'k', 'linewidth', 2, 'marker', 'none' );
for i=1:5
	
	% str = sprintf( '%s\\n%s', vocabulary{iz(1,ix(i))}, vocabulary{iz(2,ix(i))} );
	text( ix(i), phi(ix(i)), vocabulary(iz(1:5,ix(i))), 'HorizontalAlignment', 'center', ...
		'fontsize', 10, 'VerticalAlignment', 'bottom' )
end
yl = ylim();
box on
ylim( [ yl(1) yl(2)*1.25 ] ), xlim( [ 1 numel( phi ) ] )
set( gca, 'fontsize', 14 )
title( sprintf( 'M%d', iy ) )
ylabel( 'prob' )
xlabel( 'topic' )
hold off
drawnow

if nargin > 4
	fsave( sprintf( '%s_meta%d', base, iy ), './tex/images/', 'eps' )
end

end