function [ vv pp html ] = print_topics( fname, wb, Phi, thr, do_sort, tn )
%
%
%

if do_sort
	cs = sort( Phi, 'descend' );
	[ ~, ix ] = sort( sum( cs(1:thr,:) ), 'descend' );
else
	ix = 1:size( Phi, 2 );
end

vv = cell( numel( ix ), 1 );
pp = zeros( numel( ix ), 1 );
for i=1:numel( ix );
	[ vv{i} pp(i) ] = get_top_words( wb, Phi, ix(i), thr );
	
	% if nargout == 0
	% 	fprintf( 'id: %d, %s\n', ix(i), str )
	% end
end

if nargin > 5
	for i=1:3
		ix = ( i - 1 )*10 + ( 1:10 );
		fname_ = sprintf( '%s_%d.csv', fname, i );
		writetable( cell2table( [ vv{ix} ], 'VariableNames', strcat( 'T', tn(ix) ) ), fname_ );
	end
else
	% show table
	nc = [ 10 13 ];
	tb = cell( nc );
	k = 1;
	for i=1:nc(1)
		for j=1:nc(2)
			if k < numel( ix ) + 1
				str = sprintf( '<br>%s', vv{k}{:} );
				tb{i,j} = sprintf( '<strong>T%d (%1.2f)</strong>%s', k, pp(k), str );
			end
			k = k + 1;
		end
	end
	
	title = sprintf( 'Top words per topic, in parenthesis is the probability mass of the top %d words on each topic.', thr );
	html = GTHTMLtable( title, tb, 'show' );
	
	% save table
	if ~isempty( fname )
		fid = fopen( fname, 'w' );
		fprintf( fid, '%s', html );
		fclose( fid );
	end
end

end

function [ vv pp str ] = get_top_words( wb, Phi, id, thr )
%
%
%

[ ~, ix ] = sort( Phi(:,id), 'descend' );

pp = sum( Phi(ix(1:thr),id) );
vv = wb(ix(1:thr));

str = sprintf( 'p: %0.2f, v: %s', pp, sprintf( '%s ', vv{:} ) );

if nargout == 0
	fprintf( '%s\n', str );
end

end
