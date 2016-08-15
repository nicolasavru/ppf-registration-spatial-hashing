% Compute Normals for a PLY file.
function compute_normals(input_file, output_file, trans_adj)
    [tri, pts, data, ~] = ply_read(input_file, 'tri');
    TR = triangulation(tri.', pts.');
    vn = vertexNormal(TR);
    data.vertex.nx = vn(:,1);
    data.vertex.ny = vn(:,2);
    data.vertex.nz = vn(:,3);
    data = rmfield(data, 'face');

    data.vertex.x = data.vertex.x + trans_adj(1,1);
    data.vertex.y = data.vertex.y + trans_adj(2,1);
    data.vertex.z = data.vertex.z + trans_adj(3,1);

    ply_write(data, output_file, 'ascii');

    [ fid, Msg ] = fopen ( strcat(output_file, '.trans_adj'), 'wt' );
    if ( fid == -1 )
        error(Msg);
    end
    fprintf(fid, '%f %f %f\n', trans_adj(1,1), trans_adj(2,1), trans_adj(3,1));
    fclose(fid);

end
