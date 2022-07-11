import os
import numpy as np
import trimesh

import torch
import torch.nn.functional as F

import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds

from .map_utils import load_model, point_mesh_face_distance, barycentric_to_points, points_to_barycentric


class SurfaceAlignedConverter:

    def __init__(self):
        obj_path = 'zju_smpl/smplx/smpl/smpl_uv.obj'
        verts, faces, faces_idx, verts_uvs, faces_t, device = load_model(obj_path)
        self.device = device
        self.verts = verts
        self.faces_idx = faces_idx
        self.verts_uvs = verts_uvs
        self.faces_t = faces_t
        self.xyzc = verts
        self.load_mesh_topology(verts, faces_idx)

    def load_mesh_topology(self, verts, faces_idx, cache_path='zju_smpl/cache'):

        if not os.path.exists(os.path.join(cache_path, 'faces_to_corres_edges.npy')):
            print('==> Computing mesh topology... ', cache_path)
            faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces = self._parse_mesh(verts, faces_idx)
            # save cache
            os.makedirs(cache_path)
            np.save(os.path.join(cache_path, 'faces_to_corres_edges.npy'), faces_to_corres_edges.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'edges_to_corres_faces.npy'), edges_to_corres_faces.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, 'verts_to_corres_faces.npy'), verts_to_corres_faces.to('cpu').detach().numpy().copy())
            print('==> Finished! Cache saved to: ', cache_path)

        else:
            print('==> Find pre-computed mesh topology! Loading cache from: ', cache_path)
            faces_to_corres_edges = torch.from_numpy(np.load(os.path.join(cache_path, 'faces_to_corres_edges.npy')))
            edges_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, 'edges_to_corres_faces.npy')))
            verts_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, 'verts_to_corres_faces.npy')))

        self.faces_to_corres_edges = faces_to_corres_edges.long().to(self.device)  # [13776, 3]
        self.edges_to_corres_faces = edges_to_corres_faces.long().to(self.device)  # [20664, 2]
        self.verts_to_corres_faces = verts_to_corres_faces.long().to(self.device)  # [6890, 9]

    def xyz_to_xyzch(self, points, verts):

        h, barycentric, idx, local_coordinates, _, _ = self.projection(points, verts)
        xyzc_triangles = self.xyzc[self.faces_idx].repeat(len(verts), 1, 1)[idx.view(-1)]  # [batch*65536, 3, 3]
        xyzc = barycentric_to_points(xyzc_triangles, barycentric.view(-1, 3))
        xyzch = torch.cat((xyzc, h.view(-1, 1)), dim=-1)
        xyzch = xyzch.view(len(verts), -1, 4)

        return xyzch, local_coordinates

    def projection(self, points, verts, points_inside_mesh_approx=True, scaling_factor=50):

        # STEP 0: preparation
        faces = self.faces_idx[None, ...].repeat(len(verts), 1, 1)
        meshes = Meshes(verts=verts*scaling_factor, faces=faces)
        pcls = Pointclouds(points=points*scaling_factor)
        # compute nearest faces
        _, idx = point_mesh_face_distance(meshes, pcls)

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        triangles = triangles_meshes[idx]  # [batch*65536, 3, 3]

        # STEP 1: Compute the nearest point on the mesh surface
        nearest, stats = self._parse_nearest_projection(triangles, pcls.points_packed())

        if points_inside_mesh_approx:
            sign_tensor = self._calculate_points_inside_meshes_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])
        else:
            sign_tensor = self._calculate_points_inside_meshes(pcls.points_packed(), meshes.verts_packed())

        # STEP 2-6: Compute the final projection point (check self._revise_nearest() for details)
        dist = torch.norm(pcls.points_packed() - nearest, dim=1)
        nearest_new, dist, idx = self._revise_nearest(pcls.points_packed(), idx, meshes, sign_tensor, nearest, dist, stats)
        h = dist * sign_tensor

        triangles = triangles_meshes[idx]
        barycentric = points_to_barycentric(triangles, nearest_new)

        # bad case
        barycentric = torch.clamp(barycentric, min=0.)
        barycentric = barycentric / (torch.sum(barycentric, dim=1, keepdim=True) + 1e-12)

        # local_coordinates
        local_coordinates_meshes = self._calculate_local_coordinates_meshes(meshes.faces_normals_packed(), triangles_meshes)
        local_coordinates = local_coordinates_meshes[idx]

        h = h.view(len(verts), -1, 1)
        barycentric = barycentric.view(len(verts), -1, 3)
        idx = idx.view(len(verts), -1)

        # revert scaling
        h = h / scaling_factor
        nearest_new = nearest_new / scaling_factor
        nearest = nearest / scaling_factor

        return h, barycentric, idx, local_coordinates, nearest_new, nearest

    # precise inside/outside computation based on ray-tracing
    def _calculate_points_inside_meshes(self, points, verts):

        verts_trimesh = verts.to('cpu').detach().numpy().copy()
        faces_trimesh = self.faces_idx.squeeze(0).to('cpu').detach().numpy().copy()
        mesh = trimesh.Trimesh(vertices=verts_trimesh, faces=faces_trimesh, process=False)
        trimesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        points_trimesh = points.to('cpu').detach().numpy().copy()

        contains = trimesh_intersector.contains_points(points_trimesh)  # [n, ] bool
        contains = torch.tensor(contains).to(points.device)
        contains = 1 - 2*contains  # {-1, 1}, -1 for inside, +1 for outside

        return contains

    # approximate inside/outside computation using vertex normals
    def _calculate_points_inside_meshes_normals(self, points, nearest, triangles, normals_triangles):
        barycentric = points_to_barycentric(triangles, nearest)
        normal_at_s = barycentric_to_points(normals_triangles, barycentric)
        contains = ((points - nearest) * normal_at_s).sum(1) < 0.0
        contains = 1 - 2*contains
        return contains

    def _calculate_parallel_triangles(self, points, triangles, verts_normals, faces_normals):

        batch_dim = points.shape[:-1]

        # if batch dim is larger than 1:
        if points.dim() > 2:
            points = points.view(-1, 3)
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            faces_normals = faces_normals.view(-1, 3)

        dist = ((points-triangles[:, 0]) * faces_normals).sum(1)
        verts_normals_cosine = (verts_normals * faces_normals.unsqueeze(1)).sum(2)  # [batch*65536, 3]
        triangles_parallel = triangles + verts_normals * (dist.view(-1, 1) / (verts_normals_cosine + 1e-12)).unsqueeze(2)  # [batch*13776, 3, 3]

        return triangles_parallel.view(*batch_dim, 3, 3)

    def _calculate_points_inside_target_volume(self, points, triangles, verts_normals, faces_normals, return_barycentric=False):

        batch_dim = points.shape[:-1]
        # if batch dim is larger than 1:
        if points.dim() > 2:
            points = points.view(-1, 3)
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            faces_normals = faces_normals.view(-1, 3)

        triangles_parallel = self._calculate_parallel_triangles(
            points, triangles, verts_normals, faces_normals)
        barycentric = points_to_barycentric(triangles_parallel, points)
        inside = torch.prod(barycentric > 0, dim=1)

        if return_barycentric:
            return inside.view(*batch_dim), barycentric.view(*batch_dim, -1)
        else:
            return inside.view(*batch_dim)

    def _align_verts_normals(self, verts_normals, triangles, ps_sign):

        batch_dim = verts_normals.shape[:-2]
        # if batch dim is larger than 1:
        if verts_normals.dim() > 3:
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            ps_sign = ps_sign.unsqueeze(1).repeat(1, batch_dim[1]).view(-1)

        # revert the direction if points inside the mesh
        verts_normals_signed = verts_normals*ps_sign.view(-1, 1, 1)

        edge1 = triangles - triangles[:, [1, 2, 0]]
        edge2 = triangles - triangles[:, [2, 0, 1]]

        # norm edge direction
        edge1_dir = F.normalize(edge1, dim=2)
        edge2_dir = F.normalize(edge2, dim=2)

        # project verts normals onto triangle plane
        faces_normals = torch.cross(triangles[:, 0]-triangles[:, 2], triangles[:, 1]-triangles[:, 0], dim=1)
        verts_normals_projected = verts_normals_signed - torch.sum(verts_normals_signed*faces_normals.unsqueeze(1), dim=2, keepdim=True)*faces_normals.unsqueeze(1)

        p = torch.sum(edge1_dir*verts_normals_projected, dim=2, keepdim=True)
        q = torch.sum(edge2_dir*verts_normals_projected, dim=2, keepdim=True)
        r = torch.sum(edge1_dir*edge2_dir, dim=2, keepdim=True)

        inv_det = 1 / (1 - r**2 + 1e-9)
        c1 = inv_det * (p - r*q)
        c2 = inv_det * (q - r*p)

        # only align inside normals
        c1 = torch.clamp(c1, max=0.)
        c2 = torch.clamp(c2, max=0.)

        verts_normals_aligned = verts_normals_signed - c1*edge1_dir - c2*edge2_dir
        verts_normals_aligned = F.normalize(verts_normals_aligned, eps=1e-12, dim=2)

        # revert the normals direction
        verts_normals_aligned = verts_normals_aligned*ps_sign.view(-1, 1, 1)

        return verts_normals_aligned.view(*batch_dim, 3, 3)

    # Code modified from function closest_point in Trimesh: https://github.com/mikedh/trimesh/blob/main/trimesh/triangles.py#L544
    def _parse_nearest_projection(self, triangles, points, eps=1e-12):

        # store the location of the closest point
        result = torch.zeros_like(points).to(points.device)
        remain = torch.ones(len(points), dtype=bool).to(points.device)

        # get the three points of each triangle
        # use the same notation as RTCD to avoid confusion
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]

        # check if P is in vertex region outside A
        ab = b - a
        ac = c - a
        ap = points - a
        # this is a faster equivalent of:
        # diagonal_dot(ab, ap)

        d1 = torch.sum(ab * ap, dim=-1)
        d2 = torch.sum(ac * ap, dim=-1)

        # is the point at A
        is_a = torch.logical_and(d1 < eps, d2 < eps)
        if torch.any(is_a):
            result[is_a] = a[is_a]
            remain[is_a] = False

        # check if P in vertex region outside B
        bp = points - b
        d3 = torch.sum(ab * bp, dim=-1)
        d4 = torch.sum(ac * bp, dim=-1)

        # do the logic check
        is_b = (d3 > -eps) & (d4 <= d3) & remain
        if torch.any(is_b):
            result[is_b] = b[is_b]
            remain[is_b] = False

        # check if P in edge region of AB, if so return projection of P onto A
        vc = (d1 * d4) - (d3 * d2)
        is_ab = ((vc < eps) &
                 (d1 > -eps) &
                 (d3 < eps) & remain)
        if torch.any(is_ab):
            v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).view((-1, 1))
            result[is_ab] = a[is_ab] + (v * ab[is_ab])
            remain[is_ab] = False

        # check if P in vertex region outside C
        cp = points - c
        d5 = torch.sum(ab * cp, dim=-1)
        d6 = torch.sum(ac * cp, dim=-1)
        is_c = (d6 > -eps) & (d5 <= d6) & remain
        if torch.any(is_c):
            result[is_c] = c[is_c]
            remain[is_c] = False

        # check if P in edge region of AC, if so return projection of P onto AC
        vb = (d5 * d2) - (d1 * d6)
        is_ac = (vb < eps) & (d2 > -eps) & (d6 < eps) & remain
        if torch.any(is_ac):
            w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).view((-1, 1))
            result[is_ac] = a[is_ac] + w * ac[is_ac]
            remain[is_ac] = False

        # check if P in edge region of BC, if so return projection of P onto BC
        va = (d3 * d6) - (d5 * d4)
        is_bc = ((va < eps) &
                 ((d4 - d3) > - eps) &
                 ((d5 - d6) > -eps) & remain)
        if torch.any(is_bc):
            d43 = d4[is_bc] - d3[is_bc]
            w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).view((-1, 1))
            result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
            remain[is_bc] = False

        # any remaining points must be inside face region
        if torch.any(remain):
            # point is inside face region
            denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
            v = (vb[remain] * denom).reshape((-1, 1))
            w = (vc[remain] * denom).reshape((-1, 1))
            # compute Q through its barycentric coordinates
            result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

        stats = {
            'is_a': is_a,
            'is_b': is_b,
            'is_c': is_c,
            'is_bc': is_bc,
            'is_ac': is_ac,
            'is_ab': is_ab,
            'remain': remain
        }

        return result, stats

    def _revise_nearest(self,
                        points,
                        idx,
                        meshes,
                        inside,
                        nearest,
                        dist,
                        stats,
                        ):

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        faces_normals_meshes = meshes.faces_normals_packed()
        verts_normals_meshes = meshes.verts_normals_packed()[meshes.faces_packed()]

        bc_ca_ab = self.faces_to_corres_edges[idx]
        a_b_c = meshes.faces_packed()[idx]

        is_a, is_b, is_c = stats['is_a'], stats['is_b'], stats['is_c']
        is_bc, is_ac, is_ab = stats['is_bc'], stats['is_ac'], stats['is_ab']

        nearest_new, dist_new, idx_new = nearest.clone(), dist.clone(), idx.clone()

        def _revise(is_x, x_idx, x_type):

            points_is_x = points[is_x]
            inside_is_x = inside[is_x]
            if x_type == 'verts':
                verts_is_x = a_b_c[is_x][:, x_idx]
                corres_faces_is_x = self.verts_to_corres_faces[verts_is_x]
                N_repeat = 9  # maximum # of adjacent faces for verts
            elif x_type == 'edges':
                edges_is_x = bc_ca_ab[is_x][:, x_idx]
                corres_faces_is_x = self.edges_to_corres_faces[edges_is_x]
                N_repeat = 2  # maximum # of adjacent faces for edges
            else:
                raise ValueError('x_type should be verts or edges')

            # STEP 2: Find a set T of all triangles containing s~
            triangles_is_x = triangles_meshes[corres_faces_is_x]
            verts_normals_is_x = verts_normals_meshes[corres_faces_is_x]
            faces_normals_is_x = faces_normals_meshes[corres_faces_is_x]

            # STEP 3: Vertex normal alignment
            verts_normals_is_x_aligned = self._align_verts_normals(verts_normals_is_x, triangles_is_x, inside_is_x)

            # STEP 4: Check if inside control volume
            points_is_x_repeated = points_is_x.unsqueeze(1).repeat(1, N_repeat, 1)
            inside_control_volume, barycentric = \
                self._calculate_points_inside_target_volume(points_is_x_repeated, triangles_is_x, verts_normals_is_x_aligned, faces_normals_is_x, return_barycentric=True)  # (n', N_repeat):bool, (n', N_repeat, 3)
            barycentric = torch.clamp(barycentric, min=0.)
            barycentric = barycentric / (torch.sum(barycentric, dim=-1, keepdim=True) + 1e-12)

            # STEP 5: compute set of canditate surface points {s}
            surface_points_set = (barycentric[..., None] * triangles_is_x).sum(dim=2)
            surface_to_points_dist_set = torch.norm(points_is_x_repeated - surface_points_set, dim=2) + 1e10 * (1 - inside_control_volume)  # [n', N_repeat]
            _, idx_is_x = torch.min(surface_to_points_dist_set, dim=1)  # [n', ]

            # STEP 6: Choose the nearest point to x from {s} as the final projection point
            surface_points = surface_points_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', 3]
            surface_to_points_dist = surface_to_points_dist_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', ]
            faces_is_x = corres_faces_is_x[torch.arange(len(idx_is_x)), idx_is_x]

            # update
            nearest_new[is_x] = surface_points
            dist_new[is_x] = surface_to_points_dist
            idx_new[is_x] = faces_is_x

        # revise verts
        if torch.any(is_a): _revise(is_a, 0, 'verts')
        if torch.any(is_b): _revise(is_b, 1, 'verts')
        if torch.any(is_c): _revise(is_c, 2, 'verts')

        # revise edges
        if torch.any(is_bc): _revise(is_bc, 0, 'edges')
        if torch.any(is_ac): _revise(is_ac, 1, 'edges')
        if torch.any(is_ab): _revise(is_ab, 2, 'edges')

        return nearest_new, dist_new, idx_new

    def _calculate_local_coordinates_meshes(self, faces_normals_meshes, triangles_meshes):
        tangents, bitangents = self._compute_tangent_bitangent(triangles_meshes)
        local_coordinates_meshes = torch.stack((faces_normals_meshes, tangents, bitangents), dim=1)  # [N, 3, 3]
        return local_coordinates_meshes

    def _compute_tangent_bitangent(self, triangles_meshes) -> torch.Tensor:
        # Compute tangets and bitangents following:
        # https://learnopengl.com/Advanced-Lighting/Normal-Mapping
        face_uv = (self.verts_uvs[self.faces_t])  # (13776, 3, 2)
        face_xyz = triangles_meshes  # (13776, 3, 3)
        assert face_uv.shape[-2:] == (3, 2)
        assert face_xyz.shape[-2:] == (3, 3)
        assert face_uv.shape[:-2] == face_xyz.shape[:-2]
        uv0, uv1, uv2 = face_uv.unbind(-2)
        v0, v1, v2 = face_xyz.unbind(-2)
        duv10 = uv1 - uv0
        duv20 = uv2 - uv0
        duv10x = duv10[..., 0:1]
        duv10y = duv10[..., 1:2]
        duv20x = duv20[..., 0:1]
        duv20y = duv20[..., 1:2]
        det = duv10x * duv20y - duv20x * duv10y
        f = 1.0 / (det + 1e-6)
        dv10 = v1 - v0
        dv20 = v2 - v0
        tangents = f * (duv20y * dv10 - duv10y * dv20)
        bitangents = f * (-duv20x * dv10 + duv10x * dv20)
        tangents = F.normalize(tangents, p=2, dim=-1, eps=1e-6)
        bitangents = F.normalize(bitangents, p=2, dim=-1, eps=1e-6)
        return tangents, bitangents

    # parsing mesh (e.g. adjacency of faces, verts, edges, etc.)
    def _parse_mesh(self, verts, faces_idx, N_repeat_edges=2, N_repeat_verts=9):

        meshes = Meshes(verts=[verts], faces=[faces_idx])
        print('parsing mesh topology...')

        # compute faces_to_corres_edges
        faces_to_corres_edges = meshes.faces_packed_to_edges_packed()  # (13776, 3)

        # compute edges_to_corres_faces
        edges_to_corres_faces = torch.full((len(meshes.edges_packed()), N_repeat_edges), -1.0).to(self.device)  # (20664, 2)
        for i in range(len(faces_to_corres_edges)):
            for e in faces_to_corres_edges[i]:
                idx = 0
                while idx < edges_to_corres_faces.shape[1]:
                    if edges_to_corres_faces[e][idx] < 0:
                        edges_to_corres_faces[e][idx] = i
                        break
                    else:
                        idx += 1

        # compute verts_to_corres_faces
        verts_to_corres_faces = torch.full((len(verts), N_repeat_verts), -1.0).to(self.device)  # (6890, 9)
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                idx = 0
                while idx < verts_to_corres_faces.shape[1]:
                    if verts_to_corres_faces[v][idx] < 0:
                        verts_to_corres_faces[v][idx] = i
                        break
                    else:
                        idx += 1
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                verts_to_corres_faces[v][verts_to_corres_faces[v] < 0] = verts_to_corres_faces[v][0]

        return faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces
