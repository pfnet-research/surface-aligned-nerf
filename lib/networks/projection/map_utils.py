import torch

import pytorch3d
from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(mesh_path, device=device):
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path, device=device)
    faces_idx = faces.verts_idx.to(device)
    faces_t = faces.textures_idx.to(device)
    verts_uvs = aux[1].to(device)
    return verts, faces, faces_idx, verts_uvs, faces_t, device


def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    point_to_face, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points
    )
    
    return point_to_face, idxs


def diagonal_dot(a, b):
    return torch.matmul(a * b, torch.ones(a.shape[1]).to(a.device))


def barycentric_to_points(triangles, barycentric):
    return (triangles * barycentric.view((-1, 3, 1))).sum(dim=1)


def points_to_barycentric(triangles, points):

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].view((-1, 3))

    dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
    dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
    dot02 = diagonal_dot(edge_vectors[:, 0], w)
    dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
    dot12 = diagonal_dot(edge_vectors[:, 1], w)

    inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)

    barycentric = torch.zeros(len(triangles), 3).to(points.device)
    barycentric[:, 2] = (dot00 * dot12 - dot01 *
                         dot02) * inverse_denominator
    barycentric[:, 1] = (dot11 * dot02 - dot01 *
                         dot12) * inverse_denominator
    barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

    return barycentric