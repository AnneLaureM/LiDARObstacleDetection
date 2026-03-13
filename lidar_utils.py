import h5py
import numpy as np
import pandas as pd


def load_h5_data(file_path):
    """
    Charge le dataset lidar_points depuis le fichier HDF5.
    Retourne un DataFrame pandas.
    """
    with h5py.File(file_path, "r") as f:
        data = f["lidar_points"][:]

    df = pd.DataFrame(data)

    # certains datasets stockent des bytes → conversion string
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    return df


def get_unique_poses(df):
    """
    Retourne toutes les poses uniques (frames) sous forme de DataFrame.
    """
    poses = df[["ego_x", "ego_y", "ego_z", "ego_yaw"]].drop_duplicates()
    return poses


def filter_by_pose(df, pose):
    """
    Sélectionne tous les points correspondant à une pose donnée.
    """
    x, y, z, yaw = pose

    mask = (
        (df["ego_x"] == x)
        & (df["ego_y"] == y)
        & (df["ego_z"] == z)
        & (df["ego_yaw"] == yaw)
    )

    return df[mask]


def spherical_to_local_cartesian(distance_cm, azimuth_raw, elevation_raw):
    """
    Convertit les coordonnées LiDAR sphériques en coordonnées cartésiennes.

    distance_cm : distance en cm
    azimuth_raw : angle brut
    elevation_raw : angle brut
    """

    # conversion cm → m
    d = distance_cm / 100.0

    # conversion angles
    az = azimuth_raw * 0.01 * np.pi / 180.0
    el = elevation_raw * 0.01 * np.pi / 180.0

    x = d * np.cos(el) * np.cos(az)
    y = -d * np.cos(el) * np.sin(az)
    z = d * np.sin(el)

    return np.vstack((x, y, z)).T