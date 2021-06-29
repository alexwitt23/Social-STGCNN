"""Data preprocessing scripts."""

import math
import os
import pathlib
import pickle
import sys
import time

import torch
import numpy as np

import torch.utils.data as torch_data
import networkx as nx
import tqdm


def anorm(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(p1 - p2)
    if norm == 0:
        return 0
    return 1 / (norm)


def seq_to_graph(seq_: torch.Tensor, seq_rel: torch.Tensor, norm_lap_matr=True):
    """Convert a sequence to a graph trajectory.

    Args:
        seq_: the sequence of velocities for pedestrians: [pedestrian, vel, prev_frames]
        seq_: the sequence of relative velocities for pedestrians:
            [pedestrian, vel, prev_frames]

    """
    #seq_ = seq_.squeeze()
    #seq_rel = seq_rel.squeeze()
    # Get the sequence length
    seq_len = seq_.shape[2]
    # Max number of pedestrians
    max_nodes = seq_.shape[0]

    velocity = np.zeros((seq_len, max_nodes, 2))
    adjecency = np.zeros((seq_len, max_nodes, max_nodes))

    for seq_idx in range(seq_len):
        step_ = seq_[:, :, seq_idx]
        step_rel = seq_rel[:, :, seq_idx]
        # Loop over the pedestrians in the frame and find the distance from this pedestrian
        # to all others.
        num_people = len(step_)
        for pedestrian_a in range(num_people):
            velocity[seq_idx, pedestrian_a, :] = step_rel[pedestrian_a]
            adjecency[seq_idx, pedestrian_a, pedestrian_a] = 1
            for pedestrian_b in range(pedestrian_a + 1, num_people):
                # Get the adjacent norm (1 / l2_norm) between the humans
                l2_norm = anorm(step_rel[pedestrian_a], step_rel[pedestrian_b])
                adjecency[seq_idx, pedestrian_a, pedestrian_b] = l2_norm
                adjecency[seq_idx, pedestrian_b, pedestrian_a] = l2_norm
        #print(adjacency)
        #if norm_lap_matr:
        #    G = nx.from_numpy_matrix(adjecency[s, :, :])
        #    adjecency[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.Tensor(velocity), torch.Tensor(adjecency)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def _read_data_file(text_file: pathlib.Path) -> np.ndarray:
    """Read in the data text file. The files in <frame_id> <ped_id> <x> <y> format."""
    data = []
    for line in text_file.read_text().splitlines():
        # convert from str to float
        data.append([float(value) for value in line.strip().split()])

    return np.asarray(data)


class TrajectoryDataset(torch_data.Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir: pathlib.Path,
        obs_len: int = 8,
        pred_len: int = 8,
        skip: int = 1,
        threshold: float = 0.002,
        min_ped: int = 1,
        norm_lap_matr: bool = True,
    ) -> None:
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = pathlib.Path(data_dir)
        self.process_dir = self.data_dir / "processed"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.norm_lap_matr = norm_lap_matr
        self.threshold = threshold
        self.min_ped = min_ped

        self.data_objs = {
            "obs_traj": None,
            "pred_traj": None,
            "obs_traj_rel": None,
            "pred_traj_rel": None,
            "non_linear_ped": None,
            "loss_mask": None,
            "v_obs": None,
            "A_obs": None,
            "v_pred": None,
            "A_pred": None,
            "seq_start_end": None,
            "num_seq": None,
        }
        # Try to load the previously processed data. If there is a file missing, go
        # ahead and reprocess everything.
        try:
            if self.process_dir.is_dir():
                self._load_pickled_data()
            else:
                self._convert_data()
        except FileNotFoundError:
            self._convert_data()

    def _convert_data(self) -> None:
        """The first time training is run, we need to convert the data into the graph
        relations."""
        self.process_dir.mkdir(exist_ok=True, parents=True)
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        # Loop over all the data text files
        for path in self.data_dir.glob("*.txt"):
            # Read the data into <frame_id> <ped_id> <x> <y> format
            data = _read_data_file(path)
            # Get a list of the unique frames.
            frame_ids = np.unique(data[:, 0]).tolist()
            # Add the frame data to a list
            frame_data = [data[frame_id == data[:, 0], :] for frame_id in frame_ids]
            num_sequences = int(
                math.ceil((len(frame_ids) - self.seq_len + 1) / self.skip)
            )

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # Get the sequence
                curr_seq_data = np.vstack(frame_data[idx : idx + self.seq_len])
                # Get the unique pedestrian ids in this sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # Only take up to some N number of pedestrians in a frame
                self.max_peds_in_frame = max(
                    self.max_peds_in_frame, len(peds_in_curr_seq)
                )
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                # Loop over all the unique pedestrian ids in this sequence.
                for ped_id in peds_in_curr_seq:
                    # Extract this pedestrian's velocity data.
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # TODO(alex): figure out why this is needed
                    pad_front = frame_ids.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frame_ids.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    # <frame_id> <ped_id> <x> <y> -> <frame_id> <ped_id> <y> <x>
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # Make coordinates relative. Every pedestrian starts at 0.
                    # TODO(alex): Each position is (x_t - x_{t-1}, y_t - y_{t-1})?
                    rel_curr_ped_seq = np.zeros_like(curr_ped_seq)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    # Add in the non-relative pedestrian frame sequence
                    curr_seq[num_peds_considered, :, pad_front:pad_end] = curr_ped_seq
                    # Add in the relative pedestrian frame sequence
                    curr_seq_rel[
                        num_peds_considered, :, pad_front:pad_end
                    ] = rel_curr_ped_seq

                    # TODO(alex): not sure this is used.
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, self.pred_len, self.threshold)
                    )
                    # Consider this pedestrian for the loss
                    curr_loss_mask[num_peds_considered, pad_front:pad_end] = 1
                    num_peds_considered += 1
                # If there are more pedestrians in this sequence than we are processing,
                # cut some off.
                if num_peds_considered > self.min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.vstack(seq_list)
        seq_list_rel = np.vstack(seq_list_rel)
        loss_mask_list = np.vstack(loss_mask_list)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.Tensor(seq_list[:, :, :self.obs_len])
        self.pred_traj = torch.Tensor(seq_list[:, :, self.obs_len:])
        self.obs_traj_rel = torch.Tensor(seq_list_rel[:, :, :self.obs_len])
        self.pred_traj_rel = torch.Tensor(seq_list_rel[:, :, self.obs_len:])
        self.loss_mask = torch.Tensor(loss_mask_list)
        self.non_linear_ped = torch.Tensor(non_linear_ped)

        # Get the start-end indices of the pedestrians
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        for ss in tqdm.tqdm(range(len(self.seq_start_end))):

            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(
                self.obs_traj[start:end, :],
                self.obs_traj_rel[start:end, :],
                self.norm_lap_matr,
            )
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(
                self.pred_traj[start:end, :],
                self.pred_traj_rel[start:end, :],
                self.norm_lap_matr,
            )
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())

        self.data_objs = {
            "obs_traj": self.obs_traj,
            "pred_traj": self.pred_traj,
            "obs_traj_rel": self.obs_traj_rel,
            "pred_traj_rel": self.pred_traj_rel,
            "non_linear_ped": self.non_linear_ped,
            "loss_mask": self.loss_mask,
            "v_obs": self.v_obs,
            "A_obs": self.A_obs,
            "v_pred": self.v_pred,
            "A_pred": self.A_pred,
            "seq_start_end": self.seq_start_end,
            "num_seq": self.num_seq,
        }

        for file_name, data in self.data_objs.items():
            save_path = self.process_dir / file_name
            save_path.write_bytes(pickle.dumps(data))

    def _load_pickled_data(self) -> None:
        for file_name in self.data_objs.keys():
            self.data_objs[file_name] = pickle.loads(
                (self.process_dir / file_name).read_bytes()
            )
        self.seq_start_end = self.data_objs["seq_start_end"]
        self.num_seq = self.data_objs["num_seq"]
        print(self.num_seq, len(self.seq_start_end), len(self.data_objs["v_obs"]))

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        return [
            self.data_objs["obs_traj"][start:end, :],
            self.data_objs["pred_traj"][start:end, :],
            self.data_objs["obs_traj_rel"][start:end, :],
            self.data_objs["pred_traj_rel"][start:end, :],
            self.data_objs["non_linear_ped"][start:end],
            self.data_objs["loss_mask"][start:end, :],
            self.data_objs["v_obs"][index],
            self.data_objs["A_obs"][index],
            self.data_objs["v_pred"][index],
            self.data_objs["A_pred"][index],
        ]
