import numpy as np


class MCMCDA:
    """
    MCMCDA - Monte Carlo Markov Chain Data Association.
    Args(enum):
        birth_rate: float = 0.01,        # λb - birth rate
        false_alarm_rate: float = 0.05,  # λf - false alarm rate
        disappearance_prob: float = 0.05, # pz - probability of disappearance
        detection_prob: float = 0.90,    # pd - probability of detection
        image_size: tuple = (512, 512),  # Image size for normalization
        num_mcmc_iterations: int = 1000, # Number of iterations for MCMC
        temp_init: float = 10.0,         # Initial temperature for simulated annealing
        temp_final: float = 1.0,         # Final temperature
        iou_threshold: float = 0.3,      # IoU threshold
        min_track_len: int = 3,          # Minimum track length to save
        early_stopping_threshold: float = 0.01,  # Early stopping threshold
        update_freq: int = 5,            # Update frequency (every N frames)
    """

    def __init__(
        self,
        birth_rate=0.01,
        false_alarm_rate=0.05,
        disappearance_prob=0.05,
        detection_prob=0.90,
        image_size=(512, 512),
        num_mcmc_iterations=1000,
        temp_init=10.0,
        temp_final=1.0,
        iou_threshold=0.3,
        min_track_len=3,
        early_stopping_threshold=0.01,
        update_freq=5,
    ):
        self.birth_rate = birth_rate
        self.false_alarm_rate = false_alarm_rate
        self.disappearance_prob = disappearance_prob
        self.detection_prob = detection_prob
        self.image_size = image_size
        self.volume = image_size[0] * image_size[1]  # V - объем области наблюдения

        self.num_mcmc_iterations = num_mcmc_iterations
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.iou_threshold = iou_threshold

        # Optimization parameters
        self.min_track_len = min_track_len
        self.early_stopping_threshold = early_stopping_threshold
        self.update_freq = update_freq

        # Tracker state
        self.tracks = {}  # Active tracks {id: track data}
        self.frame_count = 0  # Current frame number
        self.next_track_id = 0  # Next available track ID
        self.omega = None  # Current association distribution

        self.all_observations = []
        self.history = {}

    def update_frame_observations(self, observations):
        self.frame_count += 1
        frame_observations = {
            "frame_idx": self.frame_count,
            "observations": observations,
        }
        self.all_observations.append(frame_observations)

        if self.frame_count >= 5 and (
            self.frame_count % self.update_freq == 0 or self.omega is None
        ):
            self._run_mcmc()

    def _run_mcmc(self):
        omega = self._initialize_association()

        accepted_count = 0
        total_count = 0

        cooling_factor = (self.temp_final / self.temp_init) ** (
            1.0 / self.num_mcmc_iterations
        )
        temp = self.temp_init

        current_log_posterior = self._compute_log_posterior(omega)
        prev_log_posterior = current_log_posterior

        no_improvement_count = 0

        for i in range(self.num_mcmc_iterations):
            proposed_omega = self._propose_move(omega)

            proposed_log_posterior = self._compute_log_posterior(proposed_omega)

            log_acceptance_ratio = proposed_log_posterior - current_log_posterior

            if np.log(np.random.random()) < min(0, log_acceptance_ratio / temp):
                omega = proposed_omega
                current_log_posterior = proposed_log_posterior
                accepted_count += 1

            total_count += 1

            temp *= cooling_factor

            improvement = abs(current_log_posterior - prev_log_posterior)
            if improvement < self.early_stopping_threshold:
                no_improvement_count += 1
                if (
                    no_improvement_count >= 10
                ):  # 10 iterations without significant improvements
                    break
            else:
                no_improvement_count = 0

            prev_log_posterior = current_log_posterior

        self.omega = omega

        self._update_tracks_from_mcmc(omega)

        acceptance_rate = accepted_count / total_count if total_count > 0 else 0
        self.history[self.frame_count] = {
            "iterations": i + 1,
            "log_posterior": current_log_posterior,
            "acceptance_rate": acceptance_rate,
        }

    def _initialize_association(self):
        omega = {"clutter": [], "tracks": {}, "num_tracks": 0}

        all_idx = []
        for frame_idx, frame_data in enumerate(self.all_observations):
            for obs_idx, _ in enumerate(frame_data["observations"]):
                all_idx.append((frame_idx + 1, obs_idx + 1))

        omega["clutter"] = all_idx
        return omega

    def _propose_move(self, omega):
        proposed_omega = {
            "clutter": omega["clutter"].copy(),
            "tracks": {k: v.copy() for k, v in omega["tracks"].items()},
            "num_tracks": omega["num_tracks"],
        }

        has_tracks = len(proposed_omega["tracks"]) > 0

        move_type = np.random.choice(["birth", "death", "split", "merge", "update"])
        if has_tracks:
            move_type = np.random.choice(["birth", "death", "split", "merge", "update"])
        else:
            move_type = np.random.choice(["birth", "update"])

        if move_type == "birth":
            if len(proposed_omega["clutter"]) >= 2:
                num_points = np.random.randint(
                    1, min(5, len(proposed_omega["clutter"]))
                )
                indices = np.random.choice(
                    len(proposed_omega["clutter"]), size=num_points, replace=False
                )

                new_track_id = proposed_omega["num_tracks"] + 1
                proposed_omega["tracks"][new_track_id] = []

                for idx in sorted(indices, reverse=True):
                    point = proposed_omega["clutter"].pop(idx)
                    proposed_omega["tracks"][new_track_id].append(point)

                proposed_omega["num_tracks"] += 1

        elif move_type == "death":
            if has_tracks:
                track_id = np.random.choice(list(proposed_omega["tracks"].keys()))

                proposed_omega["clutter"].extend(proposed_omega["tracks"][track_id])

                del proposed_omega["tracks"][track_id]

        elif move_type == "split":
            if has_tracks:
                valid_tracks = [
                    tid
                    for tid, points in proposed_omega["tracks"].items()
                    if len(points) > 1
                ]
                if valid_tracks:
                    track_id = np.random.choice(valid_tracks)
                    track_points = proposed_omega["tracks"][track_id]

                    split_idx = np.random.randint(1, len(track_points))

                    new_track_id = proposed_omega["num_tracks"] + 1
                    proposed_omega["tracks"][new_track_id] = track_points[split_idx:]
                    proposed_omega["tracks"][track_id] = track_points[:split_idx]
                    proposed_omega["num_tracks"] += 1

        elif move_type == "merge":
            if len(proposed_omega["tracks"]) > 1:
                track_ids = np.random.choice(
                    list(proposed_omega["tracks"].keys()), size=2, replace=False
                )

                proposed_omega["tracks"][track_ids[0]].extend(
                    proposed_omega["tracks"][track_ids[1]]
                )

                del proposed_omega["tracks"][track_ids[1]]

        elif move_type == "update":
            if np.random.random() < 0.5 and proposed_omega["clutter"] and has_tracks:
                track_id = np.random.choice(list(proposed_omega["tracks"].keys()))
                if proposed_omega["tracks"][track_id]:
                    track_idx = np.random.randint(
                        len(proposed_omega["tracks"][track_id])
                    )
                    clutter_idx = np.random.randint(len(proposed_omega["clutter"]))

                    (
                        proposed_omega["tracks"][track_id][track_idx],
                        proposed_omega["clutter"][clutter_idx],
                    ) = (
                        proposed_omega["clutter"][clutter_idx],
                        proposed_omega["tracks"][track_id][track_idx],
                    )
            elif len(proposed_omega["tracks"]) > 1:
                track_ids = np.random.choice(
                    list(proposed_omega["tracks"].keys()), size=2, replace=False
                )
                if (
                    proposed_omega["tracks"][track_ids[0]]
                    and proposed_omega["tracks"][track_ids[1]]
                ):
                    idx1 = np.random.randint(
                        len(proposed_omega["tracks"][track_ids[0]])
                    )
                    idx2 = np.random.randint(
                        len(proposed_omega["tracks"][track_ids[1]])
                    )

                    (
                        proposed_omega["tracks"][track_ids[0]][idx1],
                        proposed_omega["tracks"][track_ids[1]][idx2],
                    ) = (
                        proposed_omega["tracks"][track_ids[1]][idx2],
                        proposed_omega["tracks"][track_ids[0]][idx1],
                    )

        return proposed_omega

    def _compute_log_posterior(self, omega):
        if hasattr(self, "_cached_omega") and self._cached_omega == omega:
            return self._cached_log_posterior

        log_likelihood = 0.0  # log P(Y|ω)
        log_prior = 0.0  # log P(ω)

        track_observations = {}
        for track_id, indices in omega["tracks"].items():
            track_observations[track_id] = []
            for frame_idx, obs_idx in indices:
                frame_idx_adj = frame_idx - 1
                obs_idx_adj = obs_idx - 1

                if 0 <= frame_idx_adj < len(self.all_observations):
                    frame_data = self.all_observations[frame_idx_adj]
                    if 0 <= obs_idx_adj < len(frame_data["observations"]):
                        track_observations[track_id].append(
                            frame_data["observations"][obs_idx_adj]
                        )

        track_likelihoods = []
        for track_id, observations in track_observations.items():
            if observations:
                track_likelihood = self._compute_track_likelihood(observations)
                log_likelihood += (
                    np.log(track_likelihood) if track_likelihood > 0 else -np.inf
                )

        frame_counts = self._compute_frame_counts(omega)
        for t, counts in frame_counts.items():
            # log(pz^zt * (1-pz)^ct)
            log_prior += counts["zt"] * np.log(self.disappearance_prob) + counts[
                "ct"
            ] * np.log(1 - self.disappearance_prob)

            # log(pd^dt * (1-pd)^gt)
            log_prior += counts["dt"] * np.log(self.detection_prob) + counts[
                "gt"
            ] * np.log(1 - self.detection_prob)

            birth_term = self.birth_rate * self.volume
            false_alarm_term = self.false_alarm_rate * self.volume

            if counts["at"] > 0:
                log_prior += (
                    counts["at"] * np.log(birth_term) if birth_term > 0 else -np.inf
                )

            if counts["ft"] > 0:
                log_prior += (
                    counts["ft"] * np.log(false_alarm_term)
                    if false_alarm_term > 0
                    else -np.inf
                )

            if counts["mu_t"] > 0:
                if not hasattr(self, "_factorial_cache"):
                    self._factorial_cache = {}

                if counts["mu_t"] not in self._factorial_cache:
                    if counts["mu_t"] > 20:
                        self._factorial_cache[counts["mu_t"]] = (
                            counts["mu_t"] * np.log(counts["mu_t"]) - counts["mu_t"]
                        )
                    else:
                        self._factorial_cache[counts["mu_t"]] = np.log(
                            np.math.factorial(counts["mu_t"])
                        )

                log_prior -= self._factorial_cache[counts["mu_t"]]

        log_posterior = log_likelihood + log_prior
        self._cached_omega = omega
        self._cached_log_posterior = log_posterior

        return log_posterior

    def _compute_track_likelihood(self, observations):
        if len(observations) <= 1:
            return 1.0

        total_likelihood = 1.0

        for i in range(1, len(observations)):
            prev_obs = observations[i - 1]
            curr_obs = observations[i]

            prev_bbox = prev_obs["bbox"]
            curr_bbox = curr_obs["bbox"]

            iou = self._compute_iou(prev_bbox, curr_bbox)

            if iou >= self.iou_threshold:
                pair_likelihood = iou
            else:
                pair_likelihood = iou * 0.1

            if "frame_idx" in prev_obs and "frame_idx" in curr_obs:
                time_diff = abs(curr_obs["frame_idx"] - prev_obs["frame_idx"])
                if time_diff > 1:
                    pair_likelihood *= 0.5 ** (time_diff - 1)

            total_likelihood *= pair_likelihood

        return total_likelihood

    def _update_tracks_from_mcmc(self, omega):
        new_tracks = {}

        for track_id, indices in omega["tracks"].items():
            if len(indices) < self.min_track_len:
                continue

            track_observations = []
            valid_frames = set()

            for frame_idx, obs_idx in indices:
                frame_idx_adj = frame_idx - 1
                obs_idx_adj = obs_idx - 1

                if 0 <= frame_idx_adj < len(self.all_observations):
                    frame_data = self.all_observations[frame_idx_adj]
                    if 0 <= obs_idx_adj < len(frame_data["observations"]):
                        obs = frame_data["observations"][obs_idx_adj].copy()
                        obs["frame_idx"] = frame_idx
                        track_observations.append(obs)
                        valid_frames.add(frame_idx)

            if len(track_observations) < self.min_track_len:
                continue

            track_observations.sort(key=lambda x: x["frame_idx"])

            new_track = {
                "id": track_id,
                "observations": track_observations,
                "first_frame": track_observations[0]["frame_idx"],
                "last_frame": track_observations[-1]["frame_idx"],
                "state": self._estimate_state(track_observations),
            }
            new_tracks[track_id] = new_track

        self.tracks = new_tracks

        if len(self.all_observations) > 30:
            self.all_observations = self.all_observations[-30:]

        omega["num_tracks"] = len(new_tracks)

    def _estimate_state(self, observations):
        if not observations:
            return None

        last_obs = observations[-1]

        state = {"bbox": last_obs["bbox"], "score": last_obs["score"]}

        if "mask" in last_obs:
            state["mask"] = last_obs["mask"]

        if len(observations) > 1:
            prev_obs = observations[-2]
            curr_obs = last_obs

            prev_center_x = (prev_obs["bbox"][0] + prev_obs["bbox"][2]) / 2
            prev_center_y = (prev_obs["bbox"][1] + prev_obs["bbox"][3]) / 2
            curr_center_x = (curr_obs["bbox"][0] + curr_obs["bbox"][2]) / 2
            curr_center_y = (curr_obs["bbox"][1] + curr_obs["bbox"][3]) / 2

            if "frame_idx" in prev_obs and "frame_idx" in curr_obs:
                time_diff = curr_obs["frame_idx"] - prev_obs["frame_idx"]
                if time_diff > 0:
                    velocity_x = (curr_center_x - prev_center_x) / time_diff
                    velocity_y = (curr_center_y - prev_center_y) / time_diff
                    state["velocity"] = [velocity_x, velocity_y]

        return state

    def get_active_tracks(self):
        return self.tracks

    def get_track_states(self):
        states = {}
        for track_id, track in self.tracks.items():
            if "state" in track and track["state"]:
                states[track_id] = track["state"]
        return states

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = bbox1_area + bbox2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def _compute_frame_counts(self, omega):
        frame_counts = {}

        all_frames = set()
        for track_id, indices in omega["tracks"].items():
            for frame_idx, _ in indices:
                all_frames.add(frame_idx)

        for frame_idx in range(1, self.frame_count + 1):
            tracks_t_minus_1 = set()
            tracks_t = set()

            for track_id, indices in omega["tracks"].items():
                frames = [idx[0] for idx in indices]
                if frame_idx - 1 in frames:
                    tracks_t_minus_1.add(track_id)
                if frame_idx in frames:
                    tracks_t.add(track_id)

            et = len(tracks_t)

            et_minus_1 = len(tracks_t_minus_1) if frame_idx > 1 else 0

            zt = len(tracks_t_minus_1 - tracks_t) if frame_idx > 1 else 0

            at = len(tracks_t - tracks_t_minus_1)

            dt = sum(
                1
                for track_id in tracks_t
                for idx in omega["tracks"][track_id]
                if idx[0] == frame_idx
            )

            if frame_idx - 1 < len(self.all_observations):
                mu_t = len(self.all_observations[frame_idx - 1]["observations"])
            else:
                mu_t = 0

            ft = mu_t - dt

            ct = et_minus_1 - zt

            gt = et_minus_1 - zt + at - dt

            frame_counts[frame_idx] = {
                "et": et,
                "et_minus_1": et_minus_1,
                "zt": zt,
                "at": at,
                "dt": dt,
                "mu_t": mu_t,
                "ft": ft,
                "ct": ct,
                "gt": gt,
            }

        return frame_counts
