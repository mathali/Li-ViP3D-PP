import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class cfg:
    pred_traj_num = 6
    future_frame_num = 12
    max_dis_from_ego = 50.0
    matching_threshold = 2.0
    miss_rate_threshold = 2.0
    false_positive_penalty_coefficient = 0.5
    false_negative_penalty_coefficient = 0.5
    precision_epsilon = 1e-6
    # Hyperparameters for displacement-with-FP-penalty metrics
    tau_ADE = miss_rate_threshold
    tau_FDE = miss_rate_threshold
    beta_fp_penalty = 1.0


class GTAgent:
    def __init__(self,
                 translation: np.ndarray = np.zeros(2),
                 future_traj: np.ndarray = np.zeros((cfg.future_frame_num, 2)),
                 future_traj_is_valid: np.ndarray = np.zeros(cfg.future_frame_num, dtype=np.int)
                 ):
        self.translation = translation.copy()
        self.future_traj = future_traj.copy()
        self.future_traj_is_valid = future_traj_is_valid.copy()


class PredAgent:
    def __init__(self,
                 sample_token: str = "",
                 translation: np.ndarray = np.zeros(2),
                 pred_future_trajs: np.ndarray = np.zeros((cfg.pred_traj_num, cfg.future_frame_num, 2)),
                 score: float = 1.0,
                 ):
        self.sample_token = sample_token
        self.translation = translation.copy()
        self.pred_future_trajs = pred_future_trajs.copy()
        self.score = float(score)

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """

        if 'pred_outputs' in content:
            content['pred_future_trajs'] = content['pred_outputs']

        translation = np.array(content['translation'][:2])
        pred_future_trajs = np.array(content['pred_future_trajs'])
        score = float(content.get('tracking_score', 1.0))

        return cls(
            translation=translation,
            pred_future_trajs=pred_future_trajs,
            score=score,
        )


class Metric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def accumulate(self, value):
        if value is not None:
            self.values.append(value)

    def get_mean(self) -> float:
        if len(self.values) > 0:
            return np.mean(self.values)
        else:
            return 0.0

    def get_sum(self) -> float:
        return np.sum(self.values)


def compute_average_precision(scores: np.ndarray,
                              true_positives: np.ndarray,
                              num_gt: int) -> float:
    """
    Compute average precision given prediction scores and TP flags for one class.

    Parameters
    ----------
    scores: np.ndarray of shape (N_pred,)
    true_positives: np.ndarray of shape (N_pred,), values in {0.0, 1.0}
    num_gt: int, total number of ground-truth objects
    """
    if num_gt == 0 or scores.size == 0:
        return 0.0

    order = np.argsort(-scores)
    tp_sorted = true_positives[order]
    fp_sorted = 1.0 - tp_sorted

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)

    recall = cum_tp / float(num_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Make precision non-increasing w.r.t. recall
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    delta_recall = recall - recall_prev

    ap = float(np.sum(delta_recall * precision))
    return ap


class PredictionMetrics:
    def __init__(self) -> None:
        self.minADE = Metric()
        self.minFDE = Metric()
        self.MR = Metric()
        self.matched = Metric()
        self.unmatched = Metric()
        self.matched_and_prediction_hit = Metric()
        self.gt_agent_num = Metric()
        # Sums of per-agent displacement errors over matched agents
        self.ade_sum = Metric()
        self.fde_sum = Metric()

    def serialize(self) -> Dict[str, Any]:
        unmatched_sum = self.unmatched.get_sum()
        matched_sum = self.matched.get_sum()
        matched_and_prediction_hit = self.matched_and_prediction_hit.get_sum()
        gt_agent_num = self.gt_agent_num.get_sum()

        if gt_agent_num > 0:
            # Original EPA: only false positives are penalized
            EPA = (matched_and_prediction_hit - unmatched_sum * cfg.false_positive_penalty_coefficient) / gt_agent_num

            # Extended EPA including false negatives.
            # N_GT is the total number of valid GT agents, H = matched_and_prediction_hit.
            # Define FN = N_GT - H (valid GT agents that are not "hits").
            false_negative_count = max(gt_agent_num - matched_and_prediction_hit, 0.0)
            EPA_FN = (
                matched_and_prediction_hit
                - unmatched_sum * cfg.false_positive_penalty_coefficient
                - false_negative_count * cfg.false_negative_penalty_coefficient
            ) / gt_agent_num
        else:
            EPA = 0.0
            EPA_FN = 0.0

        EPA = max(EPA, 0.0)
        EPA_FN = max(EPA_FN, 0.0)

        # Agent-level precision over all scenes: Prec = #matched / (#matched + #FP)
        denom = matched_sum + unmatched_sum
        Prec = matched_sum / denom if denom > 0 else 0.0

        # Agent-level recall over all scenes:
        # Here we define a "hit" as in EPA: matched agents whose minFDE <= miss_rate_threshold.
        # Recall = #hits / #valid_GT
        if gt_agent_num > 0:
            Recall = matched_and_prediction_hit / gt_agent_num
        else:
            Recall = 0.0

        # NFP percentage: ratio of false positives to total predictions
        if denom > 0:
            NFP_ratio = unmatched_sum / denom
        else:
            NFP_ratio = 0.0

        # Displacement-with-FP-penalty metrics (ADE_FP and FDE_FP)
        ade_sum = self.ade_sum.get_sum()
        fde_sum = self.fde_sum.get_sum()
        N_GT = self.gt_agent_num.get_sum()
        N_FP = unmatched_sum  # total number of false positives

        if N_GT > 0:
            ADE_FP = (ade_sum + cfg.beta_fp_penalty * cfg.tau_ADE * N_FP) / N_GT
            FDE_FP = (fde_sum + cfg.beta_fp_penalty * cfg.tau_FDE * N_FP) / N_GT
        else:
            ADE_FP = 0.0
            FDE_FP = 0.0

        return dict(
            minADE=self.minADE.get_mean(),
            minFDE=self.minFDE.get_mean(),
            MR=self.MR.get_mean(),
            EPA=EPA,
            EPA_FN=EPA_FN,
            NFP=NFP_ratio,
            Prec=Prec,
            Recall=Recall,
            ADE_FP=ADE_FP,
            FDE_FP=FDE_FP,
        )


def get_distances(point, points):
    assert point.ndim == 1 and points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


def get_argmin_trajectory(future_traj, future_traj_is_valid, pred_future_trajs):
    if future_traj_is_valid.sum() == 0:
        return None, None, None

    delta: np.ndarray = pred_future_trajs - future_traj[np.newaxis, :]
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num, 2)

    delta = np.sqrt((delta * delta).sum(-1))
    assert delta.shape == (cfg.pred_traj_num, cfg.future_frame_num)

    if future_traj_is_valid[-1]:
        minFDE = delta[:, -1].min()
    else:
        minFDE = None

    delta = delta * future_traj_is_valid[np.newaxis, :]
    delta = delta.sum(-1) / future_traj_is_valid.sum()
    assert delta.shape == (cfg.pred_traj_num,)

    argmin = delta.argmin()
    minADE = delta.min()

    return argmin, minADE, minFDE


def get_gt_agents(prediction_infos, index):
    idx_2_gt_agent = {}

    for i in range(index, index + 1 + cfg.future_frame_num):
        if i >= len(prediction_infos):
            break

        info = prediction_infos[i]

        instance_inds = np.array(info['instance_inds'], dtype=np.int)
        gt_bboxes_3d = np.array(info['gt_bboxes_3d'])
        gt_labels_3d = np.array(info['gt_labels_3d'], dtype=np.int)

        assert len(instance_inds) == len(gt_bboxes_3d) == len(gt_labels_3d)

        for box_idx, instance_idx in enumerate(instance_inds):
            assert instance_idx != -1
            if i == index:
                assert instance_idx not in idx_2_gt_agent
                idx_2_gt_agent[instance_idx] = GTAgent()

            if instance_idx in idx_2_gt_agent:
                gt_prediction_box = idx_2_gt_agent[instance_idx]

                xy = gt_bboxes_3d[box_idx][:2]

                if i == index:
                    gt_prediction_box.translation[:] = xy
                else:
                    gt_prediction_box.future_traj[i - index - 1] = xy
                    gt_prediction_box.future_traj_is_valid[i - index - 1] = 1

    gt_agents = [value for key, value in idx_2_gt_agent.items()]
    return gt_agents


class PredictionEval:
    def __init__(self,
                 result_path: str = None,
                 output_dir: str = None,
                 prediction_infos_path: str = None):
        """
        Parameters
        ----------
        :param result_path: Path of the JSON result file.
        :param output_dir: Folder to save metrics.
        :param prediction_infos_path: Path of preprocessed gt boxes in JSON format
        """

        """
        Example of JSON result file:
        {
            sample_token_1: {
                {
                    translation: [900.0, 900.0],
                    pred_future_trajs: np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2),
                },
                ...
                {
                    translation: [920.0, 920.0],
                    pred_future_trajs: np.array of shape (cfg.pred_traj_num, cfg.future_frame_num, 2),
                },
            },
            ...
            sample_token_n: {
                ...
            }
        }
        """

        if output_dir is None:
            # set to the directory of `result_path`
            output_dir = os.path.split(result_path)[0]

        self.output_dir = output_dir

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        with open(result_path) as f:
            data = json.load(f)
            if 'results' in data:
                data = data['results']

            self.sample_token_2_pred_agents = {}
            for sample_token, boxes in data.items():
                self.sample_token_2_pred_agents[sample_token] = [PredAgent.deserialize(each) for each in boxes]

        with open(prediction_infos_path, 'r') as f:
            prediction_infos = json.load(f)
            self.prediction_infos = [value for key, value in prediction_infos.items()]

        self.metrics = []
        self.minFDE_list = []

    def evaluate(self):
        metrics = PredictionMetrics()

        # Accumulators for forecasting mAP on the first future frame
        all_scores_first_frame: list[float] = []
        all_tp_flags_first_frame: list[float] = []
        total_gt_first_frame: int = 0

        for index in tqdm(range(len(self.prediction_infos))):
            info = self.prediction_infos[index]
            sample_token = info['sample_token']
            if index > 0:
                assert self.prediction_infos[index]['timestamp'] > self.prediction_infos[index - 1]['timestamp']

            if sample_token not in self.sample_token_2_pred_agents:
                break

            gt_agents: List[GTAgent] = get_gt_agents(self.prediction_infos, index)
            pred_agents: List[PredAgent] = self.sample_token_2_pred_agents[sample_token]

            if len(gt_agents) > 0:

                ade_values_this_frame: list[float] = []
                fde_values_this_frame: list[float] = []

                matched_of_gt_box = np.ones(len(gt_agents), dtype=np.int) * -1
                cost_matrix = np.zeros((len(pred_agents), len(gt_agents)))
                gt_translations = np.array([each.translation for each in gt_agents])

                for i in range(len(pred_agents)):
                    cost_matrix[i] = get_distances(pred_agents[i].translation, gt_translations)
                    cost_matrix[i][np.nonzero(cost_matrix[i] > cfg.matching_threshold)] = 10000.0

                r_list, c_list = linear_sum_assignment(cost_matrix)

                for i in range(len(r_list)):
                    if cost_matrix[r_list[i], c_list[i]] <= cfg.matching_threshold:
                        matched_of_gt_box[c_list[i]] = r_list[i]

                matched = 0
                matched_and_prediction_hit = 0
                gt_not_valid = 0

                for i in range(len(gt_agents)):
                    box_idx = matched_of_gt_box[i]
                    gt_agent = gt_agents[i]

                    if box_idx == -1:
                        minADE = None
                        minFDE = None
                        MR = None
                    else:
                        matched += 1

                        pred_agent = pred_agents[box_idx]
                        argmin, minADE, minFDE = get_argmin_trajectory(gt_agent.future_traj, gt_agent.future_traj_is_valid, pred_agent.pred_future_trajs)

                        if minADE is not None and minADE > 100.0:
                            assert False, f'Error {minADE} is too large!'

                        if gt_agent.future_traj_is_valid[-1]:
                            assert minFDE is not None
                            MR = minFDE > cfg.miss_rate_threshold
                            if not MR:
                                matched_and_prediction_hit += 1
                        else:
                            MR = None
                    if minADE is not None:
                        ade_values_this_frame.append(float(minADE))
                    if minFDE is not None:
                        fde_values_this_frame.append(float(minFDE))

                    metrics.minADE.accumulate(minADE)
                    metrics.minFDE.accumulate(minFDE)
                    metrics.MR.accumulate(MR)

                    if not gt_agent.future_traj_is_valid[-1]:
                        gt_not_valid += 1

                false_positives = len(pred_agents) - matched

                metrics.matched.accumulate(matched)
                metrics.unmatched.accumulate(false_positives)
                metrics.matched_and_prediction_hit.accumulate(matched_and_prediction_hit)
                metrics.gt_agent_num.accumulate(len(gt_agents) - gt_not_valid)
                # Track sums of displacement errors over matched agents
                metrics.ade_sum.accumulate(float(np.sum(ade_values_this_frame)) if ade_values_this_frame else 0.0)
                metrics.fde_sum.accumulate(float(np.sum(fde_values_this_frame)) if fde_values_this_frame else 0.0)

                # ------------------------------------------------------------------
                # Forecasting mAP on the first future frame (index 0)
                # ------------------------------------------------------------------
                # Collect GT centers for agents that have a valid position at t=0
                valid_gt_indices: list[int] = []
                gt_future_centers: list[np.ndarray] = []
                for i_gt, gt_agent in enumerate(gt_agents):
                    if gt_agent.future_traj_is_valid[0]:
                        valid_gt_indices.append(i_gt)
                        gt_future_centers.append(gt_agent.future_traj[0])

                if len(gt_future_centers) > 0:
                    gt_future_centers_arr = np.stack(gt_future_centers, axis=0)  # (N_gt_f, 2)
                    total_gt_first_frame += gt_future_centers_arr.shape[0]

                    num_pred = len(pred_agents)
                    num_gt_f = gt_future_centers_arr.shape[0]
                    # Distance matrix between predictions and GT at first future frame
                    dist_matrix = np.full((num_pred, num_gt_f), np.inf, dtype=float)

                    scores_frame = np.zeros(num_pred, dtype=float)
                    for p_idx, pred_agent in enumerate(pred_agents):
                        scores_frame[p_idx] = float(pred_agent.score)

                        # Use the best (closest) mode at the first future frame
                        future_points = pred_agent.pred_future_trajs[:, 0, :]  # (K, 2)
                        # Distances to each GT center (min over modes)
                        dists_modes = np.linalg.norm(
                            future_points[:, np.newaxis, :] - gt_future_centers_arr[np.newaxis, :, :],
                            axis=2,
                        )  # (K, N_gt_f)
                        dist_matrix[p_idx] = dists_modes.min(axis=0)

                    # Greedy matching in descending score order (per frame)
                    order_frame = np.argsort(-scores_frame)
                    gt_matched_frame = np.zeros(num_gt_f, dtype=bool)
                    tp_flags_frame = np.zeros(num_pred, dtype=float)

                    for p_rank, p_idx in enumerate(order_frame):
                        if num_gt_f == 0:
                            break
                        g_idx = int(np.argmin(dist_matrix[p_idx]))
                        if dist_matrix[p_idx, g_idx] <= cfg.matching_threshold and not gt_matched_frame[g_idx]:
                            tp_flags_frame[p_idx] = 1.0
                            gt_matched_frame[g_idx] = True

                    # Accumulate for dataset-level AP computation
                    all_scores_first_frame.extend(scores_frame.tolist())
                    all_tp_flags_first_frame.extend(tp_flags_frame.tolist())
                else:
                    # No valid GT at first future frame: all predictions are false positives
                    for pred_agent in pred_agents:
                        all_scores_first_frame.append(float(pred_agent.score))
                        all_tp_flags_first_frame.append(0.0)

        # Compute forecasting mAP on the first future frame across the dataset
        scores_arr = np.array(all_scores_first_frame, dtype=float)
        tp_arr = np.array(all_tp_flags_first_frame, dtype=float)
        self.forecast_map_first_frame: float = compute_average_precision(
            scores_arr, tp_arr, total_gt_first_frame
        )

        return metrics

    def main(self) -> Dict[str, Any]:
        metrics: PredictionMetrics = self.evaluate()

        print(f'Saving metrics to: {self.output_dir}/prediction_metrics.json')
        metrics_summary = metrics.serialize()

        # Add forecasting mAP on first future frame if available
        forecast_map_first: float = getattr(self, 'forecast_map_first_frame', 0.0)
        metrics_summary['forecast_mAP_first_frame'] = forecast_map_first

        with open(os.path.join(self.output_dir, 'prediction_metrics.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=4)

        print(json.dumps(metrics_summary, indent=4))

        return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='Prediction evaluation')
    parser.add_argument('--result_path', help='path of prediction results in JSON format')
    parser.add_argument('--prediction_infos_path',
                        default='./nuscenes_prediction_infos_val.json',
                        help='path of preprocessed gt boxes in JSON format')
    args = parser.parse_args()

    nusc_eval = PredictionEval(result_path=args.result_path,
                               prediction_infos_path=args.prediction_infos_path)

    nusc_eval.main()


if __name__ == '__main__':
    main()

