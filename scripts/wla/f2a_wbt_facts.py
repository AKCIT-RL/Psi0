"""F2a — fatos do WBT oficial (só G1_WBT_Brainco_Supermarket_Shelf_Organizing, nenhum episódio Ψ0).

Grava $WLA_EXP/f2_validation/wbt_facts.json com:
 a) frame do EE "gripper_base": FK(URDF) vs observation.state.*_ee_pose_gripper_base sob várias hipóteses
    (base pelvis/torso, ordem do punho, ordem da cintura, tip) + offset fixo E por mínimos quadrados;
 b) sentido de fig6d (0=aberto?) e relação action vs state;
 c) semântica de base_command[2] (angle_z) e [3] (altura), vx/vy;
 d) lag entre action.*_ee_pose(t) e observation.state.*_ee_pose(t+k).
Uso: PYTHONPATH=src python scripts/wla/f2a_wbt_facts.py [--out DIR] [--n-episodes 20]
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from wla_adapter import pipeline as P
from wla_adapter.geometry import fk, rotation_angle, se3_to_xyz_rpy, xyz_rpy_to_se3

DATA_ROOT = Path(os.environ.get("WLA_DATA_ROOT", "/raid/user_marcospaulo/datasets/unifolm"))
DATA_DIR = DATA_ROOT / "UnifoLM_WBT_Dataset/G1_WBT_Brainco_Supermarket_Shelf_Organizing"
FPS = 30.0
OS = "observation.state."
K = {
    "waist": OS + "waist_state_joint", "valid": OS + "recomputed_ee_valid", "base_pose": OS + "state_base_pose",
    "base_rot": OS + "state_base_rot", "hand_state": OS + "hand_state",
    "act_cmd": "action.base_command", "act_waist": "action.waist_action_joint",
}
for s in ("left", "right"):
    K[f"{s}_arm"] = OS + f"{s}_arm"
    K[f"{s}_ee_base"] = OS + f"{s}_ee_pose_gripper_base"
    K[f"{s}_ee_torso"] = OS + f"{s}_ee_pose_gripper_torso"
    K[f"{s}_fig_s"] = OS + f"{s}_fig6d"
    K[f"{s}_fig_a"] = f"action.{s}_fig6d"
    K[f"{s}_ee_act"] = f"action.{s}_ee_pose_gripper_base"
    K[f"{s}_arm_act"] = f"action.{s}_arm"


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summ(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "p50": pct(x, 50), "p99": pct(x, 99), "max": float(x.max()), "n": int(x.size)}


def fit_E(T_fk, T_m):
    """E constante (tip->EE) tal que T_m ≈ T_fk @ E: rotação por projeção SVD da média, translação por LS."""
    M = np.einsum("nji,njk->ik", T_fk[:, :3, :3], T_m[:, :3, :3])
    U, _, Vt = np.linalg.svd(M)
    Re = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    A = T_fk[:, :3, :3].reshape(-1, 3)
    b = (T_m[:, :3, 3] - T_fk[:, :3, 3]).reshape(-1)
    pE = np.linalg.lstsq(A, b, rcond=None)[0]
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = Re, pE
    return E


def residual(T_fk, T_m, E):
    T = T_fk @ E
    return np.linalg.norm(T[:, :3, 3] - T_m[:, :3, 3], axis=-1) * 1e3, np.degrees(rotation_angle(T[:, :3, :3], T_m[:, :3, :3]))


# ---------------------------------------------------------------- a) EE frame
def part_a(cols, sel_eps, valid):
    ep = cols["episode_index"]
    rows = np.flatnonzero(np.isin(ep, sel_eps))
    fit_eps = set(sel_eps[::2])
    is_fit = np.array([e in fit_eps for e in ep[rows]])
    ok = valid[rows]
    fit_rows, held_rows = rows[is_fit & ok], rows[~is_fit & ok]
    out = {"n_frames_selected": int(len(rows)), "valid_fraction_selected": float(ok.mean()),
           "fit_episodes": [int(e) for e in sorted(fit_eps)],
           "heldout_episodes": [int(e) for e in sel_eps if e not in fit_eps]}
    waist_perms = list(itertools.permutations(range(3)))
    wrist_perms = list(itertools.permutations(range(3)))
    wn, wr = ["yaw", "roll", "pitch"], ["roll", "pitch", "yaw"]  # URDF
    table = []
    for target in ("gripper_base", "gripper_torso"):
        Tm = {s: xyz_rpy_to_se3(cols[K[f"{s}_ee_{'base' if target == 'gripper_base' else 'torso'}"]]) for s in ("left", "right")}
        for base in ("pelvis", "torso_link"):
            for wp in (waist_perms if base == "pelvis" else [(0, 1, 2)]):
                for wrp in wrist_perms:
                    res_side = {}
                    for tip in ("wrist_yaw", "hand_palm"):
                        for side in ("left", "right"):
                            qa = cols[K[f"{side}_arm"]].astype(np.float64)
                            q_arm = np.concatenate([qa[:, :4], qa[:, 4:7][:, list(wrp)]], axis=1)
                            q_w = cols[K["waist"]].astype(np.float64)[:, list(wp)]
                            Tfk = fk(q_w, q_arm, side, base_link=base, tip_link=tip)
                            E = fit_E(Tfk[fit_rows], Tm[side][fit_rows])
                            pos_h, rot_h = residual(Tfk[held_rows], Tm[side][held_rows], E)
                            pos_a, rot_a = residual(Tfk[np.r_[fit_rows, held_rows]], Tm[side][np.r_[fit_rows, held_rows]], E)
                            res_side[(tip, side)] = {"E_xyz_rpy": se3_to_xyz_rpy(E).tolist(), "held_pos_mm": summ(pos_h),
                                                     "held_rot_deg": summ(rot_h), "all_pos_mm": summ(pos_a), "all_rot_deg": summ(rot_a)}
                    for tip in ("wrist_yaw", "hand_palm"):
                        score = max(max(res_side[(tip, s)]["held_pos_mm"]["p99"] / 5.0, res_side[(tip, s)]["held_rot_deg"]["p99"] / 1.0)
                                    for s in ("left", "right"))
                        table.append({
                            "target": target, "base": base, "tip": tip,
                            "waist_cols_to_urdf(yaw,roll,pitch)": [wn[i] for i in wp] if base == "pelvis" else "n/a",
                            "wrist_cols_to_urdf(roll,pitch,yaw)": [["wrist_yaw", "wrist_roll", "wrist_pitch"][i] for i in wrp],
                            "score(<1 => VERIFIED)": score,
                            "waist_perm_idx": list(wp), "wrist_perm_idx": list(wrp),
                            "left": res_side[(tip, "left")], "right": res_side[(tip, "right")]})
    table.sort(key=lambda r: r["score(<1 => VERIFIED)"])
    out["n_hypotheses"] = len(table)
    out["best_per_target"] = {t: next(r for r in table if r["target"] == t) for t in ("gripper_base", "gripper_torso")}
    best = out["best_per_target"]["gripper_base"]
    out["verdict_gripper_base"] = "VERIFIED" if best["score(<1 => VERIFIED)"] < 1 else "UNRESOLVED"
    out["top10_hypotheses"] = [{k: r[k] for k in r if k not in ("left", "right")} | {
        "left_held_pos_p99_mm": r["left"]["held_pos_mm"]["p99"], "left_held_rot_p99_deg": r["left"]["held_rot_deg"]["p99"],
        "right_held_pos_p99_mm": r["right"]["held_pos_mm"]["p99"], "right_held_rot_p99_deg": r["right"]["held_rot_deg"]["p99"],
    } for r in table[:10]]
    # resumo por (base, wrist) — melhor waist, melhor tip
    summary = {}
    for r in table:
        key = f"{r['target']}|{r['base']}|wrist={r['wrist_cols_to_urdf(roll,pitch,yaw)']}"
        summary.setdefault(key, r["score(<1 => VERIFIED)"])
    out["best_score_by_target_base_wrist"] = dict(sorted(summary.items(), key=lambda kv: kv[1])[:20])
    out["all_hypotheses"] = [{k: r[k] for k in r if k not in ("left", "right")} | {
        "left_held_pos_p99_mm": r["left"]["held_pos_mm"]["p99"], "right_held_pos_p99_mm": r["right"]["held_pos_mm"]["p99"],
        "left_held_rot_p99_deg": r["left"]["held_rot_deg"]["p99"], "right_held_rot_p99_deg": r["right"]["held_rot_deg"]["p99"]}
        for r in table]
    return out, best


# ---------------------------------------------------------------- b) fig6d
def part_b(cols, ep_ids):
    ep = cols["episode_index"]
    first = np.array([np.flatnonzero(ep == e)[0] for e in ep_ids])
    names = ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"]
    out = {}
    for s in ("left", "right"):
        st, ac = cols[K[f"{s}_fig_s"]], cols[K[f"{s}_fig_a"]]
        ep_max = np.array([st[ep == e].max(axis=0) for e in ep_ids])
        ep_min = np.array([st[ep == e].min(axis=0) for e in ep_ids])
        d = {"dims": names, "global_min": st.min(0).tolist(), "global_max": st.max(0).tolist(),
             "global_p5_p50_p95": [np.percentile(st, q, axis=0).tolist() for q in (5, 50, 95)],
             "state_frame0_mean": st[first].mean(0).tolist(), "state_frame0_max": st[first].max(0).tolist(),
             "episode_max_mean(~peak during task)": ep_max.mean(0).tolist(), "episode_min_mean": ep_min.mean(0).tolist(),
             "frac_frames_gt_0.5": (st > 0.5).mean(0).tolist(), "frac_episodes_frame0_gt_0.5": (st[first] > 0.5).mean(0).tolist()}
        mov = ep_max.mean(0) - ep_min.mean(0) > 0.05  # dims que se movem durante a tarefa
        mov[1] = False  # thumb_lat tem sentido oposto (sobe ao fechar); avaliado à parte pelos médios abertos/fechados
        d["dims_moving_during_task(excl. thumb_lat)"] = mov.tolist()
        d["frame0_closer_to_episode_max_than_min(moving dims)"] = bool(np.all(
            (np.abs(st[first].mean(0) - ep_max.mean(0)) < np.abs(st[first].mean(0) - ep_min.mean(0)))[mov])) if mov.any() else None
        # frames em que a ação comanda fechar (dims de dedos index/middle/ring/little < 0.5)
        closed = ac[:, 2:6].mean(1) < 0.5
        d["frac_frames_cmd_closed(action idx..little mean<0.5)"] = float(closed.mean())
        d["state_mean_when_cmd_closed"] = st[closed].mean(0).tolist() if closed.any() else None
        d["state_mean_when_cmd_open"] = st[~closed].mean(0).tolist()
        d["action_mean_when_cmd_closed"] = ac[closed].mean(0).tolist() if closed.any() else None
        # action(t) vs state(t+k)
        lagres = {}
        for k in range(-2, 9):
            idx = np.arange(len(ep))
            j = idx + k
            okm = (j >= 0) & (j < len(ep))
            okm[okm] &= ep[j[okm]] == ep[idx[okm]]
            lagres[k] = float(np.abs(ac[idx[okm]] - st[j[okm]]).mean())
        kb = min(lagres, key=lagres.get)
        d["action_vs_state_lag_mae"] = {str(k): v for k, v in lagres.items()}
        d["best_lag_frames"] = kb
        j = np.arange(len(ep)) + kb
        okm = (j >= 0) & (j < len(ep))
        okm[okm] &= ep[j[okm]] == ep[np.flatnonzero(okm)]
        a, b = ac[np.flatnonzero(okm)], st[j[okm]]
        d["linear_fit_state_t+k_vs_action_t_per_dim"] = [
            ({"slope": float(np.polyfit(a[:, i], b[:, i], 1)[0]), "intercept": float(np.polyfit(a[:, i], b[:, i], 1)[1]),
              "corr": float(np.corrcoef(a[:, i], b[:, i])[0, 1])} if a[:, i].std() > 1e-6 and b[:, i].std() > 1e-6 else None)
            for i in range(6)]
        d["action_range"] = [ac.min(0).tolist(), ac.max(0).tolist()]
        out[s] = d
    flags = [out[s]["frame0_closer_to_episode_max_than_min(moving dims)"] for s in out]
    flags = [f for f in flags if f is not None]
    out["verdict"] = ("1=aberto, valores MENORES=fechado (frame 0 no máximo do episódio; valores caem quando a ação comanda fechar). "
                      "Contraria a suposição 0=aberto do spec §1.1/§8.3" if flags and all(flags)
                      else "INCONCLUSIVO: ver frame0 vs min/max por dim")
    return out


# ---------------------------------------------------------------- c) base_command
def part_c(cols, ep_ids):
    ep = cols["episode_index"]
    cmd = cols[K["act_cmd"]]
    bp = cols[K["base_pose"]]
    rows = {"angle_z": [], "dyaw": [], "yaw_unw": [], "yaw_wrap": [], "wz_body": [], "vx": [], "vy": [], "vbx": [], "vby": [], "h": [], "z": []}
    per_ep_corr_rate, per_ep_corr_abs = [], []
    yaw_int, xy_int = [], []
    L = range(-5, 31)  # comando em t vs pose em t+lag
    pooled = {l: ([], []) for l in L}
    for e in ep_ids:
        i = np.flatnonzero(ep == e)
        q = bp[i, 3:7]
        yaw = Rotation.from_quat(q).as_euler("xyz")[:, 2]
        yu = np.unwrap(yaw)
        dyaw = uniform_filter1d(np.gradient(yu) * FPS, 5)
        R = Rotation.from_quat(q).as_matrix()
        v_w = uniform_filter1d(np.gradient(bp[i, :3], axis=0) * FPS, 5, axis=0)
        v_b = np.einsum("nji,nj->ni", R, v_w)  # R^T v_world
        c = cmd[i]
        yaw_int.append((c[:, 2].sum() / FPS, yu[-1] - yu[0]))
        xy_int.append((np.einsum("nij,nj->i", R[:, :2, :2], c[:, :2]) / FPS, bp[i[-1], :2] - bp[i[0], :2]))
        for l in L:
            n = len(i)
            a_, b_ = (c[:n - l, 2], dyaw[l:]) if l >= 0 else (c[-l:, 2], dyaw[:n + l])
            pooled[l][0].append(a_)
            pooled[l][1].append(b_)
        if c[:, 2].std() > 1e-6 and dyaw.std() > 1e-9:
            per_ep_corr_rate.append(np.corrcoef(c[:, 2], dyaw)[0, 1])
        if c[:, 2].std() > 1e-6 and yu.std() > 1e-9:
            per_ep_corr_abs.append(np.corrcoef(c[:, 2], yu - yu[0])[0, 1])
        rows["angle_z"].append(c[:, 2]); rows["dyaw"].append(dyaw); rows["yaw_unw"].append(yu - yu[0]); rows["yaw_wrap"].append(yaw)
        rows["wz_body"].append(cols[K["base_rot"]][i, 5]); rows["vx"].append(c[:, 0]); rows["vy"].append(c[:, 1])
        rows["vbx"].append(v_b[:, 0]); rows["vby"].append(v_b[:, 1]); rows["h"].append(c[:, 3]); rows["z"].append(bp[i, 2])
    r = {k: np.concatenate(v) for k, v in rows.items()}
    out = {"base_command_stats[vx,vy,angle_z,height]": {
        "min": cmd.min(0).tolist(), "max": cmd.max(0).tolist(), "mean": cmd.mean(0).tolist(), "std": cmd.std(0).tolist()}}
    lag_corr = {l: float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1]) for l, (a, b) in pooled.items()}
    lb = max(lag_corr, key=lambda k: abs(lag_corr[k]))
    a, b = np.concatenate(pooled[lb][0]), np.concatenate(pooled[lb][1])
    slope, icpt = np.polyfit(a, b, 1)
    out["angle_z"] = {
        "corr_with_dyaw_dt_by_lag(cmd_t vs rate_t+lag)": {str(k): v for k, v in lag_corr.items()},
        "best_lag_frames": lb, "corr_best_lag": lag_corr[lb], "ols_dyaw_dt_vs_angle_z": {"slope": float(slope), "intercept": float(icpt)},
        "corr_with_yaw_unwrapped_minus_yaw0_pooled": float(np.corrcoef(r["angle_z"], r["yaw_unw"])[0, 1]),
        "corr_with_yaw_wrapped_pooled": float(np.corrcoef(r["angle_z"], r["yaw_wrap"])[0, 1]),
        "per_episode_corr_with_rate_lag0_median": float(np.median(per_ep_corr_rate)) if per_ep_corr_rate else None,
        "per_episode_corr_with_abs_yaw_median": float(np.median(per_ep_corr_abs)) if per_ep_corr_abs else None,
        "n_episodes_with_nonconstant_angle_z": len(per_ep_corr_rate),
        "corr_with_state_base_rot_wz_lag0": float(np.corrcoef(r["angle_z"], r["wz_body"])[0, 1]),
        "yaw_rate_std_rad_s": float(r["dyaw"].std()), "angle_z_std": float(r["angle_z"].std()),
    }
    ratio_ok = abs(lag_corr[lb]) > 0.5 and 0.5 < abs(slope) < 2.0
    out["angle_z"]["verdict"] = ("yaw-rate (rad/s)" if ratio_ok and abs(lag_corr[lb]) > abs(out["angle_z"]["corr_with_yaw_unwrapped_minus_yaw0_pooled"])
                                  else "NÃO é yaw-rate em rad/s com evidência clara: ver correlações")
    yi = np.array(yaw_int)
    out["angle_z"]["max_abs_diff_vs_state_base_rot_wz"] = float(np.abs(r["angle_z"] - r["wz_body"]).max())
    out["angle_z"]["integral_test_per_episode(sum(angle_z)/fps vs delta yaw_unwrapped)"] = {
        "slope_through_origin": float((yi[:, 0] * yi[:, 1]).sum() / (yi[:, 0] ** 2).sum()), "corr": float(np.corrcoef(yi[:, 0], yi[:, 1])[0, 1])}
    xi = np.array(xy_int)
    ix, dx = xi[:, 0].reshape(-1), xi[:, 1].reshape(-1)
    out["vx_vy_integral_test(world xy displacement)"] = {
        "slope_through_origin": float((ix * dx).sum() / (ix ** 2).sum()), "corr": float(np.corrcoef(ix, dx)[0, 1])}
    for nm, c_, v_ in (("vx", r["vx"], r["vbx"]), ("vy", r["vy"], r["vby"])):
        s_, i_ = np.polyfit(c_, v_, 1)
        out[nm] = {"corr_with_body_frame_velocity_lag0": float(np.corrcoef(c_, v_)[0, 1]), "slope": float(s_), "intercept": float(i_)}
    out["height"] = {"corr_with_state_base_pose_z": float(np.corrcoef(r["h"], r["z"])[0, 1]) if r["h"].std() > 0 and r["z"].std() > 0 else None,
                     "max_abs_diff_vs_state_base_pose_z": float(np.abs(r["h"] - r["z"]).max()),
                     "state_base_pose_z_mean_min_max": [float(r["z"].mean()), float(r["z"].min()), float(r["z"].max())],
                     "range_m": [float(r["h"].min()), float(r["h"].max())]}
    return out


# ---------------------------------------------------------------- d) lag da ação
def part_d(cols, best):
    ep = cols["episode_index"]
    n = len(ep)
    out = {}
    wp, wrp = best["waist_perm_idx"], best["wrist_perm_idx"]
    for s in ("left", "right"):
        Ta, Ts = xyz_rpy_to_se3(cols[K[f"{s}_ee_act"]]), xyz_rpy_to_se3(cols[K[f"{s}_ee_base"]])
        res = {}
        for k in range(-3, 11):
            i = np.arange(n)
            j = i + k
            okm = (j >= 0) & (j < n)
            okm[okm] &= ep[j[okm]] == ep[i[okm]]
            i, j = i[okm], j[okm]
            pos = np.linalg.norm(Ta[i, :3, 3] - Ts[j, :3, 3], axis=-1) * 1e3
            rot = np.degrees(rotation_angle(Ta[i, :3, :3], Ts[j, :3, :3]))
            res[k] = {"pos_mm": summ(pos), "rot_deg": summ(rot)}
        kb = min(res, key=lambda k: res[k]["pos_mm"]["mean"])
        kr = min(res, key=lambda k: res[k]["rot_deg"]["mean"])
        # junta: action.arm(t) vs state.arm(t+k)
        jres = {}
        qa, qs = cols[K[f"{s}_arm_act"]], cols[K[f"{s}_arm"]]
        for k in range(-3, 11):
            i = np.arange(n); j = i + k
            okm = (j >= 0) & (j < n)
            okm[okm] &= ep[j[okm]] == ep[i[okm]]
            jres[k] = float(np.abs(qa[i[okm]] - qs[j[okm]]).mean())
        out[s] = {"by_lag": {str(k): v for k, v in res.items()}, "best_lag_pos": kb, "best_lag_rot": kr,
                  "best_lag_joint_space_arm": min(jres, key=jres.get), "joint_mae_by_lag_rad": {str(k): v for k, v in jres.items()}}
        # a ação de EE é FK das juntas-alvo? (hipótese vencedora de (a), E ajustado no estado)
        E = xyz_rpy_to_se3(np.array(best[s]["E_xyz_rpy"]))
        qa_ = cols[K[f"{s}_arm_act"]].astype(np.float64)
        qa_ = np.concatenate([qa_[:, :4], qa_[:, 4:7][:, list(wrp)]], axis=1)
        Tfk = fk(cols[K["act_waist"]].astype(np.float64)[:, list(wp)], qa_, s, base_link=best["base"], tip_link=best["tip"])
        pos_a, rot_a = residual(Tfk, Ta, E)
        out[s]["action_ee_vs_FK(action_joints)@E"] = {"pos_mm": summ(pos_a), "rot_deg": summ(rot_a)}
        # 1a ordem: s(t+1)-s(t) = alpha (a(t)-s(t))
        okm = np.arange(n - 1)
        okm = okm[ep[okm] == ep[okm + 1]]
        x = (qa[okm] - qs[okm]).reshape(-1)
        y = (qs[okm + 1] - qs[okm]).reshape(-1)
        alpha = float((x * y).sum() / (x * x).sum())
        out[s]["first_order_tracking_joint"] = {"alpha_per_frame": alpha, "r2": float(1 - ((y - alpha * x) ** 2).sum() / (y ** 2).sum())}
    out["verdict"] = {"best_lag_frames_pos": {s: out[s]["best_lag_pos"] for s in ("left", "right")}, "dt_s": 1 / FPS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("WLA_EXP", "/raid/user_marcospaulo/experiments/wla"), "f2_validation"))
    ap.add_argument("--n-episodes", type=int, default=20)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(K.values()))
    cols = P.load_wbt_columns(DATA_DIR, keys)
    ep_ids = np.unique(cols["episode_index"])
    valid_all = cols[K["valid"]][:, 0] > 0.5
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(ep_ids, args.n_episodes, replace=False))

    facts = {"dataset": str(DATA_DIR), "n_frames": int(len(valid_all)), "n_episodes": int(len(ep_ids)),
             "recomputed_ee_valid_fraction_all": float(valid_all.mean()), "selected_episodes": [int(e) for e in sel]}
    a, best = part_a(cols, sel, valid_all)
    facts["a_ee_frame"] = a
    facts["b_fig6d"] = part_b(cols, ep_ids)
    facts["c_base_command"] = part_c(cols, ep_ids)
    facts["d_action_lag"] = part_d(cols, best)
    json.dump(facts, open(out_dir / "wbt_facts.json", "w"), indent=1)

    print("== a) verdict gripper_base:", a["verdict_gripper_base"])
    for t, b in a["best_per_target"].items():
        print(f"   best[{t}]: base={b['base']} tip={b['tip']} waist={b['waist_cols_to_urdf(yaw,roll,pitch)']} wrist={b['wrist_cols_to_urdf(roll,pitch,yaw)']} score={b['score(<1 => VERIFIED)']:.3f}")
        for s in ("left", "right"):
            print(f"     {s}: held pos mean/p99 mm={b[s]['held_pos_mm']['mean']:.3f}/{b[s]['held_pos_mm']['p99']:.3f} rot mean/p99 deg={b[s]['held_rot_deg']['mean']:.3f}/{b[s]['held_rot_deg']['p99']:.3f} E={np.round(b[s]['E_xyz_rpy'], 4).tolist()}")
    print("== b)", facts["b_fig6d"]["verdict"])
    print("== c) angle_z:", facts["c_base_command"]["angle_z"]["verdict"], "| best lag", facts["c_base_command"]["angle_z"]["best_lag_frames"],
          "corr", round(facts["c_base_command"]["angle_z"]["corr_best_lag"], 3))
    print("== d)", facts["d_action_lag"]["verdict"])
    c = facts["c_base_command"]
    print("   angle_z integral:", c["angle_z"]["integral_test_per_episode(sum(angle_z)/fps vs delta yaw_unwrapped)"],
          "| maxdiff vs wz:", c["angle_z"]["max_abs_diff_vs_state_base_rot_wz"], "| height maxdiff vs z:", c["height"]["max_abs_diff_vs_state_base_pose_z"])
    print("   vxvy integral:", c["vx_vy_integral_test(world xy displacement)"])
    for s in ("left", "right"):
        d = facts["d_action_lag"][s]
        print(f"   {s}: FK(action joints)@E vs action ee: {d['action_ee_vs_FK(action_joints)@E']['pos_mm']['mean']:.2f} mm mean, {d['action_ee_vs_FK(action_joints)@E']['rot_deg']['mean']:.3f} deg; alpha={d['first_order_tracking_joint']}")


if __name__ == "__main__":
    main()
