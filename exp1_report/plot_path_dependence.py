#!/usr/bin/env python3
"""Generate charts for the feature path-dependence / hysteresis experiment.

Usage:
    python plot_path_dependence.py --input path_dependence_review.zip --output charts

`--input` may be either the review ZIP or an extracted directory containing
aggregate.json, train_config.json and the per-run *_metrics.json files.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import tempfile
import zipfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRICULUM_ORDER = ["joint", "semantic_first", "watermark_first"]
START_ORDER = ["base", "wm_recognition"]
DISPLAY_CURRICULUM = {
    "joint": "Joint",
    "semantic_first": "Semantic-first",
    "watermark_first": "Watermark-first",
}
DISPLAY_START = {"base": "Base", "wm_recognition": "WM-4x"}


def resolve_root(input_path: pathlib.Path, temp: tempfile.TemporaryDirectory):
    if input_path.is_dir():
        candidates = list(input_path.rglob("aggregate.json"))
    else:
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(temp.name)
        candidates = list(pathlib.Path(temp.name).rglob("aggregate.json"))
    if not candidates:
        raise FileNotFoundError("Could not find aggregate.json")
    root = candidates[0].parent
    return root


def save(fig, out: pathlib.Path, stem: str):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_table(agg):
    rows=[]
    for start in START_ORDER:
        for curriculum in CURRICULUM_ORDER:
            g=agg["groups"][start][curriculum]
            row={"start":start,"curriculum":curriculum}
            for key,val in g.items():
                if isinstance(val,dict) and "mean" in val:
                    row[key]=val["mean"]
                    row[key+"_sd"]=val["sd"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_grouped_metric(df, metric, sd_metric, title, ylabel, stem, out, zero_line=False):
    x=np.arange(len(CURRICULUM_ORDER),dtype=float)
    width=.36
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for i,start in enumerate(START_ORDER):
        d=df[df.start==start].set_index("curriculum").loc[CURRICULUM_ORDER]
        xpos=x+(i-.5)*width
        ax.bar(xpos,d[metric],width=width,label=DISPLAY_START[start],yerr=d[sd_metric],capsize=4)
    ax.set_xticks(x, [DISPLAY_CURRICULUM[c] for c in CURRICULUM_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if zero_line: ax.axhline(0,linewidth=.8)
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2)
    save(fig,out,stem)


def plot_hysteresis_per_seed(agg,out):
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    offsets={"base":-0.08,"wm_recognition":0.08}
    markers={"base":"o","wm_recognition":"s"}
    for start in START_ORDER:
        vals=agg["path_shifts"][start]["aggregate"]["delta_id_axis_score_wm_minus_sem"]["values"]
        seeds=[r["seed"] for r in agg["path_shifts"][start]["per_seed"]]
        xs=np.arange(len(seeds))+offsets[start]
        ax.scatter(xs,vals,label=DISPLAY_START[start],marker=markers[start],s=55)
        ax.plot(xs,vals,linewidth=1,alpha=.65)
    ax.axhline(0,linewidth=.8)
    ax.set_xticks(np.arange(3),[str(s) for s in agg["path_shifts"]["base"]["per_seed"] and [r["seed"] for r in agg["path_shifts"]["base"]["per_seed"]]])
    ax.set_xlabel("Optimizer seed")
    ax.set_ylabel("Δ axis score (watermark-first − semantic-first)")
    ax.set_title("Paired hysteresis shift by seed")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2)
    save(fig,out,"03_paired_hysteresis_by_seed")


def load_histories(root):
    rows=[]
    for p in root.glob("*_metrics.json"):
        if p.name=="metrics.json": continue
        r=json.load(open(p))
        if "history" not in r or "start_mode" not in r: continue
        for h in r["history"]:
            m=h["metrics"]
            rows.append({
                "start":r["start_mode"],"curriculum":r["curriculum"],"seed":r["seed"],
                "global_step":h["global_step"],"phase":h["phase"],"phase_step":h["phase_step"],
                "train_loss_since_last":h["train_loss_since_last"],
                "axis_score":m["axis_score_wm_minus_sem"],
                "wm_follow":m["crossed_watermark_follow_rate"],
                "watermark_effect":m["watermark_effect"],
                "semantic_effect":m["semantic_effect"],
            })
    return pd.DataFrame(rows)


def plot_wm_trajectory(hist,out):
    d=hist[hist.start=="wm_recognition"]
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    for curriculum in CURRICULUM_ORDER:
        x=d[d.curriculum==curriculum].groupby("global_step")["axis_score"].mean()
        s=d[d.curriculum==curriculum].groupby("global_step")["axis_score"].std()
        ax.plot(x.index,x.values,marker="o",label=DISPLAY_CURRICULUM[curriculum])
        ax.fill_between(x.index,x.values-s.fillna(0).values,x.values+s.fillna(0).values,alpha=.12)
    ax.axhline(0,linewidth=.8)
    ax.axvline(16,linestyle="--",linewidth=.9)
    ax.text(16, ax.get_ylim()[1], " warm phase ends", va="top", fontsize=9)
    ax.set_xlabel("Global optimizer step")
    ax.set_ylabel("Validation axis score (watermark − semantic)")
    ax.set_title("WM-4x feature preference during training")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig,out,"04_wm4x_validation_trajectory")


def plot_loss(hist,out):
    d=hist[(hist.start=="wm_recognition") & hist.train_loss_since_last.notna()]
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    for curriculum in CURRICULUM_ORDER:
        x=d[d.curriculum==curriculum].groupby("global_step")["train_loss_since_last"].mean()
        s=d[d.curriculum==curriculum].groupby("global_step")["train_loss_since_last"].std()
        ax.plot(x.index,x.values,marker="o",label=DISPLAY_CURRICULUM[curriculum])
        ax.fill_between(x.index,x.values-s.fillna(0).values,x.values+s.fillna(0).values,alpha=.12)
    ax.axvline(16,linestyle="--",linewidth=.9)
    ax.set_xlabel("Global optimizer step")
    ax.set_ylabel("Mean training loss since previous evaluation")
    ax.set_title("WM-4x curricula converge to similar training loss")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig,out,"05_wm4x_training_loss")


def plot_id_vs_ood(df,out):
    d=df[df.start=="wm_recognition"].set_index("curriculum").loc[CURRICULUM_ORDER]
    x=np.arange(len(CURRICULUM_ORDER),dtype=float); width=.36
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    ax.bar(x-width/2,d["id_axis_score_wm_minus_sem"],width=width,label="In-domain",yerr=d["id_axis_score_wm_minus_sem_sd"],capsize=4)
    ax.bar(x+width/2,d["ood_axis_score_wm_minus_sem"],width=width,label="Held-out categories",yerr=d["ood_axis_score_wm_minus_sem_sd"],capsize=4)
    ax.axhline(0,linewidth=.8)
    ax.set_xticks(x,[DISPLAY_CURRICULUM[c] for c in CURRICULUM_ORDER])
    ax.set_ylabel("Axis score (watermark − semantic)")
    ax.set_title("WM-4x path dependence is stronger in-domain")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2)
    save(fig,out,"06_id_vs_heldout_axis")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=pathlib.Path)
    ap.add_argument("--output",default=pathlib.Path("charts"),type=pathlib.Path)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root=resolve_root(args.input, type("T",(),{"name":td})())
        agg=json.load(open(root/"aggregate.json"))
        df=aggregate_table(agg)
        hist=load_histories(root)
        df.to_csv(args.output/"final_aggregate_derived.csv",index=False)
        hist.to_csv(args.output/"validation_history_derived.csv",index=False)
        path_rows=[]
        for start in START_ORDER:
            path_rows.extend({"start":start,**r} for r in agg["path_shifts"][start]["per_seed"])
        pd.DataFrame(path_rows).to_csv(args.output/"paired_hysteresis_derived.csv",index=False)

        plot_grouped_metric(df,"id_axis_score_wm_minus_sem","id_axis_score_wm_minus_sem_sd",
            "Final in-domain generalization axis","Axis score (watermark − semantic)","01_final_id_axis",args.output,True)
        plot_grouped_metric(df,"id_crossed_watermark_follow_rate","id_crossed_watermark_follow_rate_sd",
            "Final crossed-cue watermark following","Crossed watermark-follow rate","02_final_crossed_wm_follow",args.output,False)
        plot_hysteresis_per_seed(agg,args.output)
        plot_wm_trajectory(hist,args.output)
        plot_loss(hist,args.output)
        plot_id_vs_ood(df,args.output)
        print(f"Wrote charts and CSVs to {args.output}")

if __name__=="__main__":
    main()
