"""Project file: load/save, person bookkeeping, merge/split/rename/recluster edits.

Layout of a project directory:
    project.json        persons, tracks (per-frame boxes), track->person mapping, confidences
    embeddings.npz      per-track ArcFace embeddings (needed for edits / re-clustering)
    crops/              candidate face crops per track (thumbnail source)
    thumbs/person_XXX/  3-5 face thumbnails per person (regenerated after every edit)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .clustering import cluster_tracks, track_descriptor

SCHEMA_VERSION = 1


def default_project_dir(video: Path) -> Path:
    video = Path(video)
    return video.with_name(video.stem + "_zoria")


class Project:
    def __init__(self, root: Path, data: dict, emb: dict[str, np.ndarray]):
        self.root = Path(root)
        self.data = data
        self.emb = emb

    # ---------- io ----------
    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root)
        if root.is_file():
            root = root.parent
        data = json.loads((root / "project.json").read_text(encoding="utf-8"))
        emb = {}
        npz = root / "embeddings.npz"
        if npz.is_file():
            with np.load(npz) as z:
                emb = {k: z[k] for k in z.files}
        return cls(root, data, emb)

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # atomic write so a crashed UI call never leaves half a JSON
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, separators=(",", ":"))
        Path(tmp).replace(self.root / "project.json")
        if self.emb:
            np.savez_compressed(self.root / "embeddings.npz", **self.emb)

    # ---------- helpers ----------
    @property
    def tracks(self) -> list[dict]:
        return self.data["tracks"]

    @property
    def persons(self) -> list[dict]:
        return self.data["persons"]

    def track(self, tid: int) -> dict:
        for t in self.tracks:
            if t["id"] == tid:
                return t
        raise KeyError(f"track {tid} not found")

    def person(self, pid: int) -> dict:
        for p in self.persons:
            if p["id"] == pid:
                return p
        raise KeyError(f"person {pid} not found")

    def track_embs(self, tid: int):
        e = self.emb.get(f"t{tid}")
        q = self.emb.get(f"t{tid}_q")
        return e, q

    # ---------- person bookkeeping ----------
    def rebuild_persons(self, keep_names: dict[int, str] | None = None, renumber: bool = False):
        """Recompute person stats, centroids, per-track/per-detection confidence, thumbnails."""
        by_pid: dict[int, list[dict]] = {}
        for t in self.tracks:
            by_pid.setdefault(t["person_id"], []).append(t)
        old = {p["id"]: p for p in self.persons}
        if renumber:
            order = sorted(by_pid, key=lambda pid: -sum(len(t["frames"]) for t in by_pid[pid]))
            remap = {pid: i + 1 for i, pid in enumerate(order)}
            for t in self.tracks:
                t["person_id"] = remap[t["person_id"]]
            old = {remap[k]: v for k, v in old.items() if k in remap}
            if keep_names:
                keep_names = {remap.get(k, k): v for k, v in keep_names.items()}
            by_pid = {remap[k]: v for k, v in by_pid.items()}
        persons = []
        for pid in sorted(by_pid):
            ts = by_pid[pid]
            all_e, all_q = [], []
            for t in ts:
                e, q = self.track_embs(t["id"])
                if e is not None and len(e):
                    all_e.append(e); all_q.append(q)
            centroid = None
            if all_e:
                centroid = track_descriptor(np.concatenate(all_e), np.concatenate(all_q))
                self.emb[f"p{pid}"] = centroid
            else:
                self.emb.pop(f"p{pid}", None)
            sims_track = []
            for t in ts:
                e, q = self.track_embs(t["id"])
                if centroid is not None and e is not None and len(e):
                    d = track_descriptor(e, q)
                    # leave-one-out: compare the track with the person built from its OTHER tracks,
                    # so a wrongly attached track cannot vouch for itself
                    others = [self.track_embs(o["id"]) for o in ts if o is not t]
                    others = [(oe, oq) for oe, oq in others if oe is not None and len(oe)]
                    ref = centroid
                    if others:
                        ref = track_descriptor(np.concatenate([o[0] for o in others]),
                                               np.concatenate([o[1] for o in others]))
                    t["confidence"] = round(float(d @ ref), 4)
                    s = (e @ centroid).tolist()
                    ef = self.emb.get(f"t{t['id']}_f")
                    fmap = {int(f): round(float(v), 4) for f, v in zip(ef, s)} if ef is not None else {}
                    t["det_sims"] = [fmap.get(f) for f in t["frames"]]
                    sims_track.append(t["confidence"])
                else:
                    t["confidence"] = 0.0
                    t["det_sims"] = [None] * len(t["frames"])
            n_emb = int(sum(len(x) for x in all_e))
            frames = sorted(f for t in ts for f in t["frames"])
            prev = old.get(pid, {})
            name = (keep_names or {}).get(pid) or prev.get("name") or f"Person {pid}"
            persons.append({
                "id": pid,
                "name": name,
                "track_ids": sorted(t["id"] for t in ts),
                "num_detections": len(frames),
                "num_embeddings": n_emb,
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "confidence": round(float(np.mean(sims_track)), 4) if sims_track else 0.0,
                "low_confidence": n_emb < 3 or len(frames) < 4,
                "thumbnails": [],
            })
        self.data["persons"] = persons
        for k in [k for k in self.emb if k.startswith("p") and int(k[1:]) not in by_pid]:
            del self.emb[k]
        self.make_thumbnails()

    def make_thumbnails(self, per_person: int = 5):
        thumbs = self.root / "thumbs"
        if thumbs.exists():
            shutil.rmtree(thumbs)
        crops = self.root / "crops"
        for p in self.persons:
            cands = []
            for tid in p["track_ids"]:
                t = self.track(tid)
                for c in t.get("crops", []):
                    cands.append((c["quality"], c["frame"], tid, c["file"]))
            cands.sort(key=lambda x: -x[0])
            chosen, used_frames = [], []
            # prefer diversity in time: skip crops too close to an already chosen one
            min_sep = max(1, int(self.data["video"]["fps_float"] * 0.75))
            for c in cands:
                if all(abs(c[1] - f) >= min_sep for f in used_frames):
                    chosen.append(c); used_frames.append(c[1])
                if len(chosen) >= per_person:
                    break
            for c in cands:          # top up to at least 3 (or 5 if available) when diversity was too strict
                if len(chosen) >= min(per_person, len(cands)):
                    break
                if c not in chosen:
                    chosen.append(c)
            d = thumbs / f"person_{p['id']:03d}"
            d.mkdir(parents=True, exist_ok=True)
            rels = []
            for i, c in enumerate(chosen):
                src = crops / c[3]
                if src.is_file():
                    dst = d / f"{i + 1:02d}_f{c[1]:06d}.jpg"
                    shutil.copyfile(src, dst)
                    rels.append(dst.relative_to(self.root).as_posix())
            p["thumbnails"] = rels

    # ---------- edits (used by CLI and later by the UI) ----------
    def _next_pid(self) -> int:
        return max([p["id"] for p in self.persons] + [0]) + 1

    def merge(self, pids: list[int]) -> int:
        """Merge persons into the first id. Returns the surviving id."""
        if len(pids) < 2:
            raise ValueError("merge needs at least two person ids")
        target = pids[0]
        for pid in pids:
            self.person(pid)
        names = {p["id"]: p["name"] for p in self.persons}
        for t in self.tracks:
            if t["person_id"] in pids:
                t["person_id"] = target
        self.rebuild_persons(keep_names={target: names[target]})
        self._log("merge", {"persons": pids, "into": target})
        return target

    def split(self, track_ids: list[int]) -> int:
        """Move the given tracks into a new person. Returns the new person id."""
        for tid in track_ids:
            self.track(tid)
        new = self._next_pid()
        for tid in track_ids:
            self.track(tid)["person_id"] = new
        self.rebuild_persons()
        self._log("split", {"tracks": track_ids, "new_person": new})
        return new

    def assign(self, track_ids: list[int], pid: int):
        self.person(pid)
        for tid in track_ids:
            self.track(tid)["person_id"] = pid
        self.rebuild_persons()
        self._log("assign", {"tracks": track_ids, "person": pid})

    def rename(self, pid: int, name: str):
        self.person(pid)["name"] = name
        self._log("rename", {"person": pid, "name": name})

    def recluster(self, sim_thresh: float):
        tids = [t["id"] for t in self.tracks]
        descs = []
        for tid in tids:
            e, q = self.track_embs(tid)
            descs.append(track_descriptor(e, q) if e is not None and len(e) else None)
        fsets = [set(self.track(t)["frames"]) for t in tids]
        # remember user-given names so they survive re-clustering (majority of detections wins)
        named = {p["id"]: p["name"] for p in self.persons if not p["name"].startswith("Person ")}
        old_pid = {t["id"]: t["person_id"] for t in self.tracks}
        labels = cluster_tracks(descs, fsets, sim_thresh)
        for tid, lab in zip(tids, labels):
            self.track(tid)["person_id"] = lab + 1
        # manual tracks have no embedding: they follow the tracks of the person they were drawn for
        for t in self.tracks:
            if t.get("manual"):
                votes: dict[int, int] = {}
                for o in self.tracks:
                    if not o.get("manual") and old_pid.get(o["id"]) == old_pid[t["id"]]:
                        votes[o["person_id"]] = votes.get(o["person_id"], 0) + len(o["frames"])
                if votes:
                    t["person_id"] = max(votes, key=votes.get)
        self.data["persons"] = []
        self.data["analysis"]["params"]["cluster_thresh"] = sim_thresh
        self.rebuild_persons(renumber=True)
        if self.data.get("bodies"):
            from .bodies import adopt_orphan_tracks
            adopt_orphan_tracks(self)
        if named:
            for p in self.persons:
                votes: dict[int, int] = {}
                for tid in p["track_ids"]:
                    op = old_pid.get(tid)
                    if op in named:
                        votes[op] = votes.get(op, 0) + len(self.track(tid)["frames"])
                if votes:
                    p["name"] = named[max(votes, key=votes.get)]
        self._log("recluster", {"threshold": sim_thresh})

    def _log(self, op, args):
        self.data.setdefault("edits", []).append({"op": op, **args})

    def summary(self) -> list[dict]:
        fps = self.data["video"]["fps_float"]
        out = []
        for p in self.persons:
            out.append({k: p[k] for k in ("id", "name", "track_ids", "num_detections", "confidence",
                                          "low_confidence", "thumbnails")} |
                       {"seconds_visible": round(p["num_detections"] * self.data["analysis"]["params"]["every"] / fps, 2)})
        return out
