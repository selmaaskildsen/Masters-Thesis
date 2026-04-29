#!/usr/bin/env python3
"""
ROS 2 node: henter veggeometri fra NVDB (API Les V4), glatter bane med cubic spline,
og publiserer som nav_msgs/Path på /reference_path.

Forbedret merge-logikk:
- filtrerer bort sterkt overlappende segmenter
- bygger sammenhengende kjeder basert på geometrisk nærhet
- bruker også retningslikhet for å velge neste segment
- forkaster kandidater som peker tydelig bakover
- velger den lengste sammenhengende kjeden
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from shapely import wkt as shapely_wkt
from shapely.geometry import LineString


JsonDict = Dict[str, Any]


class NvdbPathPublisher(Node):
    def __init__(self) -> None:
        super().__init__("nvdb_path_publisher")

        # -------------------------
        # Parametre
        # -------------------------
        self.declare_parameter("nvdb_base_url", "https://nvdbapiles.atlas.vegvesen.no")
        self.declare_parameter("x_client", "nvdb-path-publisher-ros2")
        self.declare_parameter("bearer_token", "")

        self.declare_parameter("vegsystemreferanse", "")
        self.declare_parameter("kommune", "")
        self.declare_parameter("srid", 5973)
        self.declare_parameter("historisk", False)

        self.declare_parameter("sample_spacing_m", 0.5)
        self.declare_parameter("join_tolerance_m", 1.0)
        self.declare_parameter("frame_id", "map")

        # Overlap-filter
        self.declare_parameter("overlap_tolerance_m", 1.0)
        self.declare_parameter("overlap_ratio_threshold", 0.7)

        # Retningsbasert merge
        self.declare_parameter("direction_weight", 2.0)
        self.declare_parameter("backward_dot_threshold", -0.2)

        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("retry_initial_s", 5.0)
        self.declare_parameter("retry_max_s", 120.0)

        self._nvdb_base_url: str = str(self.get_parameter("nvdb_base_url").value).rstrip("/")
        self._x_client: str = str(self.get_parameter("x_client").value).strip()
        self._bearer_token: str = str(self.get_parameter("bearer_token").value).strip()

        self._vegsystemreferanse: str = str(self.get_parameter("vegsystemreferanse").value).strip()
        self._kommune: str = str(self.get_parameter("kommune").value).strip()
        self._srid: int = int(self.get_parameter("srid").value)
        self._historisk: bool = bool(self.get_parameter("historisk").value)

        self._sample_spacing_m: float = float(self.get_parameter("sample_spacing_m").value)
        self._join_tolerance_m: float = float(self.get_parameter("join_tolerance_m").value)
        self._frame_id: str = str(self.get_parameter("frame_id").value).strip() or "map"

        self._overlap_tolerance_m: float = float(self.get_parameter("overlap_tolerance_m").value)
        self._overlap_ratio_threshold: float = float(self.get_parameter("overlap_ratio_threshold").value)

        self._direction_weight: float = float(self.get_parameter("direction_weight").value)
        self._backward_dot_threshold: float = float(self.get_parameter("backward_dot_threshold").value)

        self._publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)
        self._retry_initial_s: float = float(self.get_parameter("retry_initial_s").value)
        self._retry_max_s: float = float(self.get_parameter("retry_max_s").value)

        if not self._x_client:
            self._x_client = "nvdb-path-publisher-ros2"
            self.get_logger().warn(
                "Parameter x_client var tom. Setter til default 'nvdb-path-publisher-ros2'."
            )

        if self._sample_spacing_m <= 0.0:
            self.get_logger().warn("sample_spacing_m <= 0. Setter til 0.5.")
            self._sample_spacing_m = 0.5

        if self._join_tolerance_m <= 0.0:
            self.get_logger().warn("join_tolerance_m <= 0. Setter til 1.0.")
            self._join_tolerance_m = 1.0

        if self._overlap_tolerance_m <= 0.0:
            self.get_logger().warn("overlap_tolerance_m <= 0. Setter til 1.0.")
            self._overlap_tolerance_m = 1.0

        if not (0.0 < self._overlap_ratio_threshold <= 1.0):
            self.get_logger().warn("overlap_ratio_threshold utenfor (0,1]. Setter til 0.7.")
            self._overlap_ratio_threshold = 0.7

        # Publisher
        self._path_pub = self.create_publisher(Path, "/reference_path", 10)

        # Intern state
        self._path_msg: Optional[Path] = None
        self._next_retry_t: float = 0.0
        self._retry_s: float = self._retry_initial_s

        # Bygg bane ved oppstart
        self._try_rebuild_path(reason="startup")

        # Timer
        period_s = 1.0 / max(self._publish_rate_hz, 0.1)
        self._timer = self.create_timer(period_s, self._on_timer)

    # -------------------------
    # Timer-callback
    # -------------------------
    def _on_timer(self) -> None:
        now_ros = self.get_clock().now().to_msg()

        if self._path_msg is None:
            if time.monotonic() >= self._next_retry_t:
                self._try_rebuild_path(reason="retry")
            return

        self._path_msg.header.stamp = now_ros
        self._path_msg.header.frame_id = self._frame_id
        for pose in self._path_msg.poses:
            pose.header.stamp = now_ros
            pose.header.frame_id = self._frame_id

        self._path_pub.publish(self._path_msg)

    # -------------------------
    # Hovedflyt
    # -------------------------
    def _try_rebuild_path(self, reason: str) -> None:
        try:
            self.get_logger().info(f"Prøver å bygge referansebane (årsaken: {reason}).")

            segments = self.fetch_nvdb_segments()
            if not segments:
                raise RuntimeError("NVDB ga 0 segmenter (tomt resultat).")

            polylines = self.parse_segments_to_polylines(segments)
            if not polylines:
                raise RuntimeError("Fant ingen gyldige geometri-segmenter (etter parsing).")

            merged = self.sort_and_merge_polylines(
                polylines,
                join_tol=self._join_tolerance_m,
            )

            if merged.shape[0] < 3:
                raise RuntimeError(
                    "For få unike punkter etter merge (trenger minst 3 for stabil spline/yaw)."
                )

            path_msg = self.build_path_msg_from_polyline(
                merged,
                frame_id=self._frame_id,
                sample_spacing_m=self._sample_spacing_m,
            )

            self._path_msg = path_msg
            self._retry_s = self._retry_initial_s
            self._next_retry_t = 0.0

            self.get_logger().info(
                f"OK: Publiserer {len(path_msg.poses)} punkter på /reference_path."
            )

        except Exception as e:
            self.get_logger().error(f"Klarte ikke å bygge bane: {e}")

            self._next_retry_t = time.monotonic() + self._retry_s
            self._retry_s = min(self._retry_s * 2.0, self._retry_max_s)

            self.get_logger().warn(f"Prøver igjen om ca. {self._retry_s:.1f}s (backoff).")

    # -------------------------
    # NVDB: HTTP + paginering
    # -------------------------
    def fetch_nvdb_segments(self) -> List[JsonDict]:
        endpoint = f"{self._nvdb_base_url}/vegnett/api/v4/veglenkesekvenser/segmentert"

        headers = {"X-Client": self._x_client, "Accept": "application/json"}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"

        params: Dict[str, Any] = {
            "srid": self._srid,
            "antall": 1000,
            "inkluderAntall": "false",
        }

        if self._historisk:
            params["historisk"] = "true"

        kommune_list = self._parse_int_list(self._kommune)
        if kommune_list:
            params["kommune"] = kommune_list

        vsr_list = self._parse_str_list(self._vegsystemreferanse)
        if vsr_list:
            params["vegsystemreferanse"] = vsr_list

        if not kommune_list and not vsr_list:
            self.get_logger().warn(
                "Ingen filtre satt (kommune eller vegsystemreferanse). "
                "Dette kan gi enormt resultatsett. Sett minst ett filter."
            )

        return self._paginate_nvdb(endpoint, headers=headers, params=params)

    def _paginate_nvdb(
        self,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any]
    ) -> List[JsonDict]:
        out: List[JsonDict] = []
        next_url: Optional[str] = url
        first = True
        max_pages = 200

        for _ in range(max_pages):
            if next_url is None:
                break

            try:
                if first:
                    r = requests.get(next_url, headers=headers, params=params, timeout=(5.0, 30.0))
                    first = False
                else:
                    r = requests.get(next_url, headers=headers, timeout=(5.0, 30.0))

                if r.status_code >= 400:
                    try:
                        payload = r.json()
                        title = payload.get("title", "")
                        detail = payload.get("detail", "")
                        raise RuntimeError(f"HTTP {r.status_code}: {title} {detail}".strip())
                    except ValueError:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

                data = r.json()

            except requests.RequestException as e:
                raise RuntimeError(f"HTTP-feil mot NVDB: {e}") from e
            except ValueError as e:
                raise RuntimeError(f"Ugyldig JSON fra NVDB: {e}") from e

            if isinstance(data, dict):
                objects = data.get("objekter", [])
                if isinstance(objects, list):
                    out.extend([o for o in objects if isinstance(o, dict)])
                else:
                    self.get_logger().warn("Uventet format: 'objekter' var ikke en liste.")

                next_url = None
                meta = data.get("metadata", {})
                if isinstance(meta, dict):
                    nxt = meta.get("neste", {})
                    if isinstance(nxt, dict):
                        href = nxt.get("href")
                        if isinstance(href, str) and href:
                            next_url = href if href != r.url else None

            elif isinstance(data, list):
                out.extend([o for o in data if isinstance(o, dict)])
                next_url = None
            else:
                self.get_logger().warn("Uventet JSON-format fra NVDB.")
                next_url = None

            if next_url is None:
                break

        self.get_logger().info(f"NVDB: hentet {len(out)} segmenter totalt.")
        return out

    # -------------------------
    # Parsing: JSON -> polylines
    # -------------------------
    def parse_segments_to_polylines(
        self,
        segments: Sequence[JsonDict]
    ) -> List[Tuple[float, np.ndarray]]:
        out: List[Tuple[float, np.ndarray]] = []
        skipped = 0

        for seg in segments:
            wkt_str = self._extract_wkt(seg)
            if not wkt_str:
                skipped += 1
                continue

            coords_xy = self._wkt_to_xy(wkt_str)
            if coords_xy is None or coords_xy.shape[0] < 2:
                skipped += 1
                continue

            coords_xy = self._remove_duplicate_neighbors(coords_xy)

            if coords_xy.shape[0] < 2:
                skipped += 1
                continue

            sort_key = self._segment_sort_key(seg)
            out.append((sort_key, coords_xy))

        if skipped > 0:
            self.get_logger().warn(
                f"Parsing: hoppet over {skipped} segmenter pga. manglende/ugyldig geometri."
            )

        return out

    def _extract_wkt(self, seg: JsonDict) -> str:
        g = seg.get("geometri")
        if isinstance(g, dict):
            w = g.get("wkt")
            return w if isinstance(w, str) else ""
        if isinstance(g, str):
            return g
        return ""

    def _wkt_to_xy(self, wkt_str: str) -> Optional[np.ndarray]:
        try:
            geom = shapely_wkt.loads(wkt_str)
        except Exception:
            return None

        if not hasattr(geom, "coords"):
            return None

        coords = np.asarray(list(geom.coords), dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            return None

        return coords[:, :2].copy()

    # -------------------------
    # Sorteringsnøkkel
    # -------------------------
    def _segment_sort_key(self, seg: JsonDict) -> float:
        vsr = seg.get("vegsystemreferanse", {})
        if isinstance(vsr, dict):
            strek = vsr.get("strekning", {})
            if isinstance(strek, dict):
                fm = strek.get("fra_meter")
                tm = strek.get("til_meter")
                if isinstance(fm, (int, float)) and isinstance(tm, (int, float)):
                    return float(min(fm, tm))

        sp = seg.get("startposisjon")
        if isinstance(sp, (int, float)):
            return float(sp)

        return 0.0

    # -------------------------
    # Merge-logikk
    # -------------------------
    def _polyline_length(self, pts: np.ndarray) -> float:
        if pts.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _unit_direction(self, pts: np.ndarray, from_start: bool = True) -> Optional[np.ndarray]:
        if pts.shape[0] < 2:
            return None

        if from_start:
            v = pts[1] - pts[0]
        else:
            v = pts[-1] - pts[-2]

        n = np.linalg.norm(v)
        if n <= 1e-9:
            return None

        return v / n

    def _are_polylines_overlapping(
        self,
        pts_a: np.ndarray,
        pts_b: np.ndarray,
        overlap_tol: float = 1.0,
        overlap_ratio_thresh: float = 0.7,
    ) -> bool:
        if pts_a.shape[0] < 2 or pts_b.shape[0] < 2:
            return False

        try:
            line_a = LineString(pts_a)
            line_b = LineString(pts_b)
        except Exception:
            return False

        if line_a.length <= 1e-6 or line_b.length <= 1e-6:
            return False

        try:
            inter_len = line_a.buffer(overlap_tol, cap_style=2).intersection(line_b).length
        except Exception:
            return False

        min_len = min(line_a.length, line_b.length)
        if min_len <= 1e-6:
            return False

        overlap_ratio = inter_len / min_len
        return overlap_ratio >= overlap_ratio_thresh

    def _filter_overlapping_polylines(
        self,
        polylines: Sequence[Tuple[float, np.ndarray]],
        overlap_tol: float = 1.0,
        overlap_ratio_thresh: float = 0.7,
    ) -> List[Tuple[float, np.ndarray]]:
        kept: List[Tuple[float, np.ndarray]] = []
        removed = 0

        for sort_key, pts in polylines:
            is_duplicate = False
            for _, kept_pts in kept:
                if self._are_polylines_overlapping(
                    pts,
                    kept_pts,
                    overlap_tol=overlap_tol,
                    overlap_ratio_thresh=overlap_ratio_thresh,
                ):
                    is_duplicate = True
                    removed += 1
                    break

            if not is_duplicate:
                kept.append((sort_key, pts))

        if removed > 0:
            self.get_logger().info(
                f"Overlap-filter: fjernet {removed} sterkt overlappende segment(er)."
            )

        return kept

    def _build_connected_chains(
        self,
        polylines: Sequence[Tuple[float, np.ndarray]],
        join_tol: float,
    ) -> List[np.ndarray]:
        """
        Bygger én eller flere sammenhengende kjeder.
        Velger neste segment basert på både nærhet og retningslikhet.
        """
        if not polylines:
            return []

        unused = [(k, pts.copy()) for k, pts in sorted(polylines, key=lambda x: x[0])]
        chains: List[np.ndarray] = []

        while unused:
            _, current = unused.pop(0)
            chain_parts = [current]
            extended = True

            while extended and unused:
                extended = False
                current_end = chain_parts[-1][-1]
                prev_dir = self._unit_direction(chain_parts[-1], from_start=False)

                best_idx = None
                best_pts = None
                best_score = float("inf")

                for i, (_, cand) in enumerate(unused):
                    d_start = float(np.linalg.norm(cand[0] - current_end))
                    d_end = float(np.linalg.norm(cand[-1] - current_end))

                    cand_oriented = cand
                    join_dist = d_start

                    if d_end < d_start:
                        cand_oriented = cand[::-1].copy()
                        join_dist = d_end

                    if join_dist > join_tol:
                        continue

                    cand_dir = self._unit_direction(cand_oriented, from_start=True)

                    dir_score = 0.0
                    if prev_dir is not None and cand_dir is not None:
                        dir_score = float(np.dot(prev_dir, cand_dir))

                    # Forkast kandidater som peker tydelig bakover
                    if dir_score < self._backward_dot_threshold:
                        continue

                    # Lav score er best
                    score = join_dist - self._direction_weight * dir_score

                    if score < best_score:
                        best_score = score
                        best_idx = i
                        best_pts = cand_oriented

                if best_idx is not None and best_pts is not None:
                    if np.allclose(chain_parts[-1][-1], best_pts[0], atol=1e-6):
                        chain_parts.append(best_pts[1:])
                    else:
                        chain_parts.append(best_pts)

                    unused.pop(best_idx)
                    extended = True

            chain = np.vstack(chain_parts)
            chain = self._remove_duplicate_neighbors(chain)
            chains.append(chain)

        return chains

    def _select_best_chain(self, chains: Sequence[np.ndarray]) -> np.ndarray:
        if not chains:
            return np.zeros((0, 2), dtype=float)
        return max(chains, key=self._polyline_length)

    def sort_and_merge_polylines(
        self,
        polylines: Sequence[Tuple[float, np.ndarray]],
        join_tol: float
    ) -> np.ndarray:
        if not polylines:
            return np.zeros((0, 2), dtype=float)

        polys = sorted(polylines, key=lambda x: x[0])

        polys = self._filter_overlapping_polylines(
            polys,
            overlap_tol=self._overlap_tolerance_m,
            overlap_ratio_thresh=self._overlap_ratio_threshold,
        )

        chains = self._build_connected_chains(polys, join_tol=join_tol)

        if not chains:
            return np.zeros((0, 2), dtype=float)

        if len(chains) > 1:
            chain_lengths = [self._polyline_length(c) for c in chains]
            self.get_logger().warn(
                f"Fant {len(chains)} separate kjeder. "
                f"Velger lengste kjede ({max(chain_lengths):.1f} m)."
            )

        best_chain = self._select_best_chain(chains)
        return best_chain

    # -------------------------
    # Spline + Path message
    # -------------------------
    def build_path_msg_from_polyline(
        self,
        pts_xy: np.ndarray,
        frame_id: str,
        sample_spacing_m: float
    ) -> Path:
        pts_xy = self._remove_duplicate_neighbors(pts_xy)
        if pts_xy.shape[0] < 3:
            raise RuntimeError("For få punkter til spline (etter duplikatfjerning).")

        # Lokal koordinatramme
        origin = pts_xy[0].copy()
        pts_xy = pts_xy - origin

        s = self._cumulative_arc_length(pts_xy)
        if s.shape[0] != pts_xy.shape[0]:
            raise RuntimeError("Intern feil: s og punkter har ulik lengde.")

        if s[-1] <= 0.0:
            raise RuntimeError("Total banlengde er 0 (alle punkter identiske?).")

        if not np.all(np.diff(s) > 0):
            raise RuntimeError("s er ikke strengt voksende (for mange like punkter).")

        xs = CubicSpline(s, pts_xy[:, 0])
        ys = CubicSpline(s, pts_xy[:, 1])

        s_samples = np.arange(0.0, float(s[-1]), float(sample_spacing_m), dtype=float)
        if s_samples.size == 0 or s_samples[-1] < s[-1]:
            s_samples = np.append(s_samples, float(s[-1]))

        x = xs(s_samples)
        y = ys(s_samples)
        dx = xs(s_samples, 1)
        dy = ys(s_samples, 1)

        yaw = np.arctan2(dy, dx)
        q = R.from_euler("z", yaw, degrees=False).as_quat()

        path = Path()
        path.header.frame_id = frame_id

        for i in range(len(s_samples)):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(x[i])
            pose.pose.position.y = float(y[i])
            pose.pose.position.z = 0.0

            pose.pose.orientation.x = float(q[i, 0])
            pose.pose.orientation.y = float(q[i, 1])
            pose.pose.orientation.z = float(q[i, 2])
            pose.pose.orientation.w = float(q[i, 3])

            path.poses.append(pose)

        return path

    # -------------------------
    # Hjelpefunksjoner
    # -------------------------
    def _remove_duplicate_neighbors(self, pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        if pts.shape[0] < 2:
            return pts
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.hstack(([True], diffs > eps))
        return pts[keep]

    def _cumulative_arc_length(self, pts: np.ndarray) -> np.ndarray:
        deltas = np.diff(pts, axis=0)
        ds = np.linalg.norm(deltas, axis=1)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        return s

    def _parse_int_list(self, s: str) -> List[int]:
        if not s:
            return []
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                self.get_logger().warn(f"Ugyldig heltall i liste: '{part}'")
        return out

    def _parse_str_list(self, s: str) -> List[str]:
        if not s:
            return []
        return [p.strip() for p in s.split(",") if p.strip()]


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = NvdbPathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()