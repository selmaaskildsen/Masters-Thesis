#!/usr/bin/env python3
"""
ROS 2 node: bygger en forhåndsdefinert rute fra flere vegreferanser i NVDB,
lager en samlet referansebane med cubic spline, publiserer som nav_msgs/Path,
og kan eksportere banen til GeoJSON i globale koordinater.

Nyhet:
- støtter route_reverse, slik at hver delstrekning kan snus etter henting
- støtter eksport av interpolert bane til GeoJSON før lokal transformasjon
"""

from __future__ import annotations

import json
import time
from pathlib import Path as FilePath
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


class NvdbRoutePublisher(Node):
    def __init__(self) -> None:
        super().__init__("nvdb_route_publisher")

        # -------------------------
        # Parametre
        # -------------------------
        self.declare_parameter("nvdb_base_url", "https://nvdbapiles.atlas.vegvesen.no")
        self.declare_parameter("x_client", "nvdb-path-publisher-ros2")
        self.declare_parameter("bearer_token", "")

        self.declare_parameter("route_segments", [""])
        self.declare_parameter("route_reverse", [False])

        self.declare_parameter("srid", 5973)
        self.declare_parameter("historisk", False)

        self.declare_parameter("sample_spacing_m", 0.5)
        self.declare_parameter("join_tolerance_m", 1.0)
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("overlap_tolerance_m", 1.0)
        self.declare_parameter("overlap_ratio_threshold", 0.7)

        self.declare_parameter("direction_weight", 2.0)
        self.declare_parameter("backward_dot_threshold", -0.2)

        self.declare_parameter("route_join_max_dist_m", 10.0)

        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("retry_initial_s", 5.0)
        self.declare_parameter("retry_max_s", 120.0)

        # GeoJSON-eksport
        self.declare_parameter("export_geojson", True)
        self.declare_parameter(
            "geojson_output_file",
            "/home/itk/ros2_ws/path_exports/reference_path.geojson"
        )

        self._nvdb_base_url: str = str(self.get_parameter("nvdb_base_url").value).rstrip("/")
        self._x_client: str = str(self.get_parameter("x_client").value).strip()
        self._bearer_token: str = str(self.get_parameter("bearer_token").value).strip()

        self._route_segments: List[str] = [
            s.strip() for s in self.get_parameter("route_segments").value if str(s).strip()
        ]
        self._route_reverse: List[bool] = list(self.get_parameter("route_reverse").value)

        self._srid: int = int(self.get_parameter("srid").value)
        self._historisk: bool = bool(self.get_parameter("historisk").value)

        self._sample_spacing_m: float = float(self.get_parameter("sample_spacing_m").value)
        self._join_tolerance_m: float = float(self.get_parameter("join_tolerance_m").value)
        self._frame_id: str = str(self.get_parameter("frame_id").value).strip() or "map"

        self._overlap_tolerance_m: float = float(self.get_parameter("overlap_tolerance_m").value)
        self._overlap_ratio_threshold: float = float(self.get_parameter("overlap_ratio_threshold").value)

        self._direction_weight: float = float(self.get_parameter("direction_weight").value)
        self._backward_dot_threshold: float = float(self.get_parameter("backward_dot_threshold").value)

        self._route_join_max_dist_m: float = float(self.get_parameter("route_join_max_dist_m").value)

        self._publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)
        self._retry_initial_s: float = float(self.get_parameter("retry_initial_s").value)
        self._retry_max_s: float = float(self.get_parameter("retry_max_s").value)

        self._export_geojson: bool = bool(self.get_parameter("export_geojson").value)
        self._geojson_output_file: str = str(self.get_parameter("geojson_output_file").value).strip()

        if not self._x_client:
            self._x_client = "nvdb-path-publisher-ros2"
            self.get_logger().warn(
                "Parameter x_client var tom. Setter til default 'nvdb-path-publisher-ros2'."
            )

        if not self._route_segments:
            raise RuntimeError("route_segments er tom. Oppgi minst én vegreferanse.")

        if len(self._route_reverse) == 1 and len(self._route_segments) > 1:
            self._route_reverse = self._route_reverse * len(self._route_segments)

        if len(self._route_reverse) != len(self._route_segments):
            raise RuntimeError(
                "route_reverse må ha samme lengde som route_segments, "
                "eller være én enkelt boolsk verdi."
            )

        if self._sample_spacing_m <= 0.0:
            self.get_logger().warn("sample_spacing_m <= 0. Setter til 0.5.")
            self._sample_spacing_m = 0.5

        if self._join_tolerance_m <= 0.0:
            self.get_logger().warn("join_tolerance_m <= 0. Setter til 1.0.")
            self._join_tolerance_m = 1.0

        if self._route_join_max_dist_m <= 0.0:
            self.get_logger().warn("route_join_max_dist_m <= 0. Setter til 10.0.")
            self._route_join_max_dist_m = 10.0

        if self._overlap_tolerance_m <= 0.0:
            self.get_logger().warn("overlap_tolerance_m <= 0. Setter til 1.0.")
            self._overlap_tolerance_m = 1.0

        if not (0.0 < self._overlap_ratio_threshold <= 1.0):
            self.get_logger().warn("overlap_ratio_threshold utenfor (0,1]. Setter til 0.7.")
            self._overlap_ratio_threshold = 0.7

        self._path_pub = self.create_publisher(Path, "/path", 10)

        self._path_msg: Optional[Path] = None
        self._next_retry_t: float = 0.0
        self._retry_s: float = self._retry_initial_s

        self._try_rebuild_route(reason="startup")

        period_s = 1.0 / max(self._publish_rate_hz, 0.1)
        self._timer = self.create_timer(period_s, self._on_timer)

    # -------------------------
    # Timer
    # -------------------------
    def _on_timer(self) -> None:
        now_ros = self.get_clock().now().to_msg()

        if self._path_msg is None:
            if time.monotonic() >= self._next_retry_t:
                self._try_rebuild_route(reason="retry")
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
    def _try_rebuild_route(self, reason: str) -> None:
        try:
            self.get_logger().info(f"Prøver å bygge rute (årsaken: {reason}).")

            route_parts: List[np.ndarray] = []

            for ref, reverse in zip(self._route_segments, self._route_reverse):
                self.get_logger().info(f"Henter og bygger delstrekning: {ref}")
                poly = self.build_polyline_for_reference(ref)

                if reverse:
                    self.get_logger().info(f"Reverserer delstrekning: {ref}")
                    poly = poly[::-1].copy()

                if poly.shape[0] < 2:
                    raise RuntimeError(f"For få punkter i delstrekning {ref}")

                route_parts.append(poly)

            full_route = self.stitch_route_parts(route_parts)

            if full_route.shape[0] < 3:
                raise RuntimeError("For få punkter i samlet rute etter stitching.")

            path_msg = self.build_path_msg_from_polyline(
                full_route,
                frame_id=self._frame_id,
                sample_spacing_m=self._sample_spacing_m,
            )

            self._path_msg = path_msg
            self._retry_s = self._retry_initial_s
            self._next_retry_t = 0.0

            self.get_logger().info(
                f"OK: Publiserer samlet rute med {len(path_msg.poses)} punkter på /path."
            )

        except Exception as e:
            self.get_logger().error(f"Klarte ikke å bygge rute: {e}")

            self._next_retry_t = time.monotonic() + self._retry_s
            self._retry_s = min(self._retry_s * 2.0, self._retry_max_s)

            self.get_logger().warn(f"Prøver igjen om ca. {self._retry_s:.1f}s (backoff).")

    # -------------------------
    # Bygg én delstrekning
    # -------------------------
    def build_polyline_for_reference(self, vegref: str) -> np.ndarray:
        segments = self.fetch_nvdb_segments_for_reference(vegref)
        if not segments:
            raise RuntimeError(f"Ingen segmenter funnet for {vegref}")

        polylines = self.parse_segments_to_polylines(segments)
        if not polylines:
            raise RuntimeError(f"Ingen gyldige geometri-segmenter for {vegref}")

        merged = self.sort_and_merge_polylines(
            polylines,
            join_tol=self._join_tolerance_m,
        )

        if merged.shape[0] < 2:
            raise RuntimeError(f"Klarte ikke å bygge polyline for {vegref}")

        return merged

    # -------------------------
    # NVDB
    # -------------------------
    def fetch_nvdb_segments_for_reference(self, vegref: str) -> List[JsonDict]:
        endpoint = f"{self._nvdb_base_url}/vegnett/api/v4/veglenkesekvenser/segmentert"

        headers = {"X-Client": self._x_client, "Accept": "application/json"}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"

        params: Dict[str, Any] = {
            "srid": self._srid,
            "antall": 1000,
            "inkluderAntall": "false",
            "vegsystemreferanse": [vegref],
        }

        if self._historisk:
            params["historisk"] = "true"

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
                next_url = None

            if next_url is None:
                break

        self.get_logger().info(f"NVDB: hentet {len(out)} segmenter totalt.")
        return out

    # -------------------------
    # Parsing
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
    # Robust merge
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

                    if dir_score < self._backward_dot_threshold:
                        continue

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

        return self._select_best_chain(chains)

    # -------------------------
    # Stitch route
    # -------------------------
    def stitch_route_parts(self, route_parts: Sequence[np.ndarray]) -> np.ndarray:
        if not route_parts:
            return np.zeros((0, 2), dtype=float)

        stitched: List[np.ndarray] = [route_parts[0].copy()]

        for idx, part in enumerate(route_parts[1:], start=2):
            prev = stitched[-1]
            prev_end = prev[-1]

            d_start = float(np.linalg.norm(part[0] - prev_end))
            d_end = float(np.linalg.norm(part[-1] - prev_end))

            part_oriented = part
            join_dist = d_start

            if d_end < d_start:
                part_oriented = part[::-1].copy()
                join_dist = d_end

            self.get_logger().info(
                f"Stitch: join-avstand til del {idx} er {join_dist:.2f} m."
            )

            if join_dist > self._route_join_max_dist_m:
                self.get_logger().warn(
                    f"Stitch: stor avstand ({join_dist:.2f} m) mellom delrutene. "
                    "Vurder mer spesifikke delstrekninger eller annen retning."
                )

            if np.allclose(stitched[-1][-1], part_oriented[0], atol=1e-6):
                stitched.append(part_oriented[1:])
            else:
                stitched.append(part_oriented)

        full = np.vstack(stitched)
        full = self._remove_duplicate_neighbors(full)
        return full

    # -------------------------
    # GeoJSON export
    # -------------------------
    def export_path_to_geojson(self, pts_xy: np.ndarray, output_file: str) -> None:
        """
        Eksporterer bane som GeoJSON LineString.
        Forutsetter at pts_xy er i globale projiserte koordinater.
        """
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "srid": self._srid
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(x), float(y)] for x, y in pts_xy]
                    },
                }
            ],
        }

        out_path = FilePath(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(feature_collection, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f"Eksporterte GeoJSON-bane til: {output_file}")

    # -------------------------
    # Build path
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

        # Bruk globale koordinater for spline og eksport
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

        # Interpolert bane i globale koordinater
        x_global = xs(s_samples)
        y_global = ys(s_samples)
        dx = xs(s_samples, 1)
        dy = ys(s_samples, 1)

        global_sampled_pts = np.column_stack((x_global, y_global))

        # Eksporter global bane til GeoJSON før lokal transformasjon
        if self._export_geojson and self._geojson_output_file:
            self.export_path_to_geojson(global_sampled_pts, self._geojson_output_file)

        # Lokal koordinatramme for ROS/RViz
        origin = global_sampled_pts[0].copy()
        local_pts = global_sampled_pts - origin

        yaw = np.arctan2(dy, dx)
        q = R.from_euler("z", yaw, degrees=False).as_quat()

        path = Path()
        path.header.frame_id = frame_id

        for i in range(len(s_samples)):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(local_pts[i, 0])
            pose.pose.position.y = float(local_pts[i, 1])
            pose.pose.position.z = 0.0

            pose.pose.orientation.x = float(q[i, 0])
            pose.pose.orientation.y = float(q[i, 1])
            pose.pose.orientation.z = float(q[i, 2])
            pose.pose.orientation.w = float(q[i, 3])

            path.poses.append(pose)

        return path

    # -------------------------
    # Helpers
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
        return np.concatenate(([0.0], np.cumsum(ds)))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = NvdbRoutePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()