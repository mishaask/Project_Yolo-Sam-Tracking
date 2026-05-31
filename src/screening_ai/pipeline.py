from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import cv2

from screening_ai.appearance import configure_deep_person_reid, get_deep_person_reid_status
from screening_ai.association import assign_bag_owners, assign_object_owners, owner_link_lines
from screening_ai.deep_reid import DeepPersonReID
from screening_ai.detector import Detection, YoloDetector
from screening_ai.memory import MemoryBank, bbox_iou_xyxy
from screening_ai.privacy import FacePrivacyFilter
from screening_ai.risk import Event, RiskCluster, RiskClusterManager, RiskConfirmationFilter, RiskEngine
from screening_ai.roi_search import NestedRoiObjectSearch, RoiSearchConfig
from screening_ai.segmenter import SamBoxSegmenter, SegmentationResult
from screening_ai.utils import ensure_parent, load_yaml, parse_source
from screening_ai.visualization import (
    draw_detection,
    draw_event_banner,
    draw_fps,
    draw_owner_links,
    draw_track_trails,
)


class ScreeningPipeline:
    def __init__(
        self,
        yolo_weights: str,
        tracker_config: str = "configs/botsort_reid.yaml",
        classes_config: str = "configs/classes.yaml",
        risk_config: str = "configs/risk_config.yaml",
        memory_config: str = "configs/tracking_memory.yaml",
        target_classes: Optional[set[str]] = None,
        secondary_weights: Optional[str] = None,
        secondary_target_classes: Optional[set[str]] = None,
        secondary_conf: Optional[float] = None,
        secondary_imgsz: Optional[int] = None,
        secondary_iou: Optional[float] = None,
        secondary_max_det: Optional[int] = None,
        risk_smoothing_enabled: Optional[bool] = None,
        weapon_confirm_window: Optional[int] = None,
        weapon_confirm_min_hits: Optional[int] = None,
        weapon_confirm_match_iou: Optional[float] = None,
        weapon_confirm_match_center_px: Optional[float] = None,
        risk_cluster_enabled: Optional[bool] = None,
        risk_cluster_match_iou: Optional[float] = None,
        risk_cluster_match_center_px: Optional[float] = None,
        risk_cluster_ttl_frames: Optional[int] = None,
        risk_confirm_linked_frames: Optional[int] = None,
        risk_warning_cooldown_frames: Optional[int] = None,
        risk_repeat_warning_window_frames: Optional[int] = None,
        risk_repeat_warning_count: Optional[int] = None,
        sam_weights: str = "sam2_b.pt",
        enable_sam: bool = False,
        sam_every_n_frames: int = 5,
        sam_max_objects: int = 3,
        sam_classes: Optional[set[str]] = None,
        reuse_last_masks: bool = True,
        prefer_sam_masks: bool = False,
        sam_tracking_classes: Optional[set[str]] = None,
        conf: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
        draw_trails: bool = True,
        draw_links: bool = True,
        blur_faces: bool = False,
        pause_recording_on_face: bool = False,
        enable_roi_search: Optional[bool] = None,
        roi_every_n_frames: Optional[int] = None,
        roi_confidence: Optional[float] = None,
        roi_imgsz: Optional[int] = None,
        roi_max_parent_rois: Optional[int] = None,
        label_scale: float = 0.48,
        label_thickness: int = 1,
        event_overlay_enabled: bool = True,
        event_overlay_position: str = "bottom-right",
        event_overlay_max_lines: int = 5,
        event_overlay_ttl_frames: int = 120,
        event_overlay_scale: float = 0.46,
        debug_overlay_enabled: bool = True,
        debug_overlay_position: str = "bottom-left",
        debug_overlay_max_lines: int = 4,
        debug_overlay_ttl_frames: int = 150,
        debug_overlay_scale: float = 0.42,
    ) -> None:
        classes = load_yaml(classes_config)
        risk_cfg = load_yaml(risk_config)
        memory_cfg = load_yaml(memory_config)

        self.person_classes = set(classes.get("person_classes", ["person"]))
        self.bag_classes = set(classes.get("bag_classes", []))
        self.risk_classes = set(classes.get("risk_classes", []))

        # Keep only project-relevant classes. This prevents random COCO classes
        # such as cat/carrot/tie/teddy bear from entering tracking, SAM, CSV,
        # events, and visualization. A custom model can add names here later.
        cfg_target_classes = memory_cfg.get("target_classes", None)
        self.target_classes = self._normalize_class_set(target_classes if target_classes is not None else cfg_target_classes)

        association_cfg = risk_cfg.get("association", {})
        unattended_cfg = risk_cfg.get("unattended_bag", {})

        self.max_owner_distance_px = float(association_cfg.get("max_owner_distance_px", 180.0))
        self.min_owner_score = float(association_cfg.get("min_owner_score", 0.45))
        self.motion_window_frames = int(association_cfg.get("motion_window_frames", 20))
        self.owner_score_decay = float(association_cfg.get("score_decay", 0.98))
        self.owner_score_gain = float(association_cfg.get("score_gain", 0.12))
        self.owner_min_contact_frames = int(association_cfg.get("min_contact_frames", 45))
        self.owner_contact_distance_px = float(association_cfg.get("contact_distance_px", 150.0))
        self.owner_separation_distance_px = float(association_cfg.get("separation_distance_px", 260.0))
        self.owner_link_display_max_missing_frames = int(association_cfg.get("owner_link_display_max_missing_frames", 3))
        self.owner_link_display_min_strength = float(association_cfg.get("owner_link_display_min_strength", 0.60))

        risk_link_cfg = risk_cfg.get("risk_object_link", {}) or {}
        self.risk_owner_max_distance_px = float(risk_link_cfg.get("max_owner_distance_px", 230.0))
        self.risk_owner_min_score = float(risk_link_cfg.get("min_owner_score", 0.16))
        self.risk_owner_score_gain = float(risk_link_cfg.get("score_gain", 0.70))
        self.risk_owner_score_decay = float(risk_link_cfg.get("score_decay", 0.98))
        self.risk_owner_min_contact_frames = int(risk_link_cfg.get("min_contact_frames", 1))
        self.risk_owner_contact_distance_px = float(risk_link_cfg.get("contact_distance_px", 230.0))
        self.risk_owner_separation_distance_px = float(risk_link_cfg.get("separation_distance_px", 320.0))

        self.trail_length = int(memory_cfg.get("trail_length", 40))
        self.draw_trails = draw_trails
        self.draw_links = draw_links
        self.prefer_sam_masks = bool(prefer_sam_masks)

        # Visualization/UI tuning for live demos. These affect only drawing and
        # display readability; the saved tracking data and model outputs stay unchanged.
        self.label_scale = float(label_scale)
        self.label_thickness = max(1, int(label_thickness))
        self.event_overlay_enabled = bool(event_overlay_enabled)
        self.event_overlay_position = str(event_overlay_position)
        self.event_overlay_max_lines = max(1, int(event_overlay_max_lines))
        self.event_overlay_ttl_frames = max(1, int(event_overlay_ttl_frames))
        self.event_overlay_scale = float(event_overlay_scale)
        self.debug_overlay_enabled = bool(debug_overlay_enabled)
        self.debug_overlay_position = str(debug_overlay_position)
        self.debug_overlay_max_lines = max(1, int(debug_overlay_max_lines))
        self.debug_overlay_ttl_frames = max(1, int(debug_overlay_ttl_frames))
        self.debug_overlay_scale = float(debug_overlay_scale)

        # Important: SAM masks are useful for precise object/weapon/bag shape,
        # but they can make person tracking worse in CPU webcam demos because
        # masks are sparse, stale between SAM passes, and unstable during overlap.
        # By default, person identity memory stays YOLO/BoT-SORT-box based.
        default_sam_tracking = memory_cfg.get(
            "sam_tracking_classes",
            ["backpack", "handbag", "suitcase", "trolley_bag", "suspicious_object", "dangerous_object", "knife", "weapon"],
        )
        self.sam_tracking_classes = set(default_sam_tracking) if sam_tracking_classes is None else set(sam_tracking_classes)

        # Webcam/room demos often contain static background objects that COCO
        # mis-detects as suitcase/backpack. These filters remove obvious clutter
        # before it reaches the tracker memory. Configure in tracking_memory.yaml.
        self.ignored_zones_norm = memory_cfg.get("ignored_zones_norm", []) or []
        self.min_conf_by_class = memory_cfg.get("min_conf_by_class", {}) or {}
        self.min_area_ratio_by_class = memory_cfg.get("min_area_ratio_by_class", {}) or {}
        self.max_area_ratio_by_class = memory_cfg.get("max_area_ratio_by_class", {}) or {}
        self.min_height_width_ratio_by_class = memory_cfg.get("min_height_width_ratio_by_class", {}) or {}
        self.max_height_width_ratio_by_class = memory_cfg.get("max_height_width_ratio_by_class", {}) or {}
        self.person_low_conf_aspect_gate_below_conf = float(memory_cfg.get("person_low_conf_aspect_gate_below_conf", 0.65))
        self.person_low_conf_min_height_width_ratio = float(memory_cfg.get("person_low_conf_min_height_width_ratio", 1.15))
        self.same_class_nms_iou = float(memory_cfg.get("same_class_nms_iou", 0.82))
        self.same_class_nms_iou_by_class = memory_cfg.get("same_class_nms_iou_by_class", {}) or {}
        self.pause_recording_on_face = bool(pause_recording_on_face)

        roi_cfg_dict = dict(memory_cfg.get("roi_inner_search", {}) or {})
        if enable_roi_search is not None:
            roi_cfg_dict["enabled"] = bool(enable_roi_search)
        if roi_every_n_frames is not None:
            roi_cfg_dict["every_n_frames"] = int(roi_every_n_frames)
        if roi_confidence is not None:
            roi_cfg_dict["roi_confidence"] = float(roi_confidence)
        if roi_imgsz is not None:
            roi_cfg_dict["roi_imgsz"] = int(roi_imgsz)
        if roi_max_parent_rois is not None:
            roi_cfg_dict["max_parent_rois_per_frame"] = int(roi_max_parent_rois)
        self.roi_search_config = RoiSearchConfig.from_mapping(roi_cfg_dict)

        reid_cfg = memory_cfg.get("person_reid", {}) or {}
        self.person_reid = DeepPersonReID(
            enabled=bool(reid_cfg.get("enabled", True)),
            backend=str(reid_cfg.get("backend", "auto")),
            model_name=str(reid_cfg.get("model_name", "osnet_x0_25")),
            model_path=str(reid_cfg.get("model_path", "")),
            device=device or reid_cfg.get("device", None),
            image_height=int(reid_cfg.get("image_height", 256)),
            image_width=int(reid_cfg.get("image_width", 128)),
            allow_torchvision_fallback=bool(reid_cfg.get("allow_torchvision_fallback", False)),
            require_backend=bool(reid_cfg.get("require_backend", True)),
        )
        configure_deep_person_reid(self.person_reid)
        print(f"[ReID] {get_deep_person_reid_status()}")

        self.detector = YoloDetector(
            weights=yolo_weights,
            tracker_config=tracker_config,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )

        secondary_cfg = risk_cfg.get("secondary_detector", {}) or {}
        self.secondary_detector: Optional[YoloDetector] = None
        self.secondary_target_classes = self._normalize_class_set(
            secondary_target_classes
            if secondary_target_classes is not None
            else secondary_cfg.get("target_classes", list(self.risk_classes))
        )
        self.secondary_conf = float(
            secondary_conf
            if secondary_conf is not None
            else secondary_cfg.get("conf", 0.35)
        )
        self.secondary_imgsz = int(
            secondary_imgsz
            if secondary_imgsz is not None
            else secondary_cfg.get("imgsz", imgsz)
        )
        cfg_secondary_iou = secondary_cfg.get("iou", None)
        self.secondary_iou = (
            float(secondary_iou)
            if secondary_iou is not None
            else (float(cfg_secondary_iou) if cfg_secondary_iou is not None else None)
        )
        cfg_secondary_max_det = secondary_cfg.get("max_det", None)
        self.secondary_max_det = (
            int(secondary_max_det)
            if secondary_max_det is not None
            else (int(cfg_secondary_max_det) if cfg_secondary_max_det is not None else None)
        )
        if secondary_weights is not None and str(secondary_weights).strip():
            self.secondary_detector = YoloDetector(
                weights=str(secondary_weights),
                tracker_config=tracker_config,
                conf=self.secondary_conf,
                imgsz=self.secondary_imgsz,
                device=device,
            )
            target_text = "all classes" if self.secondary_target_classes is None else ",".join(sorted(self.secondary_target_classes))
            print(f"[Secondary YOLO] {secondary_weights} -> {target_text}")

        risk_confirmation_cfg = risk_cfg.get("risk_confirmation", {}) or {}
        smoothing_enabled = bool(risk_confirmation_cfg.get("enabled", True))
        if risk_smoothing_enabled is not None:
            smoothing_enabled = bool(risk_smoothing_enabled)
        confirmation_classes = self._normalize_class_set(
            risk_confirmation_cfg.get("classes", list(self.risk_classes))
        ) or set(self.risk_classes)
        self.risk_confirmation = RiskConfirmationFilter(
            classes=set(confirmation_classes),
            enabled=smoothing_enabled,
            window_frames=int(weapon_confirm_window if weapon_confirm_window is not None else risk_confirmation_cfg.get("window_frames", 5)),
            min_hits=int(weapon_confirm_min_hits if weapon_confirm_min_hits is not None else risk_confirmation_cfg.get("min_hits", 3)),
            match_iou=float(weapon_confirm_match_iou if weapon_confirm_match_iou is not None else risk_confirmation_cfg.get("match_iou", 0.10)),
            match_center_distance_px=float(weapon_confirm_match_center_px if weapon_confirm_match_center_px is not None else risk_confirmation_cfg.get("match_center_distance_px", 140.0)),
        )

        risk_cluster_cfg = risk_cfg.get("risk_cluster", {}) or {}
        cluster_classes = self._normalize_class_set(risk_cluster_cfg.get("classes", list(self.risk_classes))) or set(self.risk_classes)
        self.risk_cluster_manager = RiskClusterManager(
            classes=set(cluster_classes),
            enabled=bool(risk_cluster_enabled if risk_cluster_enabled is not None else risk_cluster_cfg.get("enabled", True)),
            match_iou=float(risk_cluster_match_iou if risk_cluster_match_iou is not None else risk_cluster_cfg.get("match_iou", 0.08)),
            match_center_distance_px=float(risk_cluster_match_center_px if risk_cluster_match_center_px is not None else risk_cluster_cfg.get("match_center_distance_px", 95.0)),
            ttl_frames=int(risk_cluster_ttl_frames if risk_cluster_ttl_frames is not None else risk_cluster_cfg.get("ttl_frames", 45)),
            owner_link_gap_frames=int(risk_cluster_cfg.get("owner_link_gap_frames", 3)),
            warning_cooldown_frames=int(risk_warning_cooldown_frames if risk_warning_cooldown_frames is not None else risk_cluster_cfg.get("warning_cooldown_frames", 30)),
            confirmed_linked_frames=int(risk_confirm_linked_frames if risk_confirm_linked_frames is not None else risk_cluster_cfg.get("confirmed_linked_frames", 30)),
            confirmed_cooldown_frames=int(risk_cluster_cfg.get("confirmed_cooldown_frames", 120)),
            repeated_warning_window_frames=int(risk_repeat_warning_window_frames if risk_repeat_warning_window_frames is not None else risk_cluster_cfg.get("repeated_warning_window_frames", 300)),
            repeated_warning_count=int(risk_repeat_warning_count if risk_repeat_warning_count is not None else risk_cluster_cfg.get("repeated_warning_count", 10)),
            repeated_warning_cooldown_frames=int(risk_cluster_cfg.get("repeated_warning_cooldown_frames", 300)),
            high_confidence_warning=float(risk_cluster_cfg.get("high_confidence_warning", 0.85)),
        )

        self.segmenter = SamBoxSegmenter(weights=sam_weights, enabled=enable_sam, device=device)
        self.roi_search = NestedRoiObjectSearch(self.detector, self.roi_search_config)
        self.sam_every_n_frames = max(1, int(sam_every_n_frames))
        self.sam_max_objects = max(0, int(sam_max_objects))
        self.sam_classes = sam_classes
        self.reuse_last_masks = reuse_last_masks
        self.last_masks_by_track_id: dict[int, object] = {}

        self.privacy_filter = FacePrivacyFilter(enabled=blur_faces)

        self.memory_bank = MemoryBank(
            max_missing_frames=int(memory_cfg.get("max_missing_frames", 180)),
            max_reid_gap_frames=int(memory_cfg.get("max_reid_gap_frames", 180)),
            ema_alpha=float(memory_cfg.get("ema_alpha", 0.65)),
            stationary_movement_px=float(memory_cfg.get("stationary_movement_px", 3.0)),
            min_stitch_score=float(memory_cfg.get("min_stitch_score", 0.58)),
            max_center_distance_px=float(memory_cfg.get("max_center_distance_px", 260.0)),
            min_appearance_similarity=float(memory_cfg.get("min_appearance_similarity", 0.18)),
            appearance_weight=float(memory_cfg.get("appearance_weight", 0.55)),
            iou_weight=float(memory_cfg.get("iou_weight", 0.20)),
            proximity_weight=float(memory_cfg.get("proximity_weight", 0.25)),
            person_min_stitch_score=float(memory_cfg.get("person_min_stitch_score", 0.72)),
            person_min_appearance_similarity=float(memory_cfg.get("person_min_appearance_similarity", 0.52)),
            person_min_secondary_similarity=float(memory_cfg.get("person_min_secondary_similarity", 0.18)),
            person_deep_appearance_weight=float(memory_cfg.get("person_deep_appearance_weight", 0.74)),
            person_secondary_appearance_weight=float(memory_cfg.get("person_secondary_appearance_weight", 0.08)),
            person_iou_weight=float(memory_cfg.get("person_iou_weight", 0.06)),
            person_proximity_weight=float(memory_cfg.get("person_proximity_weight", 0.12)),
            person_gallery_size=int(memory_cfg.get("person_gallery_size", 8)),
            person_gallery_duplicate_similarity=float(memory_cfg.get("person_gallery_duplicate_similarity", 0.985)),
            person_crop_edge_margin_ratio=float(memory_cfg.get("person_crop_edge_margin_ratio", 0.025)),
            person_crop_min_height_ratio=float(memory_cfg.get("person_crop_min_height_ratio", 0.28)),
            person_crop_min_area_ratio=float(memory_cfg.get("person_crop_min_area_ratio", 0.018)),
            person_crop_min_aspect_ratio=float(memory_cfg.get("person_crop_min_aspect_ratio", 0.16)),
            person_crop_max_aspect_ratio=float(memory_cfg.get("person_crop_max_aspect_ratio", 0.85)),
            person_crop_allow_bottom_edge=bool(memory_cfg.get("person_crop_allow_bottom_edge", True)),
            entry_exit_enabled=bool(memory_cfg.get("entry_exit_enabled", True)),
            entry_exit_edge_ratio=float(memory_cfg.get("entry_exit_edge_ratio", 0.055)),
            entry_exit_window_frames=int(memory_cfg.get("entry_exit_window_frames", 240)),
            entry_exit_same_side_bonus=float(memory_cfg.get("entry_exit_same_side_bonus", 0.08)),
        )
        self.risk_engine = RiskEngine(
            risk_classes=self.risk_classes,
            bag_classes=self.bag_classes,
            stationary_threshold_frames=int(unattended_cfg.get("stationary_threshold_frames", 90)),
            owner_distance_threshold_px=float(unattended_cfg.get("owner_distance_threshold_px", 250.0)),
            unattended_cooldown_frames=int(unattended_cfg.get("cooldown_frames", 60)),
            separation_threshold_frames=int(unattended_cfg.get("separation_threshold_frames", 90)),
            min_owner_contact_frames=int(unattended_cfg.get("min_owner_contact_frames", 45)),
            risk_detection_cooldown_frames=int(risk_cfg.get("risk_detection_cooldown_frames", 30)),
        )
        self.events: list[Event] = []
        self.recent_alert_messages: list[tuple[int, str, str]] = []
        self.recent_debug_messages: list[tuple[int, str, str]] = []
        self.track_rows: list[dict[str, object]] = []
        self.last_reid_message_frame: dict[int, int] = {}
        self.reid_message_cooldown_frames = int(memory_cfg.get("reid_message_cooldown_frames", 90))

        offline_cfg = memory_cfg.get("offline_merge", {}) or {}
        self.offline_merge_enabled = bool(offline_cfg.get("enabled", True))
        self.offline_merge_max_gap_frames = int(offline_cfg.get("max_gap_frames", 900))
        self.offline_merge_max_overlap_frames = int(offline_cfg.get("max_overlap_frames", 3))
        self.offline_merge_min_score = float(offline_cfg.get("min_score", 0.66))
        self.offline_merge_min_primary_similarity = float(offline_cfg.get("min_primary_similarity", 0.58))
        self.offline_merge_secondary_weight = float(offline_cfg.get("secondary_weight", 0.12))
        self.offline_merge_gap_weight = float(offline_cfg.get("gap_weight", 0.08))
        self.offline_merge_side_bonus = float(offline_cfg.get("same_side_bonus", 0.08))
        self.offline_merge_map: dict[int, int] = {}

    def run(
        self,
        source: str,
        output_video: Optional[str] = "outputs/annotated_video.mp4",
        output_json: str = "outputs/events.json",
        output_tracks_csv: Optional[str] = "outputs/tracks.csv",
        display: bool = False,
        max_frames: Optional[int] = None,
        display_scale: float = 1.0,
        display_width: Optional[int] = None,
        display_height: Optional[int] = None,
    ) -> None:
        parsed_source = parse_source(source)
        cap = cv2.VideoCapture(parsed_source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open source: {source}\n"
                "For a video file, either place it in the input/ folder as input/test_video.mp4 "
                "or pass the full path, for example: "
                "python scripts/run_video.py --source C:\\path\\to\\video.mp4 --disable-sam"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video:
            ensure_parent(output_video)
            writer = cv2.VideoWriter(
                output_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video writer: {output_video}")

        frame_idx = 0
        last_time = time.perf_counter()
        smoothed_fps = 0.0

        window_name = "Screening AI Prototype"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            target_w = int(display_width) if display_width is not None else int(width * max(float(display_scale), 0.1))
            target_h = int(display_height) if display_height is not None else int(height * max(float(display_scale), 0.1))
            if target_w > 0 and target_h > 0:
                cv2.resizeWindow(window_name, target_w, target_h)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            last_time = now
            instant_fps = 1.0 / dt
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps

            annotated = self.process_frame(frame, frame_idx)
            draw_fps(annotated, smoothed_fps, scale=self.label_scale, thickness=self.label_thickness)

            face_visible = False
            if self.pause_recording_on_face:
                face_visible = self.privacy_filter.has_face_like_region(annotated)
                if face_visible:
                    cv2.putText(
                        annotated,
                        "RECORDING PAUSED: FACE-LIKE REGION VISIBLE",
                        (20, max(35, height - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

            self.privacy_filter.apply(annotated)

            if writer is not None and not face_visible:
                writer.write(annotated)

            if display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        self._apply_offline_person_merges()

        ensure_parent(output_json)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([event.to_json() for event in self.events], f, indent=2, ensure_ascii=False)

        if output_tracks_csv:
            self._write_track_csv(output_tracks_csv)

    def _write_track_csv(self, output_tracks_csv: str) -> None:
        ensure_parent(output_tracks_csv)
        fieldnames = [
            "frame",
            "global_id",
            "raw_global_id",
            "offline_merged_into",
            "local_tracker_id",
            "class_name",
            "detection_source",
            "risk_cluster_id",
            "risk_cluster_state",
            "parent_class_name",
            "roi_level",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
            "owner_id",
            "owner_link_strength",
            "owner_contact_frames",
            "owner_separation_frames",
            "owner_last_distance_px",
            "stationary_frames",
            "missing_frames",
            "mask_area",
            "used_mask_geometry",
            "reidentified_count",
            "good_snapshot_count",
            "crop_quality",
            "crop_quality_reason",
            "last_observed_side",
            "exit_side",
            "entry_side",
        ]
        with open(output_tracks_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.track_rows)



    @staticmethod
    def _normalize_class_set(value) -> Optional[set[str]]:
        """Return a normalized class whitelist, or None when all classes are allowed."""
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip().lower() in {"", "all", "none"}:
                return None
            items = [x.strip() for x in value.split(",")]
        else:
            items = [str(x).strip() for x in value]
        normalized = {x for x in items if x}
        if not normalized or "all" in {x.lower() for x in normalized}:
            return None
        return normalized

    def _filter_detections(self, frame, detections: list[Detection]) -> list[Detection]:
        """Remove obvious webcam clutter before memory/reID.

        This does not replace training. It is a practical demo stabilizer:
        - removes detections whose center is inside configured ignore zones
        - applies class-specific confidence/area thresholds
        - suppresses duplicate same-class boxes that heavily overlap
        """
        if not detections:
            return []

        height, width = frame.shape[:2]
        frame_area = float(max(1, width * height))
        filtered: list[Detection] = []

        for det in detections:
            if self.target_classes is not None and det.class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = det.bbox_xyxy.astype(float)
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            height_width_ratio = box_h / max(box_w, 1.0)
            box_area_ratio = box_w * box_h / frame_area
            cx = (x1 + x2) / 2.0 / max(1, width)
            cy = (y1 + y2) / 2.0 / max(1, height)

            min_conf = float(self.min_conf_by_class.get(det.class_name, 0.0))
            if det.confidence < min_conf:
                continue

            min_area = float(self.min_area_ratio_by_class.get(det.class_name, 0.0))
            max_area = float(self.max_area_ratio_by_class.get(det.class_name, 1.0))
            if box_area_ratio < min_area or box_area_ratio > max_area:
                continue

            min_hw_ratio = float(self.min_height_width_ratio_by_class.get(det.class_name, 0.0))
            max_hw_ratio = float(self.max_height_width_ratio_by_class.get(det.class_name, 999.0))
            if height_width_ratio < min_hw_ratio or height_width_ratio > max_hw_ratio:
                continue

            # A common webcam false-positive is clothes/chair/backpack detected as
            # a low-confidence person with a wide, non-human-shaped bbox. Keep
            # high-confidence/partial real people, but reject low-confidence
            # person boxes that are too squat.
            if det.class_name in self.person_classes:
                if det.confidence < self.person_low_conf_aspect_gate_below_conf and height_width_ratio < self.person_low_conf_min_height_width_ratio:
                    continue

            ignored = False
            for zone in self.ignored_zones_norm:
                zone_classes = set(zone.get("classes", []))
                if zone_classes and det.class_name not in zone_classes:
                    continue
                zx1, zy1, zx2, zy2 = zone.get("xyxy", [0, 0, 0, 0])
                if float(zx1) <= cx <= float(zx2) and float(zy1) <= cy <= float(zy2):
                    ignored = True
                    break
            if ignored:
                continue

            filtered.append(det)

        # Same-class duplicate cleanup. We do not suppress person-vs-backpack
        # overlaps because that overlap is meaningful in your project.
        filtered.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for det in filtered:
            duplicate = False
            for prev in kept:
                if prev.class_name != det.class_name:
                    continue
                duplicate_iou = float(self.same_class_nms_iou_by_class.get(det.class_name, self.same_class_nms_iou))
                if bbox_iou_xyxy(prev.bbox_xyxy, det.bbox_xyxy) >= duplicate_iou:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept

    def _segment_detections_before_memory(
        self,
        frame,
        frame_idx: int,
        detections: list[Detection],
    ) -> list[Optional[SegmentationResult]]:
        segmentations: list[Optional[SegmentationResult]] = [None for _ in detections]
        should_segment = (
            self.segmenter.enabled
            and self.sam_max_objects > 0
            and frame_idx % self.sam_every_n_frames == 0
        )

        if not should_segment:
            return segmentations

        candidate_indices = []
        for i, det in enumerate(detections):
            # Always allow risk classes through SAM when SAM is enabled, even if
            # the user only passed --sam-classes backpack,handbag,suitcase.
            # This keeps the normal baseline command useful after training a
            # knife/gun/custom-dangerous-object detector.
            if self.sam_classes is not None and det.class_name not in self.sam_classes and det.class_name not in self.risk_classes:
                continue
            candidate_indices.append(i)

        # Risk objects get first priority for SAM, then higher-confidence boxes.
        # Limit count to keep webcam responsive.
        candidate_indices.sort(key=lambda i: (detections[i].class_name in self.risk_classes, detections[i].confidence), reverse=True)
        candidate_indices = candidate_indices[: self.sam_max_objects]

        boxes_for_sam = [detections[i].bbox_xyxy for i in candidate_indices]
        sam_results = self.segmenter.segment_boxes(frame, boxes_for_sam)

        for local_i, det_i in enumerate(candidate_indices):
            segmentations[det_i] = sam_results[local_i] if local_i < len(sam_results) else None

        return segmentations


    def _prefer_mask_for_class(self, class_name: str) -> bool:
        return self.prefer_sam_masks and class_name in self.sam_tracking_classes

    def _secondary_detect_frame(self, frame) -> list[Detection]:
        if self.secondary_detector is None:
            return []

        detections = self.secondary_detector.detect_frame(
            frame,
            conf=self.secondary_conf,
            imgsz=self.secondary_imgsz,
            iou=self.secondary_iou,
            max_det=self.secondary_max_det,
            source="secondary",
        )
        if self.secondary_target_classes is None:
            return detections
        return [det for det in detections if det.class_name in self.secondary_target_classes]

    def _apply_risk_confirmation(self, frame_idx: int, detections: list[Detection]) -> list[Detection]:
        if not detections or not self.risk_classes:
            return detections

        risk_detections = [det for det in detections if det.class_name in self.risk_classes]
        self.risk_confirmation.update(frame_idx, risk_detections)

        confirmed: list[Detection] = []
        for det in detections:
            if det.class_name not in self.risk_classes:
                confirmed.append(det)
                continue
            if self.risk_confirmation.is_confirmed(det, frame_idx):
                confirmed.append(det)
        return confirmed

    def process_frame(self, frame, frame_idx: int):
        detections = self.detector.track_frame(frame)

        secondary_detections = self._secondary_detect_frame(frame)
        if secondary_detections:
            detections.extend(secondary_detections)

        detections = self._filter_detections(frame, detections)

        # Second-pass ROI search: run plain YOLO detection inside selected
        # parent boxes (person/bag/suitcase) to recover small visible objects
        # that the full-frame detector missed because boxes overlap. These
        # supplemental detections are untracked by BoT-SORT; the MemoryBank can
        # still assign/reconnect project-level global IDs to them.
        roi_detections = self.roi_search.find_inner_detections(frame, frame_idx, detections)
        if roi_detections:
            detections = self._filter_detections(frame, detections + roi_detections)

        detections = self._apply_risk_confirmation(frame_idx, detections)

        seen_global_ids: set[int] = set()
        frame_events: list[Event] = []

        # SAM-first when enabled: YOLO still proposes boxes, but memory/association can
        # use SAM masks for cleaner object shape, center, and appearance.
        segmentations = self._segment_detections_before_memory(frame, frame_idx, detections)

        active_local_ids_by_class: dict[str, set[int]] = {}
        for det in detections:
            if det.track_id is not None:
                active_local_ids_by_class.setdefault(det.class_name, set()).add(int(det.track_id))

        resolutions = []
        for det, seg in zip(detections, segmentations):
            mask = seg.mask if seg is not None else None
            resolution = self.memory_bank.resolve_global_id(
                local_tracker_id=det.track_id,
                class_name=det.class_name,
                frame_idx=frame_idx,
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                frame=frame,
                excluded_global_ids=seen_global_ids,
                active_local_ids_by_class=active_local_ids_by_class,
                mask=mask,
                prefer_mask_geometry=self._prefer_mask_for_class(det.class_name),
            )
            resolutions.append(resolution)
            seen_global_ids.add(resolution.global_id)

            if resolution.was_reidentified and det.class_name not in self.risk_classes:
                if resolution.merged_from_global_id is not None:
                    self._rewrite_previous_track_rows(
                        old_global_id=resolution.merged_from_global_id,
                        new_global_id=resolution.global_id,
                    )
                last_msg_frame = self.last_reid_message_frame.get(resolution.global_id, -10**9)
                if frame_idx - last_msg_frame >= self.reid_message_cooldown_frames:
                    self.last_reid_message_frame[resolution.global_id] = frame_idx
                    if resolution.merged_from_global_id is not None:
                        message = (
                            f"Rolled {det.class_name} G{resolution.merged_from_global_id} "
                            f"back into canonical G{resolution.global_id}"
                        )
                        event_type = "track_merged"
                    else:
                        message = (
                            f"Reconnected {det.class_name} as global track G{resolution.global_id} "
                            f"after local ID change"
                        )
                        event_type = "track_reidentified"
                    frame_events.append(
                        Event(
                            frame=frame_idx,
                            type=event_type,
                            message=message,
                            track_id=resolution.global_id,
                            class_name=det.class_name,
                            confidence=resolution.match_score,
                        )
                    )

        # Reuse old masks after global IDs are known.
        if self.reuse_last_masks:
            for i, resolution in enumerate(resolutions):
                if segmentations[i] is not None and segmentations[i].mask is not None:
                    self.last_masks_by_track_id[resolution.global_id] = segmentations[i].mask
                    continue
                old_mask = self.last_masks_by_track_id.get(resolution.global_id)
                if old_mask is not None:
                    segmentations[i] = SegmentationResult(mask=old_mask)

        frame_items = []
        for det, resolution, seg in zip(detections, resolutions, segmentations):
            mask = seg.mask if seg is not None else None
            memory = self.memory_bank.update(
                track_id=resolution.global_id,
                class_name=det.class_name,
                frame_idx=frame_idx,
                bbox_xyxy=det.bbox_xyxy,
                confidence=det.confidence,
                frame=frame,
                local_tracker_id=resolution.local_tracker_id,
                mask=mask,
                prefer_mask_geometry=self._prefer_mask_for_class(det.class_name),
            )
            frame_items.append((det, resolution, seg, memory))

        self.memory_bank.mark_missing(seen_global_ids)

        # Bags need long relationship evidence before ownership is confirmed.
        assign_bag_owners(
            memory_bank=self.memory_bank,
            person_classes=self.person_classes,
            bag_classes=self.bag_classes,
            max_distance_px=self.max_owner_distance_px,
            min_owner_score=self.min_owner_score,
            motion_window_frames=self.motion_window_frames,
            frame_idx=frame_idx,
            score_decay=self.owner_score_decay,
            score_gain=self.owner_score_gain,
            min_contact_frames=int(self.owner_min_contact_frames),
            contact_distance_px=float(self.owner_contact_distance_px),
            separation_distance_px=float(self.owner_separation_distance_px),
        )

        # Risk objects such as knife/gun/custom dangerous classes should link
        # quickly to the nearest plausible person, because they may be visible
        # only briefly. This is intentionally separate from abandoned-bag logic.
        if self.risk_classes:
            assign_object_owners(
                memory_bank=self.memory_bank,
                person_classes=self.person_classes,
                object_classes=self.risk_classes,
                max_distance_px=self.risk_owner_max_distance_px,
                min_owner_score=self.risk_owner_min_score,
                motion_window_frames=self.motion_window_frames,
                frame_idx=frame_idx,
                score_decay=self.risk_owner_score_decay,
                score_gain=self.risk_owner_score_gain,
                min_contact_frames=self.risk_owner_min_contact_frames,
                contact_distance_px=self.risk_owner_contact_distance_px,
                separation_distance_px=self.risk_owner_separation_distance_px,
            )

        risk_clusters_by_global_id: dict[int, RiskCluster] = {}

        for det, resolution, seg, memory in frame_items:
            if det.class_name not in self.risk_classes:
                continue
            cluster, cluster_events = self.risk_cluster_manager.update_detection(
                frame_idx=frame_idx,
                class_name=det.class_name,
                bbox_xyxy=memory.smoothed_bbox_xyxy,
                confidence=det.confidence,
                owner_id=memory.owner_id,
                source_track_id=resolution.global_id,
            )
            if cluster is not None:
                risk_clusters_by_global_id[int(resolution.global_id)] = cluster
            frame_events.extend(cluster_events)

        for det, resolution, seg, memory in frame_items:
            mask = seg.mask if seg is not None else None
            is_risk = det.class_name in self.risk_classes
            risk_cluster = risk_clusters_by_global_id.get(int(resolution.global_id)) if is_risk else None
            display_label = None
            if is_risk:
                cluster_id = risk_cluster.cluster_id if risk_cluster is not None else "R?"
                state = (risk_cluster.display_state if risk_cluster is not None else "possible").upper()
                owner_text = f" ->G{memory.owner_id}" if memory.owner_id is not None else ""
                display_label = f"{state} {det.class_name} {cluster_id} {det.confidence:.2f}{owner_text}"

            draw_detection(
                frame=frame,
                bbox_xyxy=memory.smoothed_bbox_xyxy,
                class_name=det.class_name,
                track_id=resolution.global_id,
                confidence=det.confidence,
                mask=mask,
                owner_id=memory.owner_id,
                local_tracker_id=resolution.local_tracker_id,
                reidentified_count=memory.reidentified_count,
                is_risk=is_risk,
                label_scale=self.label_scale,
                label_thickness=self.label_thickness,
                display_label=display_label,
            )
            self._record_track_row(frame_idx, memory, det, risk_cluster=risk_cluster)

        frame_events.extend(self.risk_engine.unattended_bag_events(frame_idx, self.memory_bank))

        if self.draw_trails:
            draw_track_trails(frame, self.memory_bank, trail_length=self.trail_length)
        if self.draw_links:
            draw_owner_links(
                frame,
                owner_link_lines(
                    self.memory_bank,
                    self.bag_classes,
                    max_missing_frames=self.owner_link_display_max_missing_frames,
                    min_link_strength=self.owner_link_display_min_strength,
                    risk_classes=self.risk_classes,
                ),
                label_scale=max(0.35, self.label_scale * 0.88),
                label_thickness=self.label_thickness,
            )

        if frame_events:
            self.events.extend(frame_events)
            for event in frame_events:
                self._remember_event_message(frame_idx, event)

        if self.event_overlay_enabled:
            active_alerts = self._active_messages(
                self.recent_alert_messages,
                frame_idx=frame_idx,
                ttl_frames=self.event_overlay_ttl_frames,
            )
            self.recent_alert_messages = active_alerts[-30:]
            draw_event_banner(
                frame,
                active_alerts,
                position=self.event_overlay_position,
                max_lines=self.event_overlay_max_lines,
                scale=self.event_overlay_scale,
                thickness=self.label_thickness,
                title="SECURITY ALERTS",
            )

        if self.debug_overlay_enabled:
            active_debug = self._active_messages(
                self.recent_debug_messages,
                frame_idx=frame_idx,
                ttl_frames=self.debug_overlay_ttl_frames,
            )
            self.recent_debug_messages = active_debug[-30:]
            draw_event_banner(
                frame,
                active_debug,
                position=self.debug_overlay_position,
                max_lines=self.debug_overlay_max_lines,
                scale=self.debug_overlay_scale,
                thickness=self.label_thickness,
                title="TRACKING DEBUG",
            )
        return frame

    def _remember_event_message(self, frame_idx: int, event: Event) -> None:
        item = (int(frame_idx), str(event.type), str(event.message))
        if self._is_debug_event(event):
            self.recent_debug_messages.append(item)
            self.recent_debug_messages = self.recent_debug_messages[-30:]
        else:
            self.recent_alert_messages.append(item)
            self.recent_alert_messages = self.recent_alert_messages[-30:]

    @staticmethod
    def _is_debug_event(event: Event) -> bool:
        event_type = str(event.type).lower()
        return "reidentified" in event_type or "merge" in event_type or event_type.startswith("track_")

    @staticmethod
    def _active_messages(
        messages: list[tuple[int, str, str]],
        frame_idx: int,
        ttl_frames: int,
    ) -> list[tuple[int, str, str]]:
        return [
            item
            for item in messages
            if int(frame_idx) - int(item[0]) <= int(ttl_frames)
        ]

    def _rewrite_previous_track_rows(self, old_global_id: int, new_global_id: int) -> None:
        """Rewrite already-collected CSV rows after an ID rollback/merge."""
        for row in self.track_rows:
            if int(row.get("global_id", -1)) == int(old_global_id):
                row["global_id"] = int(new_global_id)
            owner_id = row.get("owner_id", "")
            if owner_id != "" and int(owner_id) == int(old_global_id):
                row["owner_id"] = int(new_global_id)

    def _apply_offline_person_merges(self) -> None:
        """After the video ends, merge highly likely split person IDs in CSV/events.

        Online tracking stays conservative to avoid collapsing two visible people.
        This pass is allowed to be more global: it compares complete track segments,
        OSNet gallery snapshots, time gaps, overlap, and entry/exit side continuity.
        """
        if not self.offline_merge_enabled:
            return

        person_tracks = [
            t for t in self.memory_bank.tracks.values()
            if t.class_name in self.person_classes and t.good_snapshot_count > 0
        ]
        if len(person_tracks) < 2:
            return

        parent: dict[int, int] = {int(t.global_id): int(t.global_id) for t in person_tracks}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return parent.get(x, x)

        candidates: list[tuple[float, int, int, float, float, bool]] = []
        for i, left in enumerate(person_tracks):
            for right in person_tracks[i + 1:]:
                score, primary, secondary, same_side = self._offline_person_merge_score(left, right)
                if score >= self.offline_merge_min_score and primary >= self.offline_merge_min_primary_similarity:
                    candidates.append((score, int(left.global_id), int(right.global_id), primary, secondary, same_side))

        candidates.sort(reverse=True, key=lambda item: item[0])
        merge_events: list[Event] = []

        for score, left_id, right_id, primary, secondary, same_side in candidates:
            left_root = find(left_id)
            right_root = find(right_id)
            if left_root == right_root:
                continue

            left_track = self.memory_bank.tracks.get(left_root) or self.memory_bank.tracks.get(left_id)
            right_track = self.memory_bank.tracks.get(right_root) or self.memory_bank.tracks.get(right_id)
            if left_track is None or right_track is None:
                continue

            if (left_track.first_frame, left_track.global_id) <= (right_track.first_frame, right_track.global_id):
                target_id, source_id = int(left_root), int(right_root)
            else:
                target_id, source_id = int(right_root), int(left_root)

            parent[source_id] = target_id
            message = (
                f"Offline merge: person G{source_id} -> G{target_id} "
                f"score={score:.3f}, osnet={primary:.3f}, secondary={secondary:.3f}, same_side={same_side}"
            )
            merge_events.append(
                Event(
                    frame=max(left_track.last_frame, right_track.last_frame),
                    type="offline_track_merge",
                    message=message,
                    track_id=target_id,
                    class_name="person",
                    confidence=score,
                )
            )

        final_map: dict[int, int] = {}
        for track_id in list(parent):
            root = find(track_id)
            if root != track_id:
                final_map[track_id] = root

        if not final_map:
            return

        self.offline_merge_map = final_map
        self._rewrite_track_rows_after_offline_merge(final_map)
        self._rewrite_events_after_offline_merge(final_map)
        self.events.extend(merge_events)

    def _offline_person_merge_score(self, left, right) -> tuple[float, float, float, bool]:
        overlap = min(left.last_frame, right.last_frame) - max(left.first_frame, right.first_frame) + 1
        if overlap > self.offline_merge_max_overlap_frames:
            return 0.0, 0.0, 0.0, False

        if left.last_frame < right.first_frame:
            gap = right.first_frame - left.last_frame
            earlier, later = left, right
        elif right.last_frame < left.first_frame:
            gap = left.first_frame - right.last_frame
            earlier, later = right, left
        else:
            gap = 0
            earlier, later = left, right

        if gap > self.offline_merge_max_gap_frames:
            return 0.0, 0.0, 0.0, False

        primary, secondary = self.memory_bank.track_gallery_similarity(left, right)
        same_side = (
            earlier.exit_side is not None
            and later.entry_side is not None
            and earlier.exit_side == later.entry_side
        )
        gap_score = 1.0 - min(float(gap) / max(1.0, float(self.offline_merge_max_gap_frames)), 1.0)
        score = 0.80 * primary + self.offline_merge_secondary_weight * secondary + self.offline_merge_gap_weight * gap_score
        if same_side:
            score += self.offline_merge_side_bonus
        return score, primary, secondary, same_side

    def _rewrite_track_rows_after_offline_merge(self, final_map: dict[int, int]) -> None:
        def canonical(value: int) -> int:
            seen = set()
            while value in final_map and value not in seen:
                seen.add(value)
                value = int(final_map[value])
            return int(value)

        for row in self.track_rows:
            raw_global_id = int(row.get("raw_global_id", row.get("global_id", -1)))
            new_global_id = canonical(raw_global_id)
            row["raw_global_id"] = raw_global_id
            row["global_id"] = new_global_id
            row["offline_merged_into"] = "" if new_global_id == raw_global_id else new_global_id
            owner_id = row.get("owner_id", "")
            if owner_id != "":
                row["owner_id"] = canonical(int(owner_id))

    def _rewrite_events_after_offline_merge(self, final_map: dict[int, int]) -> None:
        def canonical(value: int | None) -> int | None:
            if value is None:
                return None
            current = int(value)
            seen = set()
            while current in final_map and current not in seen:
                seen.add(current)
                current = int(final_map[current])
            return current

        replacements: list[tuple[int, int]] = []
        for old_id, new_id in final_map.items():
            replacements.append((int(old_id), int(canonical(new_id))))

        for event in self.events:
            event.track_id = canonical(event.track_id)
            event.owner_id = canonical(event.owner_id)
            for old_id, new_id in replacements:
                event.message = event.message.replace(f"G{old_id}", f"G{new_id}")
                event.message = event.message.replace(f"track {old_id}", f"track {new_id}")


    def _record_track_row(self, frame_idx: int, memory, detection: Optional[Detection] = None, risk_cluster: RiskCluster | None = None) -> None:
        x1, y1, x2, y2 = memory.smoothed_bbox_xyxy.astype(float)
        cx, cy = memory.last_center
        self.track_rows.append(
            {
                "frame": int(frame_idx),
                "global_id": int(memory.global_id),
                "raw_global_id": int(memory.global_id),
                "offline_merged_into": "",
                "local_tracker_id": "" if memory.local_tracker_id is None else int(memory.local_tracker_id),
                "class_name": memory.class_name,
                "detection_source": "" if detection is None else detection.source,
                "risk_cluster_id": "" if risk_cluster is None else risk_cluster.cluster_id,
                "risk_cluster_state": "" if risk_cluster is None else risk_cluster.display_state,
                "parent_class_name": "" if detection is None or detection.parent_class_name is None else detection.parent_class_name,
                "roi_level": "" if detection is None else int(detection.roi_level),
                "confidence": float(memory.last_confidence),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "center_x": round(float(cx), 2),
                "center_y": round(float(cy), 2),
                "owner_id": "" if memory.owner_id is None else int(memory.owner_id),
                "owner_link_strength": round(float(memory.owner_link_strength), 3),
                "owner_contact_frames": "" if memory.owner_id is None else int(memory.owner_contact_frames.get(memory.owner_id, 0)),
                "owner_separation_frames": "" if memory.owner_id is None else int(memory.owner_separation_frames.get(memory.owner_id, 0)),
                "owner_last_distance_px": round(float(memory.owner_last_distance_px), 2),
                "stationary_frames": int(memory.stationary_frames),
                "missing_frames": int(memory.missing_frames),
                "mask_area": int(memory.last_mask_area),
                "used_mask_geometry": bool(memory.used_mask_geometry),
                "reidentified_count": int(memory.reidentified_count),
                "good_snapshot_count": int(memory.good_snapshot_count),
                "crop_quality": round(float(memory.last_crop_quality), 3),
                "crop_quality_reason": memory.last_crop_quality_reason,
                "last_observed_side": "" if memory.last_observed_side is None else memory.last_observed_side,
                "exit_side": "" if memory.exit_side is None else memory.exit_side,
                "entry_side": "" if memory.entry_side is None else memory.entry_side,
            }
        )
