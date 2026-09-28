from __future__ import annotations

import io
import json
import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = MagicMock()
    sys.modules["psycopg2"] = psycopg2_stub

import app  # noqa: E402


def test_market_label_for_path_parses_identity_and_camera():
    label = app._market_label_for_path("/tmp/query/0007_c3s1_000551_00.jpg")

    assert label == {"identity": 7, "camera": 3}


def test_partition_metrics_reports_perfect_clustering():
    metrics = app._partition_metrics([1, 1, 2, 2], [0, 0, 1, 1])

    assert metrics["available"] is True
    assert metrics["pairwise_f1"] == 1.0
    assert metrics["cluster_purity"] == 1.0


def test_ground_truth_coverage_counts_manual_string_labels():
    testset = {
        "images": [
            {"label": {"identity": "simba"}},
            {"label": {"identity": "simba"}},
            {"label": {"identity": "mufasa"}},
            {"label": {"identity": None}},
        ]
    }

    coverage = app._ground_truth_coverage(testset)

    assert coverage["labeled_count"] == 3
    assert coverage["unlabeled_count"] == 1
    assert coverage["identity_count"] == 2


def test_normalize_ground_truth_identity_accepts_ints_and_names():
    assert app._normalize_ground_truth_identity("42") == 42
    assert app._normalize_ground_truth_identity(" simba ") == "simba"
    assert app._normalize_ground_truth_identity(" ") is None


def test_cluster_features_splits_different_vectors():
    labels = app._cluster_features(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        threshold=0.95,
    )

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_image_feature_vector_is_normalized(tmp_path):
    image_path = tmp_path / "0001_c1_test.jpg"
    image = Image.new("RGB", (32, 64), (200, 20, 20))
    image.save(image_path)

    vector = app._feature_for_image(str(image_path), "whole_embedding")

    norm = sum(value * value for value in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_sample_records_prefers_detections_when_available():
    testset = {
        "images": [
            {
                "path": "/tmp/a.jpg",
                "filename": "a.jpg",
                "label": {"identity": "image_label"},
                "detections": [
                    {
                        "id": "det-1",
                        "bbox": [1, 2, 10, 20],
                        "class_id": 0,
                        "class_name": "person",
                        "label": {"identity": "person_a"},
                    }
                ],
            }
        ]
    }

    records = app._sample_records_for_testset(testset)

    assert len(records) == 1
    assert records[0]["kind"] == "detection"
    assert records[0]["detection_id"] == "det-1"
    assert records[0]["label"]["identity"] == "person_a"


def test_sample_records_labeled_only_filters_unlabeled_detections():
    testset = {
        "images": [
            {
                "path": "/tmp/a.jpg",
                "filename": "a.jpg",
                "detections": [
                    {"id": "det-1", "bbox": [1, 2, 10, 20], "label": {"identity": "person_a"}},
                    {"id": "det-2", "bbox": [1, 2, 10, 20], "label": {"identity": None}},
                ],
            }
        ]
    }

    records = app._sample_records_for_testset(testset, labeled_only=True)

    assert len(records) == 1
    assert records[0]["detection_id"] == "det-1"


def test_nms_suppresses_overlapping_same_class_boxes():
    detections = [
        {"bbox": [0, 0, 100, 100], "score": 0.9, "class_id": 0},
        {"bbox": [5, 5, 95, 95], "score": 0.8, "class_id": 0},
        {"bbox": [5, 5, 95, 95], "score": 0.7, "class_id": 56},
    ]

    kept = app._nms_detections(detections, iou_threshold=0.5, limit=10)

    assert len(kept) == 2
    assert kept[0]["score"] == 0.9
    assert kept[1]["class_id"] == 56


def test_yolo_object_class_preset_excludes_person_class():
    classes = app._resolve_yolo_detection_classes("objects", None)

    assert 0 not in classes
    assert classes[0] == 1
    assert classes[-1] == len(app.COCO_CLASSES) - 1


def test_whole_plus_face_variants_exist_for_model_families():
    expected = {
        "whole_plus_face": "dinov3",
        "vit_whole_plus_face": "vit",
        "osnet_whole_plus_face": "osnet",
        "osnet_finetuned_whole_plus_face": "osnet_finetuned",
        "osnet_finetuned_improved_whole_plus_face": "osnet_finetuned_improved",
        "convnext_whole_plus_face": "convnext",
        "efficientnetv2_whole_plus_face": "efficientnetv2",
        "deit_whole_plus_face": "deit",
        "fastreid_whole_plus_face": "fastreid",
    }

    for variant, family in expected.items():
        assert variant in app.CLUSTER_VARIANTS
        base_variant, model_type, wb, bright = app._parse_variant(variant)
        assert base_variant == "whole_plus_face"
        assert model_type == family
        assert wb is False
        assert bright is False

    # Pose2ID NFC variants should also exist and parse to the same base model
    for variant, family in expected.items():
        nfc_variant = f"pose2id_nfc_{variant}"
        assert nfc_variant in app.CLUSTER_VARIANTS
        base_variant, model_type, wb, bright = app._parse_variant(nfc_variant)
        assert base_variant == "whole_plus_face"
        assert model_type == family
        assert wb is False
        assert bright is False
        assert app._variant_has_nfc(nfc_variant) is True


def test_cluster_strategy_variants_exist_for_all_model_families():
    families = {
        "dinov3": "",
        "vit": "vit_",
        "osnet": "osnet_",
        "osnet_finetuned": "osnet_finetuned_",
        "osnet_finetuned_improved": "osnet_finetuned_improved_",
        "convnext": "convnext_",
        "efficientnetv2": "efficientnetv2_",
        "deit": "deit_",
        "fastreid": "fastreid_",
    }
    base_variants = [
        "face_region",
        "person_parts",
        "upscaled_lanczos",
        "combined_upscaled",
        "upscaled_esrgan",
        "combined_upscaled_esrgan",
    ]

    for family, prefix in families.items():
        for base_variant in base_variants:
            variant = f"{prefix}{base_variant}"
            assert variant in app.CLUSTER_VARIANTS
            parsed_base, model_type, _wb, _bright = app._parse_variant(variant)
            assert parsed_base == base_variant
            assert model_type == family

            # Pose2ID NFC counterpart
            nfc_variant = f"pose2id_nfc_{variant}"
            assert nfc_variant in app.CLUSTER_VARIANTS
            parsed_base, model_type, _wb, _bright = app._parse_variant(nfc_variant)
            assert parsed_base == base_variant
            assert model_type == family
            assert app._variant_has_nfc(nfc_variant) is True


def test_realesrgan_variants_parse_with_expected_preprocessing_flags():
    assert app._parse_variant("vit_upscaled_esrgan") == ("upscaled_esrgan", "vit", False, False)
    assert app._parse_variant("vit_combined_upscaled_esrgan") == (
        "combined_upscaled_esrgan",
        "vit",
        True,
        True,
    )


def test_pose2id_nfc_variants_parse_correctly():
    assert app._parse_variant("pose2id_nfc_osnet_whole") == ("whole", "osnet", False, False)
    assert app._parse_variant("pose2id_nfc_vit_whole_wb_bright") == ("whole", "vit", True, True)
    assert app._variant_has_nfc("pose2id_nfc_osnet_whole") is True
    assert app._variant_has_nfc("osnet_whole") is False


def test_apply_nfc_centralizes_features():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    # Create 4 features: two pairs that are close to each other
    features = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.99, 0.01],
    ]
    centralized = app._apply_nfc(features, k1=1, k2=1)
    assert len(centralized) == 4
    # Features should still be L2-normalized
    for feat in centralized:
        norm = sum(v * v for v in feat) ** 0.5
        assert abs(norm - 1.0) < 1e-4


def test_cluster_testset_detect_route_persists_yolo_detections(tmp_path, monkeypatch):
    store_dir = tmp_path / "cluster_testsets"
    store_dir.mkdir()
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (64, 64), (20, 200, 20)).save(image_path)
    testset = {
        "id": "sample",
        "name": "sample",
        "images": [
            {
                "id": "img-00000",
                "path": str(image_path),
                "filename": image_path.name,
                "label": {"identity": None, "camera": None},
            }
        ],
    }
    (store_dir / "sample.json").write_text(json.dumps(testset), encoding="utf-8")

    monkeypatch.setattr(app, "CLUSTER_TESTSET_STORE_DIR", store_dir)
    monkeypatch.setattr(app, "CLUSTER_TESTSET_YOLO_MODEL", "/tmp/yolo.onnx")
    monkeypatch.setattr(app, "_load_yolo_session", lambda _model_path: object())
    monkeypatch.setattr(
        app,
        "_run_yolo_on_image",
        lambda *_args, **_kwargs: [
            {"bbox": [1.0, 2.0, 30.0, 40.0], "score": 0.91, "class_id": 0, "class_name": "person"}
        ],
    )

    result = app.run_cluster_testset_yolo_detection(
        "sample",
        app.ClusterYoloDetectionRequest(confidence=0.3, max_detections_per_image=5, classes=[0]),
    )

    saved = json.loads((store_dir / "sample.json").read_text(encoding="utf-8"))
    assert result["processed_images"] == 1
    assert result["detection_count"] == 1
    assert saved["images"][0]["detections"][0]["id"] == "img-00000-det-000"
    assert saved["images"][0]["detections"][0]["class_name"] == "person"
    assert saved["detection_config"]["classes"] == [0]


def test_cluster_testset_detect_route_supports_objects_only_preset(tmp_path, monkeypatch):
    store_dir = tmp_path / "cluster_testsets"
    store_dir.mkdir()
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (64, 64), (20, 200, 20)).save(image_path)
    testset = {
        "id": "objects-sample",
        "name": "objects-sample",
        "images": [
            {
                "id": "img-00000",
                "path": str(image_path),
                "filename": image_path.name,
                "label": {"identity": None, "camera": None},
            }
        ],
    }
    (store_dir / "objects-sample.json").write_text(json.dumps(testset), encoding="utf-8")
    seen_classes = {}

    def fake_run_yolo(*_args, **kwargs):
        seen_classes["classes"] = kwargs.get("classes")
        return [{"bbox": [1.0, 2.0, 30.0, 40.0], "score": 0.91, "class_id": 56, "class_name": "chair"}]

    monkeypatch.setattr(app, "CLUSTER_TESTSET_STORE_DIR", store_dir)
    monkeypatch.setattr(app, "CLUSTER_TESTSET_YOLO_MODEL", "/tmp/yolo.onnx")
    monkeypatch.setattr(app, "_load_yolo_session", lambda _model_path: object())
    monkeypatch.setattr(app, "_run_yolo_on_image", fake_run_yolo)

    result = app.run_cluster_testset_yolo_detection(
        "objects-sample",
        app.ClusterYoloDetectionRequest(confidence=0.3, max_detections_per_image=5, class_preset="objects"),
    )

    saved = json.loads((store_dir / "objects-sample.json").read_text(encoding="utf-8"))
    assert result["detection_count"] == 1
    assert 0 not in seen_classes["classes"]
    assert saved["images"][0]["detections"][0]["class_name"] == "chair"
    assert saved["detection_config"]["class_preset"] == "objects"
    assert 0 not in saved["detection_config"]["classes"]
