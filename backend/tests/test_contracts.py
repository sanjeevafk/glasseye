from app.schemas import DefectClass, Detection, DetectorFrame


def test_detector_contract_allows_explicitly_null_mask():
    frame = DetectorFrame(
        frame_id="frame-0001",
        timestamp=1.25,
        image_id="image-0001",
        model_version="glasseye-yolo-v1",
        detections=[
            Detection(
                class_name=DefectClass.CLEANABLE,
                class_id=0,
                confidence=0.91,
                bbox_xyxy=[1, 2, 30, 40],
                mask=None,
                track_id=None,
            )
        ],
    )

    assert frame.model_dump(mode="json")["detections"][0]["mask"] is None
