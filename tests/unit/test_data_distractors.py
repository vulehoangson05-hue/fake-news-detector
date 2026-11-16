"""Unit tests for vnfake.data.distractors (Brief #5 §11.4)."""

from __future__ import annotations

from vnfake.data.distractors import (
    DistractorPair,
    DistractorPool,
    filter_distractors_by_containment,
    generate_dev_distractors,
    generate_same_article_distractors,
)


def test_filter_distractors_by_containment() -> None:
    pairs = [
        DistractorPair(
            sample_idx=0,
            statement="Statement 1",
            statement_segmented="Statement 1",
            distractor_evidence="Đoạn văn hoàn toàn không liên quan gì đến câu bằng chứng gốc ở đây.",
            distractor_evidence_segmented="Đoạn văn hoàn toàn không liên quan gì đến câu bằng chứng gốc ở đây.",
            distractor_type="hard_bm25",
            source_article_id="art1",
            original_label=0,
            target_label=2,
            containment_with_gold=0.05,
        ),
        DistractorPair(
            sample_idx=1,
            statement="Statement 2",
            statement_segmented="Statement 2",
            distractor_evidence="Đoạn văn này có chứa đúng câu bằng chứng vàng nguyên vẹn ở đây.",
            distractor_evidence_segmented="Đoạn văn này có chứa đúng câu bằng chứng vàng nguyên vẹn ở đây.",
            distractor_type="hard_bm25",
            source_article_id="art2",
            original_label=1,
            target_label=2,
            containment_with_gold=0.75,
        ),
        DistractorPair(
            sample_idx=2,
            statement="Statement 3",
            statement_segmented="Statement 3",
            distractor_evidence="Một đoạn văn thứ ba có chứa một vài từ tương đồng khoảng hai mươi lăm phần trăm.",
            distractor_evidence_segmented="Một đoạn văn thứ ba có chứa một vài từ tương đồng khoảng hai mươi lăm phần trăm.",
            distractor_type="random",
            source_article_id="art3",
            original_label=2,
            target_label=2,
            containment_with_gold=0.25,
        ),
    ]

    filtered = filter_distractors_by_containment(pairs, max_containment=0.3)
    assert len(filtered) == 2
    assert [p.sample_idx for p in filtered] == [0, 2]

    # Test with gold_map override
    gold_map = {
        0: "Đoạn văn hoàn toàn không liên quan gì đến câu bằng chứng gốc ở đây.",  # 1.0 containment!
        1: "Câu hoàn toàn khác không trùng chữ nào cả.",  # 0.0 containment!
        2: "Một đoạn văn thứ ba không liên quan.",
    }
    filtered_map = filter_distractors_by_containment(pairs, gold_map=gold_map, max_containment=0.3)
    # Pair 0 now has containment 1.0 -> dropped
    # Pair 1 now has containment 0.0 -> kept
    # Pair 2 has containment with gold_map[2]
    assert pairs[0] not in filtered_map
    assert pairs[1] in filtered_map


def test_generate_same_article_distractors() -> None:
    records = [
        {
            "sample_idx": 10,
            "Statement": "Công ty VinFast công bố ra mắt mẫu xe điện mới tại Hà Nội.",
            "Evidence": "VinFast chính thức giới thiệu mẫu xe điện thông minh tại sự kiện triển lãm Hà Nội.",
            "Context": (
                "VinFast chính thức giới thiệu mẫu xe điện thông minh tại sự kiện triển lãm Hà Nội. "
                "Thời tiết ngày hôm đó tại thủ đô khá mát mẻ và thuận lợi cho khách tham quan. "
                "Nhiều chuyên gia công nghệ nhận định thị trường xe điện Việt Nam đang phát triển mạnh mẽ. "
                "Sự kiện thu hút hơn 5000 lượt khách tham dự trong ngày đầu tiên mở cửa."
            ),
            "labels": 0,
        }
    ]

    distractors = generate_same_article_distractors(records, max_containment=0.3)
    assert len(distractors) == 1
    p = distractors[0]
    assert p.sample_idx == 10
    assert p.distractor_type == "same_article"
    assert p.target_label == 2
    # Verify distractor evidence does NOT contain the gold evidence sentence (containment < 0.3)
    assert p.containment_with_gold < 0.3
    assert "Thời tiết" in p.distractor_evidence or "chuyên gia" in p.distractor_evidence or "5000" in p.distractor_evidence


def test_distractor_pool_and_generation() -> None:
    articles = [
        {
            "sample_idx": 0,
            "Statement": "Giá vàng thế giới tăng mạnh.",
            "Evidence": "Giá vàng thế giới hôm nay tăng vọt lên mức kỷ lục.",
            "Context": (
                "Giá vàng thế giới hôm nay tăng vọt lên mức kỷ lục do lo ngại lạm phát. "
                "Các nhà đầu tư đổ xô mua kim loại quý để bảo vệ tài sản."
            ),
            "labels": 0,
        },
        {
            "sample_idx": 1,
            "Statement": "Đội tuyển bóng đá Việt Nam giành chiến thắng.",
            "Evidence": "Đội tuyển Việt Nam thắng trận giao hữu quốc tế với tỉ số 2-0.",
            "Context": (
                "Đội tuyển Việt Nam thắng trận giao hữu quốc tế với tỉ số 2-0 trên sân Mỹ Đình. "
                "HLV trưởng tỏ ra rất hài lòng với màn trình diễn của các cầu thủ trẻ."
            ),
            "labels": 1,
        },
    ]

    pool = DistractorPool.from_records(articles)
    assert len(pool.articles) == 2

    dev_dist = generate_dev_distractors(articles, pool, seed=42)
    assert len(dev_dist) == 4  # 2 per claim: 1 random, 1 hard BM25
    assert all(p.target_label == 2 for p in dev_dist)
    assert all(p.distractor_type in ("random", "hard_bm25") for p in dev_dist)
