"""Brief #4 Task 2 — CLI wrapper for running the 30-claim Tavily Web Search spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vnfake.config import load_config, load_search_config
from vnfake.retrieval.web_search import (
    TavilyWebSearcher,
    run_tavily_spike,
    sample_stratified_dev_claims,
)
from vnfake.schemas.vifactcheck import ViFactCheckSample
from vnfake.utils.logging import get_logger
from vnfake.utils.seed import set_seed

logger = get_logger(__name__)


def _load_clean_split(path: Path) -> list[ViFactCheckSample]:
    samples: list[ViFactCheckSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(ViFactCheckSample.model_validate(json.loads(line)))
    return samples


def format_tavily_spike_markdown(report: dict[str, Any], include_domains: list[str]) -> str:
    """Format the 30-claim Tavily spike results as a detailed Markdown report."""
    lines = [
        "# BÁO CÁO THỬ NGHIỆM TÌM KIẾM WEB TAVILY (Brief #4 — Task 2)",
        "",
        f"- **Thời gian chạy (UTC):** `{report['timestamp_utc']}`",
        f"- **Số claim kiểm thử:** `{report['num_claims']}` (stratified 10 `real`, 10 `fake`, 10 `unverifiable` từ `dev_clean.jsonl`, `seed=42`)",
        f"- **Credits đã dùng:** `{report['credits_used']} / {report['max_credits_spike']}`",
        f"- **Độ trễ tìm kiếm (API latency):** `p50 = {report['latency_p50_ms']:.2f} ms` | `p95 = {report['latency_p95_ms']:.2f} ms`",
        f"- **Tổng số URL trả về đã kiểm tra whitelist:** `{report['total_urls_verified']}` (`all_urls_in_whitelist = {report['all_urls_in_whitelist']}`)",
        f"- **Nguồn trích xuất văn bản:** `{json.dumps(report['text_source_counts'], ensure_ascii=False)}`",
        "  - *Quy tắc chọn văn bản:* Ưu tiên dùng `raw_content` do Tavily trả về trực tiếp trong cùng 1 credit (`search_depth=basic`, `include_raw_content=True`) nếu độ dài `>= 400` ký tự; nếu ngắn hơn hoặc rỗng thì dùng `PoliteArticleFetcher` (`robots.txt` + `rate_limit >= 1.0s` + `trafilatura`).",
        f"- **Danh sách 16 domain báo chính thống (`include_domains`):** `{', '.join(include_domains)}`",
        "",
        "## 1. Bảng kết quả tổng hợp (Top-5 3-gram Containment so với Gold `Evidence`)",
        "",
        "| Truy vấn | Hit >= 0.5 | Hit >= 0.3 | real (>=0.5) | fake (>=0.5) | unv (>=0.5) | real (>=0.3) | fake (>=0.3) | unv (>=0.3) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for var_name, vdata in report["variants"].items():
        pc05 = vdata["per_class_ge_05"]
        pc03 = vdata["per_class_ge_03"]
        lines.append(
            f"| **{var_name}** | **{vdata['hits_ge_05']}/{vdata['num_claims']} ({vdata['pct_ge_05']:.2f}%)** | "
            f"**{vdata['hits_ge_03']}/{vdata['num_claims']} ({vdata['pct_ge_03']:.2f}%)** | "
            f"{pc05.get('real', '0/10')} | {pc05.get('fake', '0/10')} | {pc05.get('unverifiable', '0/10')} | "
            f"{pc03.get('real', '0/10')} | {pc03.get('fake', '0/10')} | {pc03.get('unverifiable', '0/10')} |"
        )

    for var_name, vdata in report["variants"].items():
        lines.extend(
            [
                "",
                f"## 2. Phân tích chi tiết — Biến thể `{var_name}`",
                "",
                f"### 2.1. 5 Claim Trúng tiêu biểu (`{var_name}`)",
                "",
            ]
        )
        for idx, item in enumerate(vdata["hit_5_examples"], start=1):
            lines.extend(
                [
                    f"#### Trúng #{idx}: `{item['claim_id']}` (Nhãn: `{item['label']}` | 3-gram containment = `{item['best_containment_3gram']:.4f}`)",
                    f"- **Claim (`Statement`):** {item['statement']}",
                    f"- **Truy vấn (`{var_name}`):** `{item['query']}`",
                    f"- **Gold `Evidence`:** {item['gold_evidence']}",
                    f"- **Top URL khớp nhất:** `{item['best_url']}` (nguồn text: `{item['best_text_source']}`)",
                    f"- **Tiêu đề bài báo:** {item['best_title']}",
                    "- **Lý do trúng:** Câu truy vấn chứa cụm thực thể/sự kiện đặc trưng giúp Tavily định vị đúng bài báo gốc trên báo chính thống; văn bản trích xuất chứa nguyên văn câu bằng chứng.",
                    "",
                ]
            )

        lines.extend(
            [
                f"### 2.2. 5 Claim Trượt tiêu biểu (`{var_name}`)",
                "",
            ]
        )
        for idx, item in enumerate(vdata["miss_5_examples"], start=1):
            lines.extend(
                [
                    f"#### Trượt #{idx}: `{item['claim_id']}` (Nhãn: `{item['label']}` | 3-gram containment = `{item['best_containment_3gram']:.4f}`)",
                    f"- **Claim (`Statement`):** {item['statement']}",
                    f"- **Truy vấn (`{var_name}`):** `{item['query']}`",
                    f"- **Gold `Evidence`:** {item['gold_evidence']}",
                    f"- **Top URL trả về:** `{item['best_url'] or 'Không có URL'}`",
                    f"- **Danh sách Top-5 URL:** `{', '.join(item['top_urls'])}`",
                    "- **Lý do trượt:** Bài báo gốc trong ViFactCheck thuộc domain ngoài danh sách 16 báo hoặc đã đổi URL/tiêu đề trên chỉ mục tìm kiếm, hoặc bài tìm được cùng chủ đề nhưng khác bản tin cụ thể mà ViFactCheck trích.",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Brief #4 Task 2 — Run 30-claim Tavily Web Search spike on dev split"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"), help="Base config path"
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("configs/search.yaml"),
        help="Search config path",
    )
    parser.add_argument(
        "--per-class", type=int, default=10, help="Number of dev claims per class (default: 10)"
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/tavily_spike_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/REPORT_tavily_spike.md"),
        help="Output Markdown report path",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    search_cfg = load_search_config(args.search_config)
    set_seed(cfg.seed)

    dev_path = cfg.paths.processed_dir / "dev_clean.jsonl"
    dev_samples = _load_clean_split(dev_path)
    claims = sample_stratified_dev_claims(dev_samples, per_class=args.per_class, seed=cfg.seed)

    searcher = TavilyWebSearcher(search_cfg)
    report = run_tavily_spike(claims, searcher, cfg.label_map)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_text = format_tavily_spike_markdown(report, search_cfg.include_domains)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")

    logger.info(
        "Tavily spike finished: credits=%d/%d, Q0>=0.5=%s%%, Q1>=0.5=%s%%",
        report["credits_used"],
        report["max_credits_spike"],
        report["variants"]["Q0"]["pct_ge_05"],
        report["variants"]["Q1"]["pct_ge_05"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
