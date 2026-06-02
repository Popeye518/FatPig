import argparse
import base64
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


class ConfluenceClient:
    def __init__(self, pat):
        self.pat = pat
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {pat}",
            "Accept": "application/json"
        })

    def extract_page_id(self, page_url):
        parsed = urlparse(page_url)
        query = parse_qs(parsed.query)

        if "pageId" in query:
            return query["pageId"][0]

        match = re.search(r"/pages/(\d+)", parsed.path)
        if match:
            return match.group(1)

        match = re.search(r"/pageId=(\d+)", page_url)
        if match:
            return match.group(1)

        numbers = re.findall(r"/(\d+)(?:/|$)", parsed.path)
        if numbers:
            return numbers[-1]

        raise ValueError(f"Could not extract pageId from URL: {page_url}")

    def get_base_candidates(self, page_url):
        parsed = urlparse(page_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        candidates = []
        if "/wiki/" in parsed.path:
            candidates.append(f"{origin}/wiki/rest/api")
        candidates.append(f"{origin}/rest/api")

        unique = []
        for item in candidates:
            if item not in unique:
                unique.append(item)
        return unique

    def request_json(self, url):
        response = self.session.get(url, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"Request failed: {response.status_code} {response.text[:500]}")
        return response.json()

    def fetch_page(self, page_url):
        page_id = self.extract_page_id(page_url)
        last_error = None

        for api_base in self.get_base_candidates(page_url):
            try:
                url = f"{api_base}/content/{page_id}?expand=body.storage,version,space,ancestors"
                data = self.request_json(url)
                data["_api_base"] = api_base
                data["_page_id"] = page_id
                data["_source_url"] = page_url
                return data
            except Exception as ex:
                last_error = ex

        raise RuntimeError(f"Could not fetch Confluence page. Last error: {last_error}")

    def fetch_attachments(self, page):
        api_base = page["_api_base"]
        page_id = page["_page_id"]
        attachments = []
        start = 0
        limit = 100

        while True:
            url = f"{api_base}/content/{page_id}/child/attachment?limit={limit}&start={start}&expand=version"
            data = self.request_json(url)
            results = data.get("results", [])
            attachments.extend(results)

            size = data.get("size", len(results))
            if size < limit:
                break

            start += limit

        return attachments

    def download_url(self, page, url):
        if not url:
            return None

        parsed_page = urlparse(page["_source_url"])
        origin = f"{parsed_page.scheme}://{parsed_page.netloc}"

        if url.startswith("/"):
            download_url = origin + url
        else:
            download_url = url

        response = self.session.get(download_url, timeout=90)
        if response.status_code >= 400:
            return None
        return response.content

    def download_attachment_by_name(self, page, attachments, filename):
        if not filename:
            return None

        for attachment in attachments:
            title = attachment.get("title", "")
            if title == filename:
                links = attachment.get("_links", {})
                download_link = links.get("download")
                return self.download_url(page, download_link)

        normalized = filename.strip().lower()
        for attachment in attachments:
            title = attachment.get("title", "").strip().lower()
            if title == normalized:
                links = attachment.get("_links", {})
                download_link = links.get("download")
                return self.download_url(page, download_link)

        return None


class ConfluenceParser:
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def page_html(self, page):
        return page.get("body", {}).get("storage", {}).get("value", "") or ""

    def parse_sections(self, html):
        soup = BeautifulSoup(html, "html.parser")
        sections = []
        current = None

        for node in soup.find_all(True):
            tag = node.name.lower() if node.name else ""

            if tag in self.HEADING_TAGS:
                heading = self.clean_text(node.get_text(" ", strip=True))
                if heading:
                    current = {
                        "heading": heading,
                        "level": tag,
                        "text_parts": [],
                    }
                    sections.append(current)
                continue

            if current and tag in {"p", "li", "td", "th"}:
                text = self.clean_text(node.get_text(" ", strip=True))
                if text and text not in current["text_parts"]:
                    current["text_parts"].append(text)

        for section in sections:
            section["text"] = self.clean_text(" ".join(section["text_parts"]))
            section.pop("text_parts", None)

        if not sections:
            text = self.clean_text(soup.get_text(" ", strip=True))
            sections.append({
                "heading": "Page Content",
                "level": "h1",
                "text": text
            })

        return sections

    def extract_template_expectations(self, html):
        sections = self.parse_sections(html)
        expectations = []

        for section in sections:
            heading = section.get("heading", "")
            text = section.get("text", "")
            if not heading:
                continue

            items = self.extract_expected_items_from_text(text)

            expectations.append({
                "topic": heading,
                "expected_details": items,
                "raw_text": text
            })

        return expectations

    def extract_expected_items_from_text(self, text):
        if not text:
            return []

        candidates = []

        splitters = re.split(r"[.;]\s+|\n+|•|\u2022", text)
        for item in splitters:
            item = self.clean_text(item)
            if len(item) >= 8:
                candidates.append(item)

        deduped = []
        seen = set()
        for item in candidates:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped[:20]

    def nearest_section_for_node(self, node):
        previous = node
        while previous:
            previous = previous.find_previous()
            if previous and previous.name and previous.name.lower() in self.HEADING_TAGS:
                return self.clean_text(previous.get_text(" ", strip=True))
        return "Unmapped Diagram"

    def surrounding_text_for_node(self, node):
        heading = None
        previous = node

        while previous:
            previous = previous.find_previous()
            if previous and previous.name and previous.name.lower() in self.HEADING_TAGS:
                heading = previous
                break

        parts = []
        cursor = heading.find_next_sibling() if heading else node.find_previous()
        while cursor and cursor != node:
            if getattr(cursor, "name", None):
                tag = cursor.name.lower()
                if tag in self.HEADING_TAGS:
                    break
                if tag in {"p", "li", "td", "th", "div"}:
                    text = self.clean_text(cursor.get_text(" ", strip=True))
                    if text:
                        parts.append(text)
            cursor = cursor.find_next_sibling()

        if not parts:
            for prev in node.find_all_previous(["p", "li"], limit=5):
                text = self.clean_text(prev.get_text(" ", strip=True))
                if text:
                    parts.append(text)

        return self.clean_text(" ".join(reversed(parts[-8:])))

    def extract_filename_from_ac_image(self, image_node):
        for child in image_node.find_all(True):
            name = child.name or ""
            attrs = child.attrs or {}

            if name.endswith("attachment"):
                for key, value in attrs.items():
                    if key.endswith("filename"):
                        return value

            for key, value in attrs.items():
                if key.endswith("filename"):
                    return value

        return None

    def extract_images_with_headings(self, html):
        soup = BeautifulSoup(html, "html.parser")
        diagrams = []
        counter = 1

        for node in soup.find_all(True):
            tag = node.name or ""
            tag_lower = tag.lower()

            is_ac_image = tag_lower.endswith("image")
            is_html_img = tag_lower == "img"

            if not is_ac_image and not is_html_img:
                continue

            heading = self.nearest_section_for_node(node)
            context = self.surrounding_text_for_node(node)

            src = None
            filename = None

            if is_html_img:
                src = node.get("src")
                filename = node.get("data-linked-resource-default-alias") or node.get("alt")

            if is_ac_image:
                filename = self.extract_filename_from_ac_image(node)

            diagrams.append({
                "diagram_id": f"diagram_{counter:03d}",
                "heading": heading,
                "context_text": context,
                "image_src": src,
                "attachment_filename": filename
            })
            counter += 1

        return diagrams


class GeminiAnalyzer:
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.enabled = bool(api_key and genai and types)

        if self.enabled:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def safe_json_from_text(self, text):
        if not text:
            return None

        text = text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

        return None

    def analyze_diagram(self, diagram, image_bytes, template_expectations):
        if not self.enabled:
            return self.heuristic_analysis(diagram, template_expectations)

        mime_type = self.detect_mime_type(image_bytes)
        template_text = json.dumps(template_expectations, indent=2, ensure_ascii=False)[:30000]

        prompt = f"""
You are reviewing a solution design document diagram from Confluence.

Return only valid JSON. Do not include markdown.

Evidence diagram information:
- Diagram ID: {diagram.get("diagram_id")}
- Heading: {diagram.get("heading")}
- Nearby evidence text: {diagram.get("context_text")}

Template expectations:
{template_text}

Tasks:
1. Explain the architecture or diagram.
2. Identify visible components.
3. Identify visible data flow or control flow.
4. Identify integrations, dependencies, database, deployment, network, security, and environment details if visible.
5. Compare the diagram and nearby evidence text with the relevant template expectations.
6. Mark each required detail as Present, Partially Present, Missing, Unclear, or Not Applicable.
7. Assess quality using clarity, labeling, flow direction, technical detail, consistency, and review readiness.
8. Do not assume details that are not visible in the diagram or not mentioned in the evidence text.

Required JSON schema:
{{
  "diagram_explanation": "",
  "components_identified": [],
  "flows_identified": [],
  "integrations_identified": [],
  "security_details_identified": [],
  "deployment_details_identified": [],
  "database_details_identified": [],
  "completeness_checks": [
    {{
      "required_detail": "",
      "status": "",
      "evidence_found": "",
      "missing_details": [],
      "remarks": ""
    }}
  ],
  "quality": {{
    "clarity_score": 0,
    "labeling_score": 0,
    "flow_score": 0,
    "technical_detail_score": 0,
    "consistency_score": 0,
    "review_readiness_score": 0,
    "overall_quality_score": 0,
    "rating": "",
    "strengths": [],
    "issues": [],
    "recommendations": []
  }},
  "confidence": ""
}}
"""

        try:
            contents = [prompt]

            if image_bytes:
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )

            parsed = self.safe_json_from_text(response.text)
            if parsed:
                return parsed

            fallback = self.heuristic_analysis(diagram, template_expectations)
            fallback["diagram_explanation"] = response.text[:3000] if response.text else fallback["diagram_explanation"]
            return fallback

        except Exception as ex:
            fallback = self.heuristic_analysis(diagram, template_expectations)
            fallback["analysis_error"] = str(ex)
            return fallback

    def detect_mime_type(self, image_bytes):
        if not image_bytes:
            return "image/png"

        if image_bytes.startswith(b"\x89PNG"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if image_bytes.startswith(b"GIF"):
            return "image/gif"
        if image_bytes[:4] == b"RIFF" and b"WEBP" in image_bytes[:20]:
            return "image/webp"

        return "image/png"

    def heuristic_analysis(self, diagram, template_expectations):
        heading = diagram.get("heading", "")
        context = diagram.get("context_text", "")
        combined = f"{heading} {context}".lower()

        checks = []
        for expectation in template_expectations:
            topic = expectation.get("topic", "")
            expected_details = expectation.get("expected_details", [])

            topic_key = topic.lower()
            if topic_key and topic_key in combined:
                status = "Partially Present" if context else "Unclear"
            else:
                status = "Unclear"

            if expected_details:
                for item in expected_details[:5]:
                    item_text = item.lower()
                    words = [w for w in re.findall(r"[a-zA-Z0-9]+", item_text) if len(w) > 4]
                    matched = any(word in combined for word in words)

                    checks.append({
                        "required_detail": f"{topic} - {item[:120]}",
                        "status": "Partially Present" if matched else "Unclear",
                        "evidence_found": context[:500] if matched else "",
                        "missing_details": [] if matched else [item[:200]],
                        "remarks": "Heuristic check used because Gemini API was not available."
                    })
            else:
                checks.append({
                    "required_detail": topic,
                    "status": status,
                    "evidence_found": context[:500] if context else "",
                    "missing_details": [],
                    "remarks": "Heuristic check used because Gemini API was not available."
                })

        if not checks:
            checks.append({
                "required_detail": "Template expectation",
                "status": "Unclear",
                "evidence_found": context[:500],
                "missing_details": [],
                "remarks": "No structured template expectation was extracted."
            })

        return {
            "diagram_explanation": (
                f"The diagram is mapped under heading '{heading}'. "
                f"Gemini analysis was not available, so the explanation is based on nearby Confluence text."
            ),
            "components_identified": [],
            "flows_identified": [],
            "integrations_identified": [],
            "security_details_identified": [],
            "deployment_details_identified": [],
            "database_details_identified": [],
            "completeness_checks": checks,
            "quality": {
                "clarity_score": 50,
                "labeling_score": 50,
                "flow_score": 50,
                "technical_detail_score": 50,
                "consistency_score": 50,
                "review_readiness_score": 50,
                "overall_quality_score": 50,
                "rating": "Average",
                "strengths": ["Diagram was found in the evidence page."],
                "issues": ["Image-level analysis was not available without Gemini."],
                "recommendations": ["Provide Gemini API key for stronger diagram explanation and quality review."]
            },
            "confidence": "low"
        }


class ReportBuilder:
    def __init__(self, output_json, output_pdf):
        self.output_json = output_json
        self.output_pdf = output_pdf

    def compute_completeness_score(self, topics):
        score_map = {
            "present": 1.0,
            "partially present": 0.5,
            "unclear": 0.25,
            "missing": 0.0,
        }

        total = 0
        achieved = 0

        for topic in topics:
            checks = topic.get("required_details_check", [])
            for check in checks:
                status = str(check.get("status", "")).strip().lower()
                if status == "not applicable":
                    continue

                total += 1
                achieved += score_map.get(status, 0)

        if total == 0:
            return 0

        return round((achieved / total) * 100, 2)

    def compute_quality_score(self, topics):
        scores = []

        for topic in topics:
            quality = topic.get("quality", {})
            value = quality.get("overall_quality_score")
            try:
                scores.append(float(value))
            except Exception:
                pass

        if not scores:
            return 0

        return round(sum(scores) / len(scores), 2)

    def overall_status(self, completeness_score, quality_score):
        average = (completeness_score + quality_score) / 2

        if average >= 85:
            return "Excellent"
        if average >= 70:
            return "Good"
        if average >= 50:
            return "Needs Improvement"
        return "Poor"

    def build_json_report(self, evidence_page, template_page, topics, output_dir):
        completeness_score = self.compute_completeness_score(topics)
        quality_score = self.compute_quality_score(topics)

        missing_items = []
        recommendations = []

        for topic in topics:
            for check in topic.get("required_details_check", []):
                status = str(check.get("status", "")).lower()
                if status in {"missing", "partially present", "unclear"}:
                    missing_items.append({
                        "topic": topic.get("topic_name"),
                        "required_detail": check.get("required_detail"),
                        "status": check.get("status"),
                        "missing_details": check.get("missing_details", []),
                        "remarks": check.get("remarks", "")
                    })

            for recommendation in topic.get("quality", {}).get("recommendations", []):
                recommendations.append({
                    "topic": topic.get("topic_name"),
                    "recommendation": recommendation
                })

        report = {
            "metadata": {
                "evidence_page_title": evidence_page.get("title", ""),
                "template_page_title": template_page.get("title", ""),
                "evidence_page_url": evidence_page.get("_source_url", ""),
                "template_page_url": template_page.get("_source_url", ""),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": self.overall_status(completeness_score, quality_score),
                "overall_completeness_score": completeness_score,
                "overall_quality_score": quality_score
            },
            "topics": topics,
            "overall_missing_items": missing_items,
            "overall_recommendations": recommendations
        }

        output_path = Path(output_dir) / self.output_json
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        return report, output_path

    def p(self, text):
        if text is None:
            return ""
        text = str(text)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return text

    def build_pdf_report(self, report, output_dir):
        output_path = Path(output_dir) / self.output_pdf

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_LEFT
        ))
        styles.add(ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceAfter=8
        ))

        story = []

        metadata = report.get("metadata", {})

        story.append(Paragraph("Architecture Diagram Review Report", styles["Title"]))
        story.append(Spacer(1, 12))

        summary_data = [
            ["Evidence Page", self.p(metadata.get("evidence_page_title", ""))],
            ["Template Page", self.p(metadata.get("template_page_title", ""))],
            ["Analysis Date", self.p(metadata.get("analysis_date", ""))],
            ["Overall Status", self.p(metadata.get("overall_status", ""))],
            ["Completeness Score", str(metadata.get("overall_completeness_score", 0))],
            ["Quality Score", str(metadata.get("overall_quality_score", 0))]
        ]

        summary_table = Table(summary_data, colWidths=[1.8 * inch, 4.8 * inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F2F5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 18))

        story.append(Paragraph("Overall Missing Items", styles["SectionTitle"]))
        missing_items = report.get("overall_missing_items", [])

        if missing_items:
            table_rows = [["Topic", "Required Detail", "Status", "Missing / Remarks"]]
            for item in missing_items[:50]:
                missing_text = "; ".join(item.get("missing_details", [])) or item.get("remarks", "")
                table_rows.append([
                    Paragraph(self.p(item.get("topic", "")), styles["Small"]),
                    Paragraph(self.p(item.get("required_detail", "")), styles["Small"]),
                    Paragraph(self.p(item.get("status", "")), styles["Small"]),
                    Paragraph(self.p(missing_text), styles["Small"])
                ])

            table = Table(table_rows, colWidths=[1.2 * inch, 2.1 * inch, 1.0 * inch, 2.3 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No missing items were identified.", styles["BodyText"]))

        story.append(PageBreak())

        for topic in report.get("topics", []):
            story.append(Paragraph(self.p(topic.get("topic_name", "")), styles["Heading1"]))
            story.append(Paragraph(f"Diagram ID: {self.p(topic.get('diagram_id', ''))}", styles["Small"]))
            story.append(Paragraph(f"Diagram Available: {self.p(topic.get('diagram_available', ''))}", styles["Small"]))
            story.append(Spacer(1, 8))

            image_path = topic.get("local_image_path")
            if image_path and os.path.exists(image_path):
                try:
                    story.append(self.pdf_image(image_path))
                    story.append(Spacer(1, 8))
                except Exception:
                    story.append(Paragraph("Diagram image could not be rendered in PDF.", styles["Small"]))

            story.append(Paragraph("Diagram Explanation", styles["SectionTitle"]))
            story.append(Paragraph(self.p(topic.get("diagram_explanation", "")), styles["BodyText"]))
            story.append(Spacer(1, 10))

            story.append(Paragraph("Required Details Check", styles["SectionTitle"]))

            checks = topic.get("required_details_check", [])
            if checks:
                rows = [["Required Detail", "Status", "Evidence Found", "Missing Details / Remarks"]]
                for check in checks[:40]:
                    missing = "; ".join(check.get("missing_details", [])) or check.get("remarks", "")
                    rows.append([
                        Paragraph(self.p(check.get("required_detail", "")), styles["Small"]),
                        Paragraph(self.p(check.get("status", "")), styles["Small"]),
                        Paragraph(self.p(check.get("evidence_found", "")), styles["Small"]),
                        Paragraph(self.p(missing), styles["Small"])
                    ])

                check_table = Table(rows, colWidths=[1.8 * inch, 0.9 * inch, 2.0 * inch, 1.9 * inch])
                check_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5E9")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]))
                story.append(check_table)
            else:
                story.append(Paragraph("No required detail checks were generated.", styles["BodyText"]))

            story.append(Spacer(1, 12))
            story.append(Paragraph("Quality Review", styles["SectionTitle"]))

            quality = topic.get("quality", {})
            quality_rows = [
                ["Overall Quality Score", str(quality.get("overall_quality_score", ""))],
                ["Rating", self.p(quality.get("rating", ""))],
                ["Strengths", self.p("; ".join(quality.get("strengths", [])))],
                ["Issues", self.p("; ".join(quality.get("issues", [])))],
                ["Recommendations", self.p("; ".join(quality.get("recommendations", [])))]
            ]

            quality_table = Table(quality_rows, colWidths=[1.8 * inch, 4.8 * inch])
            quality_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FFF3CD")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(quality_table)
            story.append(PageBreak())

        doc.build(story)
        return output_path

    def pdf_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size

        max_width = 6.4 * inch
        max_height = 4.5 * inch

        ratio = min(max_width / width, max_height / height)
        display_width = width * ratio
        display_height = height * ratio

        return ReportImage(image_path, width=display_width, height=display_height)


def sanitize_filename(name):
    name = name or "diagram"
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:120]


def save_image_bytes(image_bytes, output_dir, diagram_id):
    if not image_bytes:
        return None

    try:
        img = Image.open(BytesIO(image_bytes))
        ext = img.format.lower() if img.format else "png"
        if ext == "jpeg":
            ext = "jpg"

        path = Path(output_dir) / f"{diagram_id}.{ext}"
        img.save(path)
        return str(path)
    except Exception:
        path = Path(output_dir) / f"{diagram_id}.bin"
        with open(path, "wb") as file:
            file.write(image_bytes)
        return str(path)


def resolve_diagram_image(client, page, attachments, diagram):
    filename = diagram.get("attachment_filename")
    src = diagram.get("image_src")

    image_bytes = None

    if filename:
        image_bytes = client.download_attachment_by_name(page, attachments, filename)

    if not image_bytes and src:
        image_bytes = client.download_url(page, src)

    return image_bytes


def build_topics(evidence_page, template_expectations, diagrams, analyzer, client, attachments, output_dir):
    topics = []

    for diagram in diagrams:
        image_bytes = resolve_diagram_image(client, evidence_page, attachments, diagram)
        local_image_path = save_image_bytes(image_bytes, output_dir, diagram.get("diagram_id"))

        analysis = analyzer.analyze_diagram(
            diagram=diagram,
            image_bytes=image_bytes,
            template_expectations=template_expectations
        )

        topic = {
            "topic_name": diagram.get("heading", "Unmapped Diagram"),
            "diagram_id": diagram.get("diagram_id"),
            "diagram_heading": diagram.get("heading"),
            "diagram_available": bool(image_bytes),
            "diagram_attachment_filename": diagram.get("attachment_filename"),
            "diagram_image_src": diagram.get("image_src"),
            "local_image_path": local_image_path,
            "nearby_evidence_text": diagram.get("context_text", ""),
            "diagram_explanation": analysis.get("diagram_explanation", ""),
            "components_identified": analysis.get("components_identified", []),
            "flows_identified": analysis.get("flows_identified", []),
            "integrations_identified": analysis.get("integrations_identified", []),
            "security_details_identified": analysis.get("security_details_identified", []),
            "deployment_details_identified": analysis.get("deployment_details_identified", []),
            "database_details_identified": analysis.get("database_details_identified", []),
            "required_details_check": analysis.get("completeness_checks", []),
            "missing_expected_details": [
                check for check in analysis.get("completeness_checks", [])
                if str(check.get("status", "")).lower() in {"missing", "partially present", "unclear"}
            ],
            "quality": analysis.get("quality", {}),
            "confidence": analysis.get("confidence", ""),
        }

        if "analysis_error" in analysis:
            topic["analysis_error"] = analysis["analysis_error"]

        topics.append(topic)

    if not topics:
        topics.append({
            "topic_name": "No Diagram Found",
            "diagram_id": "diagram_000",
            "diagram_heading": "",
            "diagram_available": False,
            "diagram_attachment_filename": "",
            "diagram_image_src": "",
            "local_image_path": None,
            "nearby_evidence_text": "",
            "diagram_explanation": "No diagrams were found in the evidence Confluence page.",
            "components_identified": [],
            "flows_identified": [],
            "integrations_identified": [],
            "security_details_identified": [],
            "deployment_details_identified": [],
            "database_details_identified": [],
            "required_details_check": [
                {
                    "required_detail": "Architecture or design diagram",
                    "status": "Missing",
                    "evidence_found": "",
                    "missing_details": ["No image or diagram was extracted from the evidence page."],
                    "remarks": "Check whether diagrams are embedded as attachments, external images, draw.io macros, or unsupported macros."
                }
            ],
            "missing_expected_details": [
                {
                    "required_detail": "Architecture or design diagram",
                    "status": "Missing",
                    "missing_details": ["No image or diagram was extracted from the evidence page."]
                }
            ],
            "quality": {
                "clarity_score": 0,
                "labeling_score": 0,
                "flow_score": 0,
                "technical_detail_score": 0,
                "consistency_score": 0,
                "review_readiness_score": 0,
                "overall_quality_score": 0,
                "rating": "Poor",
                "strengths": [],
                "issues": ["No diagram found."],
                "recommendations": ["Add architecture or design diagrams to the evidence page."]
            },
            "confidence": "high"
        })

    return topics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-url", required=True)
    parser.add_argument("--template-url", required=True)
    parser.add_argument("--confluence-pat", default=os.getenv("CONFLUENCE_PAT"))
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--output-dir", default="confluence_review_output")
    parser.add_argument("--output-json", default="architecture_review_report.json")
    parser.add_argument("--output-pdf", default="architecture_review_report.pdf")
    args = parser.parse_args()

    if not args.confluence_pat:
        print("Missing Confluence PAT. Pass --confluence-pat or set CONFLUENCE_PAT.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ConfluenceClient(args.confluence_pat)
    parser_obj = ConfluenceParser()
    analyzer = GeminiAnalyzer(api_key=args.gemini_api_key, model=args.gemini_model)
    report_builder = ReportBuilder(args.output_json, args.output_pdf)

    print("Fetching evidence page...")
    evidence_page = client.fetch_page(args.evidence_url)

    print("Fetching template page...")
    template_page = client.fetch_page(args.template_url)

    evidence_html = parser_obj.page_html(evidence_page)
    template_html = parser_obj.page_html(template_page)

    print("Fetching evidence attachments...")
    evidence_attachments = client.fetch_attachments(evidence_page)

    print("Extracting diagrams from evidence page...")
    diagrams = parser_obj.extract_images_with_headings(evidence_html)

    print("Extracting expectations from template page...")
    template_expectations = parser_obj.extract_template_expectations(template_html)

    template_expectations_path = output_dir / "template_expectations.json"
    with open(template_expectations_path, "w", encoding="utf-8") as file:
        json.dump(template_expectations, file, indent=2, ensure_ascii=False)

    print("Analyzing diagrams...")
    topics = build_topics(
        evidence_page=evidence_page,
        template_expectations=template_expectations,
        diagrams=diagrams,
        analyzer=analyzer,
        client=client,
        attachments=evidence_attachments,
        output_dir=output_dir
    )

    print("Generating JSON report...")
    report, json_path = report_builder.build_json_report(
        evidence_page=evidence_page,
        template_page=template_page,
        topics=topics,
        output_dir=output_dir
    )

    print("Generating PDF report...")
    pdf_path = report_builder.build_pdf_report(report, output_dir=output_dir)

    print("")
    print("Done.")
    print(f"JSON report: {json_path}")
    print(f"PDF report: {pdf_path}")
    print(f"Template expectations: {template_expectations_path}")
    print(f"Diagrams found: {len(diagrams)}")
    print(f"Gemini enabled: {analyzer.enabled}")


if __name__ == "__main__":
    main()


pip install requests beautifulsoup4 reportlab pillow google-genai



python confluence_diagram_review.py ^
  --evidence-url "YOUR_EVIDENCE_CONFLUENCE_PAGE_URL" ^
  --template-url "YOUR_TEMPLATE_CONFLUENCE_PAGE_URL" ^
  --confluence-pat "YOUR_CONFLUENCE_PAT" ^
  --gemini-api-key "YOUR_GEMINI_API_KEY"



set CONFLUENCE_PAT=your_pat
set GEMINI_API_KEY=your_key

 confluence_diagram_review.py --evidence-url "..." --template-url "..."
