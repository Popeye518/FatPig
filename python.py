
import argparse
import json
import os
import re
import sys
import urllib3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
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


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SSL_VERIFY = False
CONFLUENCE_PAT_ENV_NAME = "CONFLUENCE_PAT"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


DEFAULT_EXPECTATIONS_BY_TOPIC = {
    "architecture": [
        "Main application components are shown",
        "Request or data flow is clear",
        "Integration points are shown",
        "Database or storage layer is shown",
        "Security and deployment considerations are visible or explained",
    ],
    "logical": [
        "Application components are shown",
        "Component relationships are shown",
        "Logical flow between components is clear",
        "Internal and external boundaries are clear",
    ],
    "functional": [
        "Major functional modules are shown",
        "Functional flow is clear",
        "User or system interactions are shown",
        "Responsibilities of each module are understandable",
    ],
    "integration": [
        "Upstream systems are shown",
        "Downstream systems are shown",
        "Integration protocols or channels are mentioned",
        "Data exchange direction is clear",
        "External dependencies are identified",
    ],
    "deployment": [
        "Hosting environment is shown",
        "Application runtime or server layer is shown",
        "Database or storage layer is shown",
        "Environment details such as dev, test, prod, cloud, or on-prem are mentioned",
        "Deployment topology is understandable",
    ],
    "security": [
        "Authentication mechanism is mentioned",
        "Authorization or access control is mentioned",
        "Security boundary is shown",
        "Sensitive data protection or encryption is mentioned",
        "Network security controls are shown if applicable",
    ],
    "database": [
        "Database entities or data stores are shown",
        "ER or data relationship is shown if expected",
        "Tables or major data objects are identified",
        "Data flow to and from database is clear",
    ],
    "network": [
        "Network zones or boundaries are shown",
        "Inbound and outbound communication paths are clear",
        "Gateway, firewall, load balancer, ports, or protocols are mentioned if applicable",
    ],
    "environment": [
        "Environment details are mentioned",
        "Application hosting location is shown",
        "Deployment or runtime environment is clear",
    ],
    "disaster": [
        "Backup or recovery strategy is mentioned",
        "DR environment or failover approach is shown",
        "Availability or resiliency design is explained",
    ],
}


class ConfluenceClient:
    def __init__(self, pat):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {pat}",
                "Accept": "application/json",
            }
        )

    def get_origin(self, page_url):
        parsed = urlparse(page_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get_api_bases(self, page_url):
        parsed = urlparse(page_url)
        origin = self.get_origin(page_url)

        bases = []

        if "/wiki/" in parsed.path:
            bases.append(f"{origin}/wiki/rest/api")

        bases.append(f"{origin}/rest/api")

        unique = []

        for base in bases:
            if base not in unique:
                unique.append(base)

        return unique

    def request_json(self, url):
        response = self.session.get(url, timeout=120, verify=SSL_VERIFY)

        if response.status_code >= 400:
            raise RuntimeError(f"Request failed {response.status_code}: {response.text[:1000]}")

        return response.json()

    def request_bytes(self, url):
        response = self.session.get(url, timeout=180, verify=SSL_VERIFY)

        if response.status_code >= 400:
            return None

        return response.content

    def extract_page_id_from_url(self, page_url):
        parsed = urlparse(page_url)
        query = parse_qs(parsed.query)

        if "pageId" in query:
            return query["pageId"][0]

        patterns = [
            r"/pages/(\d+)",
            r"[?&]pageId=(\d+)",
            r"/(\d+)(?:/|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, page_url)

            if match:
                return match.group(1)

        return None

    def extract_space_and_title_from_display_url(self, page_url):
        parsed = urlparse(page_url)
        match = re.search(r"/display/([^/]+)/(.+)$", parsed.path)

        if not match:
            return None, None

        space_key = unquote(match.group(1))
        title = unquote(match.group(2)).replace("+", " ")

        return space_key, title

    def fetch_page_by_id(self, page_url, page_id):
        last_error = None

        for api_base in self.get_api_bases(page_url):
            try:
                url = f"{api_base}/content/{page_id}?expand=body.storage,body.view,version,space,ancestors"
                data = self.request_json(url)
                data["_api_base"] = api_base
                data["_page_id"] = str(data.get("id", page_id))
                data["_source_url"] = page_url
                return data
            except Exception as ex:
                last_error = ex

        raise RuntimeError(f"Could not fetch page by id. Last error: {last_error}")

    def fetch_page_by_space_and_title(self, page_url, space_key, title):
        last_error = None

        for api_base in self.get_api_bases(page_url):
            try:
                safe_space = requests.utils.quote(space_key)
                safe_title = requests.utils.quote(title)

                url = (
                    f"{api_base}/content"
                    f"?spaceKey={safe_space}"
                    f"&title={safe_title}"
                    f"&expand=body.storage,body.view,version,space,ancestors"
                )

                data = self.request_json(url)
                results = data.get("results", [])

                if not results:
                    raise RuntimeError("No page found for this space/title")

                page = results[0]
                page["_api_base"] = api_base
                page["_page_id"] = str(page.get("id"))
                page["_source_url"] = page_url

                return page

            except Exception as ex:
                last_error = ex

        raise RuntimeError(f"Could not fetch page by space/title. Last error: {last_error}")

    def fetch_page(self, page_url):
        page_id = self.extract_page_id_from_url(page_url)

        if page_id:
            return self.fetch_page_by_id(page_url, page_id)

        space_key, title = self.extract_space_and_title_from_display_url(page_url)

        if space_key and title:
            return self.fetch_page_by_space_and_title(page_url, space_key, title)

        raise ValueError("Could not extract pageId or space/title from Confluence URL")

    def fetch_attachments(self, page):
        api_base = page["_api_base"]
        page_id = page["_page_id"]

        attachments = []
        start = 0
        limit = 100

        while True:
            url = (
                f"{api_base}/content/{page_id}/child/attachment"
                f"?limit={limit}&start={start}&expand=version,metadata"
            )

            data = self.request_json(url)
            results = data.get("results", [])
            attachments.extend(results)

            if len(results) < limit:
                break

            start += limit

        return attachments

    def absolute_url(self, page, url):
        if not url:
            return None

        if url.startswith("http://") or url.startswith("https://"):
            return url

        origin = self.get_origin(page["_source_url"])

        if url.startswith("/"):
            return origin + url

        return origin + "/" + url

    def download_url(self, page, url):
        final_url = self.absolute_url(page, url)

        if not final_url:
            return None

        return self.request_bytes(final_url)

    def download_attachment_by_name(self, page, attachments, filename):
        if not filename:
            return None

        filename = str(filename).strip()
        filename_lower = filename.lower()

        for attachment in attachments:
            title = str(attachment.get("title", "")).strip()

            if title == filename or title.lower() == filename_lower:
                download_link = attachment.get("_links", {}).get("download")
                return self.download_url(page, download_link)

        return None


class ConfluenceParser:
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def clean_text(self, value):
        if value is None:
            return ""

        value = re.sub(r"\s+", " ", str(value))
        return value.strip()

    def get_storage_html(self, page):
        return page.get("body", {}).get("storage", {}).get("value", "") or ""

    def get_view_html(self, page):
        return page.get("body", {}).get("view", {}).get("value", "") or ""

    def get_html(self, page):
        storage_html = self.get_storage_html(page)
        view_html = self.get_view_html(page)

        if storage_html and view_html:
            return storage_html + "\n" + view_html

        return storage_html or view_html

    def parse_sections(self, html):
        soup = BeautifulSoup(html or "", "html.parser")
        sections = []
        current = None

        for node in soup.find_all(True):
            tag = (node.name or "").lower()

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

            if current and tag in {"p", "li", "td", "th", "div"}:
                text = self.clean_text(node.get_text(" ", strip=True))

                if text and text not in current["text_parts"]:
                    current["text_parts"].append(text)

        if not sections:
            page_text = self.clean_text(soup.get_text(" ", strip=True))
            return [{"heading": "Page Content", "level": "h1", "text": page_text}]

        for section in sections:
            section["text"] = self.clean_text(" ".join(section["text_parts"]))
            section.pop("text_parts", None)

        return sections

    def extract_tables_as_text(self, html):
        soup = BeautifulSoup(html or "", "html.parser")
        table_texts = []

        for index, table in enumerate(soup.find_all("table"), start=1):
            rows = []

            for tr in table.find_all("tr"):
                cells = []

                for cell in tr.find_all(["th", "td"]):
                    text = self.clean_text(cell.get_text(" ", strip=True))

                    if text:
                        cells.append(text)

                if cells:
                    rows.append(" | ".join(cells))

            if rows:
                table_texts.append(
                    {
                        "table_id": f"table_{index:03d}",
                        "text": "\n".join(rows),
                    }
                )

        return table_texts

    def extract_expected_items_from_text(self, text):
        text = self.clean_text(text)

        if not text:
            return []

        parts = re.split(r"(?:\n|•|\u2022|;|\. )", text)
        items = []

        for part in parts:
            part = self.clean_text(part)

            if len(part) < 8:
                continue

            if part.lower() in {"yes", "no", "na", "n/a"}:
                continue

            items.append(part)

        deduped = []
        seen = set()

        for item in items:
            key = item.lower()

            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped[:30]

    def add_default_expectations_if_needed(self, topic, items):
        topic_lower = topic.lower()
        final_items = list(items or [])

        for key, defaults in DEFAULT_EXPECTATIONS_BY_TOPIC.items():
            if key in topic_lower:
                for item in defaults:
                    if item not in final_items:
                        final_items.append(item)

        return final_items[:40]

    def extract_template_expectations(self, html):
        sections = self.parse_sections(html)
        tables = self.extract_tables_as_text(html)

        expectations = []

        for section in sections:
            topic = section.get("heading", "")
            raw_text = section.get("text", "")
            items = self.extract_expected_items_from_text(raw_text)
            items = self.add_default_expectations_if_needed(topic, items)

            expectations.append(
                {
                    "topic": topic,
                    "expected_details": items,
                    "raw_text": raw_text,
                    "source": "section",
                }
            )

        for table in tables:
            raw_text = table["text"]
            items = self.extract_expected_items_from_text(raw_text)

            if items:
                expectations.append(
                    {
                        "topic": f"Template Table {table['table_id']}",
                        "expected_details": items,
                        "raw_text": raw_text,
                        "source": "table",
                    }
                )

        return expectations

    def nearest_heading_for_node(self, node):
        previous = node

        while previous:
            previous = previous.find_previous()

            if previous and previous.name and previous.name.lower() in self.HEADING_TAGS:
                return self.clean_text(previous.get_text(" ", strip=True))

        return "Unmapped Diagram"

    def surrounding_text_for_node(self, node):
        heading_node = None
        previous = node

        while previous:
            previous = previous.find_previous()

            if previous and previous.name and previous.name.lower() in self.HEADING_TAGS:
                heading_node = previous
                break

        parts = []

        if heading_node:
            cursor = heading_node.find_next_sibling()

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
            for prev in node.find_all_previous(["p", "li", "td", "th"], limit=10):
                text = self.clean_text(prev.get_text(" ", strip=True))

                if text:
                    parts.append(text)

        return self.clean_text(" ".join(reversed(parts[-12:])))

    def find_attachment_filename_in_node(self, node):
        nodes_to_check = [node] + list(node.find_all(True))

        for current in nodes_to_check:
            attrs = current.attrs or {}

            for key, value in attrs.items():
                key_lower = str(key).lower()

                if key_lower.endswith("filename") or "filename" in key_lower:
                    if value:
                        return str(value)

            current_name = current.name or ""

            if current_name.lower().endswith("attachment"):
                for value in attrs.values():
                    if value:
                        return str(value)

        return None

    def find_macro_name(self, node):
        attrs = node.attrs or {}

        for key, value in attrs.items():
            key_lower = str(key).lower()

            if key_lower.endswith("name") or key_lower == "name":
                return str(value)

        return ""

    def find_macro_parameters(self, node):
        params = {}

        for child in node.find_all(True):
            name = child.name or ""

            if "parameter" not in name.lower():
                continue

            param_name = None

            for key, value in (child.attrs or {}).items():
                if str(key).lower().endswith("name"):
                    param_name = str(value)
                    break

            if param_name:
                params[param_name] = self.clean_text(child.get_text(" ", strip=True))

        return params

    def normalize_diagram_key(self, diagram):
        filename = diagram.get("attachment_filename") or ""
        src = diagram.get("image_src") or ""
        diagram_type = diagram.get("diagram_type") or ""
        macro_name = diagram.get("macro_name") or ""
        heading = diagram.get("heading") or ""

        key = f"{diagram_type}|{macro_name}|{filename}|{src}|{heading}"
        return key.strip().lower()

    def extract_diagrams_from_html(self, html, source_name):
        soup = BeautifulSoup(html or "", "html.parser")
        diagrams = []

        for node in soup.find_all(True):
            tag = (node.name or "").lower()

            is_html_img = tag == "img"
            is_ac_image = tag.endswith("image")
            is_macro = tag.endswith("structured-macro") or "structured-macro" in tag

            if not is_html_img and not is_ac_image and not is_macro:
                continue

            diagram_type = None
            src = None
            filename = None
            macro_name = ""

            if is_html_img:
                diagram_type = "html_image"
                src = (
                    node.get("src")
                    or node.get("data-src")
                    or node.get("data-image-src")
                    or node.get("data-original-src")
                )

                filename = (
                    node.get("data-linked-resource-default-alias")
                    or node.get("data-linked-resource-id")
                    or node.get("data-attachment-name")
                    or node.get("alt")
                    or node.get("title")
                )

            if is_ac_image:
                diagram_type = "confluence_image"
                filename = self.find_attachment_filename_in_node(node)

            if is_macro:
                macro_name = self.find_macro_name(node)
                macro_name_lower = macro_name.lower()

                if not any(
                    keyword in macro_name_lower
                    for keyword in ["draw", "gliffy", "plantuml", "diagram", "mermaid"]
                ):
                    continue

                params = self.find_macro_parameters(node)
                diagram_type = f"macro_{macro_name_lower or 'unknown'}"

                filename = (
                    self.find_attachment_filename_in_node(node)
                    or params.get("diagramName")
                    or params.get("name")
                    or params.get("filename")
                    or params.get("attachment")
                    or params.get("page")
                )

            if not src and not filename and not macro_name:
                continue

            diagrams.append(
                {
                    "diagram_type": diagram_type,
                    "macro_name": macro_name,
                    "heading": self.nearest_heading_for_node(node),
                    "context_text": self.surrounding_text_for_node(node),
                    "image_src": src,
                    "attachment_filename": filename,
                    "html_source": source_name,
                }
            )

        return diagrams

    def extract_diagrams(self, storage_html, view_html=None):
        all_diagrams = []

        all_diagrams.extend(self.extract_diagrams_from_html(storage_html, "storage"))

        if view_html:
            all_diagrams.extend(self.extract_diagrams_from_html(view_html, "view"))

        final_diagrams = []
        seen = set()
        counter = 1

        for diagram in all_diagrams:
            key = self.normalize_diagram_key(diagram)

            if key in seen:
                continue

            seen.add(key)
            diagram["diagram_id"] = f"diagram_{counter:03d}"
            final_diagrams.append(diagram)
            counter += 1

        return final_diagrams


class GeminiAnalyzer:
    def __init__(self, model):
        self.model = model
        self.enabled = bool(genai and types)

        try:
            self.client = genai.Client() if self.enabled else None
        except Exception:
            self.client = None
            self.enabled = False

    def detect_mime_type(self, image_bytes):
        if not image_bytes:
            return None

        if image_bytes.startswith(b"\x89PNG"):
            return "image/png"

        if image_bytes.startswith(b"\xff\xd8"):
            return "image/jpeg"

        if image_bytes.startswith(b"GIF"):
            return "image/gif"

        if image_bytes[:4] == b"RIFF" and b"WEBP" in image_bytes[:20]:
            return "image/webp"

        return "image/png"

    def parse_json_response(self, text):
        if not text:
            return None

        value = text.strip()
        value = re.sub(r"^```json\s*", "", value)
        value = re.sub(r"^```\s*", "", value)
        value = re.sub(r"\s*```$", "", value)

        try:
            return json.loads(value)
        except Exception:
            pass

        match = re.search(r"\{.*\}", value, flags=re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

        return None

    def select_relevant_expectations(self, diagram, expectations):
        heading = diagram.get("heading", "").lower()
        context = diagram.get("context_text", "").lower()
        combined = f"{heading} {context}"

        matched = []

        for expectation in expectations:
            topic = expectation.get("topic", "")
            words = [
                word
                for word in re.findall(r"[a-zA-Z0-9]+", topic.lower())
                if len(word) > 3
            ]

            if any(word in combined for word in words):
                matched.append(expectation)

        if matched:
            return matched[:8]

        architecture_related = []

        for expectation in expectations:
            topic_lower = expectation.get("topic", "").lower()

            if any(
                key in topic_lower
                for key in [
                    "architecture",
                    "logical",
                    "functional",
                    "integration",
                    "deployment",
                    "security",
                    "database",
                    "network",
                    "environment",
                    "solution",
                    "schematic",
                    "diagram",
                ]
            ):
                architecture_related.append(expectation)

        return architecture_related[:10] if architecture_related else expectations[:10]

    def analyze(self, diagram, image_bytes, expectations):
        relevant_expectations = self.select_relevant_expectations(diagram, expectations)

        if not self.enabled:
            return self.heuristic_analysis(diagram, relevant_expectations)

        prompt = f"""
You are reviewing architecture/design diagrams from a Confluence evidence page.

Return only valid JSON. No markdown.

Evidence diagram:
{json.dumps(diagram, indent=2, ensure_ascii=False)}

Relevant template expectations:
{json.dumps(relevant_expectations, indent=2, ensure_ascii=False)[:35000]}

Tasks:
1. Explain the diagram based only on the image and nearby evidence text.
2. Identify visible architecture components.
3. Identify visible request flow, data flow, integration flow, or control flow.
4. Identify database, deployment, environment, network, integration, and security details only if visible or mentioned.
5. Compare evidence against template expectations.
6. For every important expected item, mark status as Present, Partially Present, Missing, Unclear, or Not Applicable.
7. Assess quality using clarity, labeling, flow direction, technical detail, consistency with nearby text, and review readiness.
8. Do not assume anything. If something is not visible or not mentioned, mark it Missing or Unclear.

Return JSON with this schema:
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
                mime_type = self.detect_mime_type(image_bytes)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
            )

            parsed = self.parse_json_response(response.text)

            if parsed:
                return parsed

            fallback = self.heuristic_analysis(diagram, relevant_expectations)
            fallback["diagram_explanation"] = response.text[:4000] if response.text else fallback["diagram_explanation"]
            fallback["analysis_warning"] = "Model response was not valid JSON, fallback structure used."
            return fallback

        except Exception as ex:
            fallback = self.heuristic_analysis(diagram, relevant_expectations)
            fallback["analysis_error"] = str(ex)
            return fallback

    def heuristic_analysis(self, diagram, expectations):
        heading = diagram.get("heading", "")
        context = diagram.get("context_text", "")
        combined = f"{heading} {context}".lower()

        checks = []

        for expectation in expectations:
            topic = expectation.get("topic", "")
            items = expectation.get("expected_details", [])

            if not items:
                items = [topic]

            for item in items[:8]:
                item_words = [
                    word
                    for word in re.findall(r"[a-zA-Z0-9]+", item.lower())
                    if len(word) >= 5
                ]

                matched_words = [word for word in item_words if word in combined]
                status = "Partially Present" if matched_words else "Unclear"

                checks.append(
                    {
                        "required_detail": f"{topic} - {item}",
                        "status": status,
                        "evidence_found": context[:700] if matched_words else "",
                        "missing_details": [] if matched_words else [item],
                        "remarks": "Heuristic check used because model analysis was unavailable.",
                    }
                )

        if not checks:
            checks.append(
                {
                    "required_detail": "Architecture/design details",
                    "status": "Unclear",
                    "evidence_found": context[:700],
                    "missing_details": [],
                    "remarks": "No relevant template expectations were extracted.",
                }
            )

        return {
            "diagram_explanation": (
                f"The diagram is mapped under heading '{heading}'. "
                "Image-level model explanation was unavailable, so this is based on heading and nearby text only."
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
                "strengths": ["Diagram or diagram reference was found in the evidence page."],
                "issues": ["Model image analysis was unavailable in current environment."],
                "recommendations": ["Check model authentication/environment setup if image-level explanation is expected."],
            },
            "confidence": "low",
        }


class ReportBuilder:
    def __init__(self, output_json, output_pdf):
        self.output_json = output_json
        self.output_pdf = output_pdf

    def completeness_score(self, topics):
        score_map = {
            "present": 1.0,
            "partially present": 0.5,
            "unclear": 0.25,
            "missing": 0.0,
        }

        total = 0
        achieved = 0

        for topic in topics:
            for check in topic.get("required_details_check", []):
                status = str(check.get("status", "")).strip().lower()

                if status == "not applicable":
                    continue

                total += 1
                achieved += score_map.get(status, 0)

        return round((achieved / total) * 100, 2) if total else 0

    def quality_score(self, topics):
        values = []

        for topic in topics:
            score = topic.get("quality", {}).get("overall_quality_score")

            try:
                values.append(float(score))
            except Exception:
                pass

        return round(sum(values) / len(values), 2) if values else 0

    def overall_status(self, completeness, quality):
        average = (completeness + quality) / 2

        if average >= 85:
            return "Excellent"

        if average >= 70:
            return "Good"

        if average >= 50:
            return "Needs Improvement"

        return "Poor"

    def escape(self, text):
        if text is None:
            return ""

        value = str(text)
        value = value.replace("&", "&amp;")
        value = value.replace("<", "&lt;")
        value = value.replace(">", "&gt;")

        return value

    def build_json(self, evidence_page, template_page, topics, output_dir):
        completeness = self.completeness_score(topics)
        quality = self.quality_score(topics)

        missing_items = []
        recommendations = []

        for topic in topics:
            for check in topic.get("required_details_check", []):
                status = str(check.get("status", "")).strip().lower()

                if status in {"missing", "partially present", "unclear"}:
                    missing_items.append(
                        {
                            "topic": topic.get("topic_name"),
                            "diagram_id": topic.get("diagram_id"),
                            "required_detail": check.get("required_detail"),
                            "status": check.get("status"),
                            "evidence_found": check.get("evidence_found"),
                            "missing_details": check.get("missing_details", []),
                            "remarks": check.get("remarks", ""),
                        }
                    )

            for item in topic.get("quality", {}).get("recommendations", []):
                recommendations.append(
                    {
                        "topic": topic.get("topic_name"),
                        "diagram_id": topic.get("diagram_id"),
                        "recommendation": item,
                    }
                )

        report = {
            "metadata": {
                "evidence_page_title": evidence_page.get("title", ""),
                "template_page_title": template_page.get("title", ""),
                "evidence_page_url": evidence_page.get("_source_url", ""),
                "template_page_url": template_page.get("_source_url", ""),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": self.overall_status(completeness, quality),
                "overall_completeness_score": completeness,
                "overall_quality_score": quality,
            },
            "topics": topics,
            "overall_missing_items": missing_items,
            "overall_recommendations": recommendations,
        }

        output_path = Path(output_dir) / self.output_json

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        return report, output_path

    def pdf_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size

        page_width, page_height = landscape(A4)

        max_width = page_width - 80
        max_height = page_height - 120

        ratio = min(max_width / width, max_height / height)

        display_width = width * ratio
        display_height = height * ratio

        return PdfImage(image_path, width=display_width, height=display_height)

    def build_pdf(self, report, output_dir):
        output_path = Path(output_dir) / self.output_pdf

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(A4),
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )

        styles = getSampleStyleSheet()

        styles.add(
            ParagraphStyle(
                name="SmallText",
                parent=styles["BodyText"],
                fontSize=8,
                leading=10,
            )
        )

        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                spaceAfter=8,
            )
        )

        story = []
        metadata = report.get("metadata", {})

        story.append(Paragraph("Architecture Diagram Review Report", styles["Title"]))
        story.append(Spacer(1, 12))

        summary_rows = [
            ["Evidence Page", self.escape(metadata.get("evidence_page_title", ""))],
            ["Template Page", self.escape(metadata.get("template_page_title", ""))],
            ["Analysis Date", self.escape(metadata.get("analysis_date", ""))],
            ["Overall Status", self.escape(metadata.get("overall_status", ""))],
            ["Completeness Score", str(metadata.get("overall_completeness_score", ""))],
            ["Quality Score", str(metadata.get("overall_quality_score", ""))],
        ]

        summary_table = Table(summary_rows, colWidths=[2.0 * inch, 7.0 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 18))

        story.append(Paragraph("Overall Missing / Unclear Items", styles["SectionTitle"]))

        missing_items = report.get("overall_missing_items", [])

        if missing_items:
            rows = [["Topic", "Required Detail", "Status", "Missing / Remarks"]]

            for item in missing_items[:70]:
                missing_text = "; ".join(item.get("missing_details", [])) or item.get("remarks", "")

                rows.append(
                    [
                        Paragraph(self.escape(item.get("topic", "")), styles["SmallText"]),
                        Paragraph(self.escape(item.get("required_detail", "")), styles["SmallText"]),
                        Paragraph(self.escape(item.get("status", "")), styles["SmallText"]),
                        Paragraph(self.escape(missing_text), styles["SmallText"]),
                    ]
                )

            table = Table(rows, colWidths=[1.7 * inch, 3.0 * inch, 1.2 * inch, 3.6 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ]
                )
            )

            story.append(table)
        else:
            story.append(Paragraph("No missing or unclear items were identified.", styles["BodyText"]))

        story.append(PageBreak())

        for topic in report.get("topics", []):
            story.append(Paragraph(self.escape(topic.get("topic_name", "")), styles["Heading1"]))
            story.append(Paragraph(f"Diagram ID: {self.escape(topic.get('diagram_id', ''))}", styles["SmallText"]))
            story.append(Paragraph(f"Diagram Type: {self.escape(topic.get('diagram_type', ''))}", styles["SmallText"]))
            story.append(Paragraph(f"HTML Source: {self.escape(topic.get('html_source', ''))}", styles["SmallText"]))
            story.append(Paragraph(f"Diagram Available: {self.escape(topic.get('diagram_available', ''))}", styles["SmallText"]))
            story.append(Spacer(1, 8))

            image_path = topic.get("local_image_path")

            if image_path and os.path.exists(image_path):
                try:
                    story.append(PageBreak())
                    story.append(Paragraph("Diagram", styles["SectionTitle"]))
                    story.append(self.pdf_image(image_path))
                    story.append(PageBreak())
                except Exception:
                    story.append(
                        Paragraph(
                            "Diagram image could not be rendered in PDF.",
                            styles["SmallText"],
                        )
                    )

            story.append(Paragraph("Diagram Explanation", styles["SectionTitle"]))
            story.append(Paragraph(self.escape(topic.get("diagram_explanation", "")), styles["BodyText"]))
            story.append(Spacer(1, 10))

            story.append(Paragraph("Completeness Check", styles["SectionTitle"]))

            checks = topic.get("required_details_check", [])

            if checks:
                rows = [["Required Detail", "Status", "Evidence Found", "Missing / Remarks"]]

                for check in checks[:50]:
                    missing_text = "; ".join(check.get("missing_details", [])) or check.get("remarks", "")

                    rows.append(
                        [
                            Paragraph(self.escape(check.get("required_detail", "")), styles["SmallText"]),
                            Paragraph(self.escape(check.get("status", "")), styles["SmallText"]),
                            Paragraph(self.escape(check.get("evidence_found", "")), styles["SmallText"]),
                            Paragraph(self.escape(missing_text), styles["SmallText"]),
                        ]
                    )

                check_table = Table(rows, colWidths=[2.4 * inch, 1.2 * inch, 3.0 * inch, 2.9 * inch])
                check_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCFCE7")),
                            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ]
                    )
                )

                story.append(check_table)
            else:
                story.append(Paragraph("No completeness checks were generated.", styles["BodyText"]))

            story.append(Spacer(1, 12))
            story.append(Paragraph("Quality Review", styles["SectionTitle"]))

            quality = topic.get("quality", {})

            quality_rows = [
                ["Overall Quality Score", str(quality.get("overall_quality_score", ""))],
                ["Rating", self.escape(quality.get("rating", ""))],
                ["Strengths", self.escape("; ".join(quality.get("strengths", [])))],
                ["Issues", self.escape("; ".join(quality.get("issues", [])))],
                ["Recommendations", self.escape("; ".join(quality.get("recommendations", [])))],
            ]

            quality_table = Table(quality_rows, colWidths=[2.0 * inch, 7.0 * inch])
            quality_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FEF3C7")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]
                )
            )

            story.append(quality_table)
            story.append(PageBreak())

        doc.build(story)

        return output_path


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
    image_source = None

    if filename:
        image_bytes = client.download_attachment_by_name(page, attachments, filename)

        if image_bytes:
            image_source = f"attachment:{filename}"

    if not image_bytes and src:
        image_bytes = client.download_url(page, src)

        if image_bytes:
            image_source = f"url:{src}"

    return image_bytes, image_source


def add_missing_image_attachments_as_diagrams(diagrams, attachments):
    existing_files = set()

    for diagram in diagrams:
        filename = diagram.get("attachment_filename")

        if filename:
            existing_files.add(str(filename).strip().lower())

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".svg",
    }

    next_number = len(diagrams) + 1

    for attachment in attachments:
        title = str(attachment.get("title", "")).strip()

        if not title:
            continue

        title_lower = title.lower()

        if title_lower in existing_files:
            continue

        media_type = str(attachment.get("metadata", {}).get("mediaType", "")).lower()
        is_image_by_extension = any(title_lower.endswith(ext) for ext in image_extensions)
        is_image_by_media_type = media_type.startswith("image/")

        if not is_image_by_extension and not is_image_by_media_type:
            continue

        diagrams.append(
            {
                "diagram_id": f"diagram_{next_number:03d}",
                "diagram_type": "image_attachment_fallback",
                "macro_name": "",
                "heading": "Image Attachment",
                "context_text": "",
                "image_src": None,
                "attachment_filename": title,
                "html_source": "attachment_fallback",
            }
        )

        existing_files.add(title_lower)
        next_number += 1

    return diagrams


def build_topics(evidence_page, expectations, diagrams, analyzer, client, attachments, output_dir):
    topics = []

    for diagram in diagrams:
        image_bytes, image_source = resolve_diagram_image(client, evidence_page, attachments, diagram)
        local_image_path = save_image_bytes(image_bytes, output_dir, diagram.get("diagram_id"))

        analysis = analyzer.analyze(diagram, image_bytes, expectations)
        checks = analysis.get("completeness_checks", [])

        missing = [
            check
            for check in checks
            if str(check.get("status", "")).lower() in {"missing", "partially present", "unclear"}
        ]

        topic = {
            "topic_name": diagram.get("heading", "Unmapped Diagram"),
            "diagram_id": diagram.get("diagram_id"),
            "diagram_type": diagram.get("diagram_type"),
            "macro_name": diagram.get("macro_name"),
            "html_source": diagram.get("html_source"),
            "diagram_heading": diagram.get("heading"),
            "diagram_available": bool(image_bytes),
            "diagram_image_source": image_source,
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
            "required_details_check": checks,
            "missing_expected_details": missing,
            "quality": analysis.get("quality", {}),
            "confidence": analysis.get("confidence", ""),
        }

        if "analysis_error" in analysis:
            topic["analysis_error"] = analysis["analysis_error"]

        if "analysis_warning" in analysis:
            topic["analysis_warning"] = analysis["analysis_warning"]

        topics.append(topic)

    if not topics:
        topics.append(
            {
                "topic_name": "No Diagram Found",
                "diagram_id": "diagram_000",
                "diagram_type": "none",
                "macro_name": "",
                "html_source": "",
                "diagram_heading": "",
                "diagram_available": False,
                "diagram_image_source": None,
                "diagram_attachment_filename": None,
                "diagram_image_src": None,
                "local_image_path": None,
                "nearby_evidence_text": "",
                "diagram_explanation": "No diagram/image/macro diagram was detected in the evidence Confluence page.",
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
                        "missing_details": [
                            "No supported diagram was extracted from the evidence page."
                        ],
                        "remarks": "Check evidence_storage.html, evidence_view.html and evidence_attachments.json to confirm diagram storage format.",
                    }
                ],
                "missing_expected_details": [
                    {
                        "required_detail": "Architecture or design diagram",
                        "status": "Missing",
                        "missing_details": [
                            "No supported diagram was extracted from the evidence page."
                        ],
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
                    "issues": ["No diagram was found."],
                    "recommendations": [
                        "Add architecture/design diagrams to the evidence page or extend parser for the exact macro type used."
                    ],
                },
                "confidence": "high",
            }
        )

    return topics


def save_debug_file(path, content):
    with open(path, "w", encoding="utf-8") as file:
        if isinstance(content, str):
            file.write(content)
        else:
            json.dump(content, file, indent=2, ensure_ascii=False)


def get_required_env(name):
    value = os.getenv(name)

    if not value:
        print(f"Missing environment variable: {name}", file=sys.stderr)
        print(f"Set it like this on Windows: set {name}=your_value", file=sys.stderr)
        sys.exit(1)

    return value


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--evidence-url", required=True)
    cli.add_argument("--template-url", required=True)
    cli.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    cli.add_argument("--output-dir", default="confluence_review_output")
    cli.add_argument("--output-json", default="architecture_review_report.json")
    cli.add_argument("--output-pdf", default="architecture_review_report.pdf")

    args = cli.parse_args()

    confluence_pat = get_required_env(CONFLUENCE_PAT_ENV_NAME)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ConfluenceClient(confluence_pat)
    parser = ConfluenceParser()
    analyzer = GeminiAnalyzer(args.gemini_model)
    report_builder = ReportBuilder(args.output_json, args.output_pdf)

    print("Fetching evidence page...")
    evidence_page = client.fetch_page(args.evidence_url)

    print("Fetching template page...")
    template_page = client.fetch_page(args.template_url)

    evidence_storage_html = parser.get_storage_html(evidence_page)
    evidence_view_html = parser.get_view_html(evidence_page)
    template_storage_html = parser.get_storage_html(template_page)
    template_view_html = parser.get_view_html(template_page)

    evidence_html = evidence_storage_html + "\n" + evidence_view_html
    template_html = template_storage_html + "\n" + template_view_html

    save_debug_file(output_dir / "evidence_page_raw.json", evidence_page)
    save_debug_file(output_dir / "template_page_raw.json", template_page)
    save_debug_file(output_dir / "evidence_storage.html", evidence_storage_html)
    save_debug_file(output_dir / "evidence_view.html", evidence_view_html)
    save_debug_file(output_dir / "template_storage.html", template_storage_html)
    save_debug_file(output_dir / "template_view.html", template_view_html)

    print("Fetching evidence attachments...")
    evidence_attachments = client.fetch_attachments(evidence_page)
    save_debug_file(output_dir / "evidence_attachments.json", evidence_attachments)

    print("Extracting diagrams from storage and view HTML...")
    diagrams = parser.extract_diagrams(evidence_storage_html, evidence_view_html)

    print("Adding missing image attachments as fallback diagrams...")
    diagrams = add_missing_image_attachments_as_diagrams(diagrams, evidence_attachments)

    save_debug_file(output_dir / "extracted_diagrams.json", diagrams)

    print("Extracting template expectations...")
    expectations = parser.extract_template_expectations(template_html)
    save_debug_file(output_dir / "template_expectations.json", expectations)

    print("Analyzing diagrams...")
    topics = build_topics(
        evidence_page=evidence_page,
        expectations=expectations,
        diagrams=diagrams,
        analyzer=analyzer,
        client=client,
        attachments=evidence_attachments,
        output_dir=output_dir,
    )

    print("Generating JSON report...")
    report, json_path = report_builder.build_json(
        evidence_page=evidence_page,
        template_page=template_page,
        topics=topics,
        output_dir=output_dir,
    )

    print("Generating PDF report...")
    pdf_path = report_builder.build_pdf(report, output_dir)

    print("")
    print("Done.")
    print(f"JSON report: {json_path}")
    print(f"PDF report: {pdf_path}")
    print(f"Evidence raw page: {output_dir / 'evidence_page_raw.json'}")
    print(f"Template raw page: {output_dir / 'template_page_raw.json'}")
    print(f"Evidence storage HTML: {output_dir / 'evidence_storage.html'}")
    print(f"Evidence view HTML: {output_dir / 'evidence_view.html'}")
    print(f"Template storage HTML: {output_dir / 'template_storage.html'}")
    print(f"Template view HTML: {output_dir / 'template_view.html'}")
    print(f"Evidence attachments: {output_dir / 'evidence_attachments.json'}")
    print(f"Extracted diagrams: {output_dir / 'extracted_diagrams.json'}")
    print(f"Template expectations: {output_dir / 'template_expectations.json'}")
    print(f"Diagrams found: {len(diagrams)}")
    print(f"Model name: {args.gemini_model}")
    print(f"Model enabled: {analyzer.enabled}")
    print(f"SSL verify: {SSL_VERIFY}")
    print("Authorization mode: Bearer token from CONFLUENCE_PAT env variable")


if __name__ == "__main__":
    main()
